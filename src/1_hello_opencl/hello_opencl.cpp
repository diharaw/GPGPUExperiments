#if defined(__APPLE__)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#include <iostream>
#include <vector>

int main()
{
	std::vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	for (int i = 0; i < platforms.size(); i++)
	{
		std::cout << "Platform " << i << " : " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
	}

	cl::Platform platform = platforms[0];

	std::vector<cl::Device> devices;

	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	for (int i = 0; i < devices.size(); i++)
	{
		std::cout << "Device " << i << " : " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
	}

	cl::Device device = devices[0];

	cl::Context context = cl::Context(device);

	const char* source_string =
		" __kernel void parallel_add(__global float* x, __global float* y, __global float* z){ "
		" const int i = get_global_id(0); " // get a unique number identifying the work item in the global pool
		" z[i] = y[i] + x[i];    " // add two arrays 
		"}";

	cl::Program program = cl::Program(context, source_string);

	cl_int result = program.build({ device }, "");

	if (result)
		std::cout << "Failed to compiled program: " << result << std::endl;

	cl::Kernel kernel = cl::Kernel(program, "parallel_add");

	const int numElements = 10;
	float cpuArrayX[numElements] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
	float cpuArrayY[numElements] = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f };
	float cpuOutput[numElements] = {};

	cl::Buffer clBufferX = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numElements * sizeof(cl_int), cpuArrayX);
	cl::Buffer clBufferY = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, numElements * sizeof(cl_int), cpuArrayY);
	cl::Buffer clBufferOutput = cl::Buffer(context, CL_MEM_WRITE_ONLY, numElements * sizeof(cl_int), NULL);

	kernel.setArg(0, clBufferX);
	kernel.setArg(1, clBufferY);
	kernel.setArg(2, clBufferOutput);

	cl::CommandQueue queue = cl::CommandQueue(context, device);

	size_t global_work_size = numElements;
	size_t local_work_size = 1;

	queue.enqueueNDRangeKernel(kernel, NULL, global_work_size, local_work_size);
	queue.enqueueReadBuffer(clBufferOutput, CL_TRUE, 0, numElements * sizeof(cl_float), cpuOutput);

	for (int i = 0; i < numElements; i++)
		std::cout << cpuArrayX[i] << " + " << cpuArrayY[i] << " = " << cpuOutput[i] << std::endl;

	std::cin.get();

	return 0;
}