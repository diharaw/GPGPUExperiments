cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(OpenCL REQUIRED)

set(HELLO_OPENCL_SOURCE ${PROJECT_SOURCE_DIR}/src/1_hello_opencl/hello_opencl.cpp)

include_directories("${OpenCL_INCLUDE_DIRS}")

add_executable(1_hello_opencl ${HELLO_OPENCL_SOURCE})				
target_link_libraries(1_hello_opencl ${OpenCL_LIBRARIES})