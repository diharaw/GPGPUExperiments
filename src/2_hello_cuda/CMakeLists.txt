cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

find_package(CUDA REQUIRED)

set(HELLO_CUDA_SOURCE ${PROJECT_SOURCE_DIR}/src/2_hello_cuda/hello_cuda.cpp)
cuda_add_executable(2_hello_cuda ${HELLO_CUDA_SOURCE})