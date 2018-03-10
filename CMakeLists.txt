cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project("GPGPUExperiments")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Experiments
add_subdirectory(src/1_hello_opencl)
add_subdirectory(src/2_hello_cuda)