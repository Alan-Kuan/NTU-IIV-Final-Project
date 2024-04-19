cmake_minimum_required(VERSION 3.10)
project(lanedet LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "-std=c++11")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_50,code=sm_50)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Werror")
set(CMAKE CXX_STANDARD_REQUIRED True)
set(libs
    opencv_core
    opencv_imgproc
    opencv_videoio)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

include_directories(include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})

add_subdirectory(src)