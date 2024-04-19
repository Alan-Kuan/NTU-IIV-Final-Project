CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(lanedet LANGUAGES CXX CUDA)
ENABLE_LANGUAGE(CUDA)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wno-unknown-pragmas -Werror")
SET(CMAKE CXX_STANDARD_REQUIRED True)
SET(CMAKE_CUDA_FLAGS "-std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_50,code=sm_50)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
SET(libs
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_videoio)

INCLUDE_DIRECTORIES(include
                    ${OpenCV_INCLUDE_DIRS}
                    ${CUDA_INCLUDE_DIRS})

ADD_LIBRARY(htlib HoughTransform.cu)

ADD_EXECUTABLE(lanedet
               LaneDetection.cpp
               Line.cpp
               Preprocessing.cpp)
SET_TARGET_PROPERTIES(lanedet PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(lanedet htlib ${libs})
