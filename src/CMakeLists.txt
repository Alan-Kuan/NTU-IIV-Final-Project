add_library(htlib HoughTransform.cu)
add_library(handle Handle.cu)

add_executable(lanedet
    LaneDetection.cpp
    Line.cpp
    Preprocessing.cpp)
set_target_properties(lanedet
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(lanedet handle htlib ${libs})