add_library(htlib HoughTransform.cu)

add_executable(lanedet
    LaneDetection.cu
    Line.cpp
    Preprocessing.cpp)
set_target_properties(lanedet
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(lanedet htlib ${libs})