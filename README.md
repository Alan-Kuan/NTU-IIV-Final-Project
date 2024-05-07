# NTU IIV Final Project
This work originates from the author of the repo [cuda-lane-detection](https://github.com/jonaspfab/cuda-lane-detection).
We forked and optimized it by different means, such as reducing the region to process and distributing the workload to 2 GPU devices.

## Dependencies
- CMake
- Make / Ninja
- OpenCV
- NCCL

## Build
We chose CMake to configure the building environment.
You can build the project with preferred build system.

Make:
```sh
mkdir build
cmake -B build
make -C build
```

or

Ninja:
```sh
mkdir build
cmake -B build -G Ninja
ninja -C build
```

## Usage
The executable is generated in the directory `build`.

```
Usage: ./build/lanedet inputVideo outputVideo [options]

 inputVideo    Input video for which lanes are detected
 outputVideo   Name of resulting output video

Options:
 --seq         Perform hough transform sequentially on the CPU (if omitted, CUDA is used)
 --ss=<num>    How to split the frame (default: 0, should not be used when --nd=1)
   0           no split
   1           left half & right half
   2           top half & bottom half
   3           cyclic split from left to right
   4           cyclic split from top to bottom
 --nd=<num>    Number of GPU devices (default: 1)
```

Example:
```sh
./build/lanedet test-video.mp4 output.avi --nd=2 --ss=1
```

## License
The file `cmake/modules/FindNCCL.cmake` came from the repo [dmlc/xgboost](https://github.com/dmlc/xgboost/blob/master/cmake/modules/FindNccl.cmake),
which is licensed under **the Apache License, Version 2.0**.
