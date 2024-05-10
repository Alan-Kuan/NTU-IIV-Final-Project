#ifndef HANDLE_H
#define HANDLE_H

#include <cstddef>

#include <cuda_runtime.h>
#include <nccl.h>

#define THETA_STEP_SIZE 0.1
#define RHO_STEP_SIZE 2
#define THRESHOLD 125
#define THETA_A 45.0
#define THETA_B 135.0
#define THETA_VARIATION 16.0
#define MAX_NUM_LINES 10

enum HoughStrategy { kSeq, kCuda };
enum SplitStrategy {
    kNone,
    kLeftRight,
    kTopBottom,
    kLeftRightCyclic,
    kTopBottomCyclic,
};

const char *const splitStrategyName[] = {
    "no split",
    "left half & right half",
    "top half & bottom half",
    "cyclic split from left to right",
    "cyclic split from top to bottom"
};

/** 
 * Handle which tracks info that is required for every execution of hough 
 * transform. The handle allows the hough transforms to reuse variables between
 * executions of different frames.
 */
struct HoughTransformHandle {
    int frameWidth;  // actual frame width, for p_frame
    // additional param in order to crop roi
    int roiFrameWidth;
    int roiFrameHeight;
    int roiStartX;
    int roiStartY;
    /// Number of rows in the accumulator (number of possible rhos)
    int nRows;
    /// Number of columns in the accumulator (number of possible thetas)
    int nCols;
};

/**
 * Tracks accumulator required for sequential execution. Memory space for the 
 * accumulator needs to be allocated only once for all frames.
 */
struct SeqHandle: HoughTransformHandle {
    /// 2D matrix where votes for combinations of rho/theta values are tracked
    int *accumulator;
};

/**
 * Tracks info required for the CUDA hough transform. Memory space on host as 
 * well as device only needs to be allocated only once for all frames.
 */
struct CudaHandle: HoughTransformHandle {
    // attributes
    int nDevs;
    SplitStrategy splitStrategy;
    int roiFrameSize;
    int roiStart;
    int *frameOffset;  // for copying roi frame
    size_t accCount, accSize;
    size_t linesSize;

    // buffers
    int **lines;
    int **d_lines;
    int *lineCounter;
    int **d_lineCounter;
    unsigned char *p_frame;  // global size
    unsigned char **d_frame;  // local roi size
    int **d_accumulator;

    // nccl
    ncclComm_t *comms;
    int *devs;

    // kernel params
    dim3 houghBlockDim;
    dim3 houghGridDim;
    dim3 findLinesBlockDim;
    dim3 findLinesGridDim;
};

/**
 * Initializes handle object for given hough strategy
 * 
 * @param handle Handle to be initialized
 * @param houghStrategy Strategy used to perform hough transform
 * @param splitStrategy
 * @param nDevs Number of GPU devices
 */
void createHandle(HoughTransformHandle *&handle, int frameWidth, int frameHeight,
    HoughStrategy houghStrategy, SplitStrategy splitStrategy, int nDevs);

/**
 * Frees memory on host and device that was allocated for the handle
 * 
 * @param handle Handle to be destroyed
 * @param houghStrategy Hough strategy that was used to create the handle
 */
void destroyHandle(HoughTransformHandle *&handle, HoughStrategy houghStrategy);

#endif  // HANDLE_H