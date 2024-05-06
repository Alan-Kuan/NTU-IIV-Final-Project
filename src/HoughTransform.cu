#include "HoughTransform.hpp"

#include <iostream>

#include <cmath>
#include <cstring>
#include <vector>

#include <nccl.h>

#include "Line.hpp"

#define THETA_STEP_SIZE 0.1
#define RHO_STEP_SIZE 2
#define THRESHOLD 125
#define THETA_A 45.0
#define THETA_B 135.0
#define THETA_VARIATION 16.0
#define MAX_NUM_LINES 10

/**
 * Calculates rho based on the equation r = x cos(θ) + y sin(θ)
 *
 * @param x X coordinate of the pixel
 * @param y Y coordinate of the pixel
 * @param theta Angle between x axis and line connecting origin with closest
 * point on tested line
 *
 * @return Rho describing distance of origin to closest point on tested line
 */
__host__ __device__ double calcRho(double x, double y, double theta) {
    double thetaRadian = (theta * M_PI) / 180.0;
    double sinVal, cosVal;

    sincos(thetaRadian, &sinVal, &cosVal);
    return x * cosVal + y * sinVal;
}

/**
 * Calculates index in accumulator for given parameters
 *
 * @param nRows Number of rows in the accumulator (possible rho values)
 * @param nCols Number of columns in the accumulator (possibel theta values)
 * @param rho Rho value for determining row in accumulator
 * @param theta Theta value for determining column in accumulator
 */
__host__ __device__ int index(int nRows, int nCols, int rho, double theta) {
    return ((rho / RHO_STEP_SIZE) + (nRows / 2)) * nCols + 
            (int) ((theta - (THETA_A-THETA_VARIATION)) / THETA_STEP_SIZE + 0.5);
}

/**
 * Checks whether value at i and j is a local maximum
 *
 * In order to only find the local maximum all surrounding values are checked if
 * they are bigger
 */
 __host__ __device__ bool isLocalMaximum(int i, int j, int nRows, int nCols, int *accumulator) {
    for (int i_delta = -50; i_delta <= 50; i_delta++) {
        for (int j_delta = -50; j_delta <= 50; j_delta++) {
            if (i + i_delta > 0 && i + i_delta < nRows && j + j_delta > 0 && j + j_delta < nCols &&
                accumulator[(i + i_delta) * nCols + j + j_delta] > accumulator[i * nCols + j]) {
                return false;
            }
        }
    }

    return true;
}

/**
 * Performs hough transform for given frame sequentially and adds found lines
 * in 'lines' vector
 *
 * @param handle Handle tracking relevant info accross executions
 * @param frame Video frame on which hough transform is applied
 * @param lines Vector to which found lines are added to
 */
void houghTransformSeq(HoughTransformHandle *handle, cv::Mat frame, std::vector<Line> &lines) {
    SeqHandle *h = (SeqHandle *) handle;

    std::memset(h->accumulator, 0, h->nCols * h->nRows * sizeof(int));
    int rho;
    double theta;

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            int new_i = i + h->roiStartY;
            int new_j = j + h->roiStartX;

            if ((int) frame.at<unsigned char>(new_i, new_j) == 0)
                continue;

            // thetas of interest will be close to 45 and close to 135 (vertical lines)
            // we are doing 2 thetas at a time, 1 for each theta of Interest
            // we use thetas varying 15 degrees more and less
            for (int k = 0; k < 2 * THETA_VARIATION * (1 / THETA_STEP_SIZE); k++) {
                theta = THETA_A - THETA_VARIATION + ((double) k * THETA_STEP_SIZE);
                rho = calcRho(new_j, new_i, theta);
                h->accumulator[index(h->nRows, h->nCols, rho, theta)] += 1;

                theta = THETA_B-THETA_VARIATION + ((double) k * THETA_STEP_SIZE);
                rho = calcRho(new_j, new_i, theta);
                h->accumulator[index(h->nRows, h->nCols, rho, theta)] += 1;
            }
        }
    }

    // Find lines
    for (int i = 0; i < h->nRows; i++) {
        for (int j = 0; j < h->nCols; j++) {
            if (h->accumulator[i * h->nCols + j] >= THRESHOLD &&
                isLocalMaximum(i, j, h->nRows, h->nCols, h->accumulator))
                lines.push_back(Line((THETA_A-THETA_VARIATION) + (j * THETA_STEP_SIZE), (i - (h->nRows / 2)) * RHO_STEP_SIZE));
        }
    }
}

/**
 * CUDA kernel responsible for trying all different rho/theta combinations for
 * non-zero pixels and adding votes to accumulator
 */
__global__ void houghKernel(int roiFrameWidth, int roiFrameHeight, unsigned char* roiFrame, int roiStartX, int roiStartY,
        int nRows, int nCols, int *accumulator, int dev, SplitStrategy splitStrategy) {
    int i = blockIdx.x * blockDim.y + threadIdx.y;
    int j = blockIdx.y * blockDim.z + threadIdx.z;
    double theta;
    int rho;

    if (i < roiFrameHeight && j < roiFrameWidth && ((int) roiFrame[(i * roiFrameWidth) + j]) != 0) {
        int x = j + roiStartX;
        int y = i + roiStartY;
        switch (splitStrategy) {
        case SplitStrategy::kLeftRight:
            x += dev * roiFrameWidth;
            break;
        case SplitStrategy::kTopBottom:
            y += dev * roiFrameHeight;
            break;
        }

        // thetas of interest will be close to 45 and close to 135 (vertical lines)
        // we are doing 2 thetas at a time, 1 for each theta of Interest
        // we use thetas varying 15 degrees more and less
        for (int k = threadIdx.x * (1 / THETA_STEP_SIZE); k < (threadIdx.x + 1) * (1 / THETA_STEP_SIZE); k++) {
            theta = THETA_A-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(x, y, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);

            theta = THETA_B-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(x, y, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);
        }
    }
}

/**
 * CUDA kernel responsible for finding lines based on the number of votes for
 * every rho/theta combination
 */
__global__ void findLinesKernel(int nRows, int nCols, int *accumulator, int *lines, int *lineCounter, int dev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y + dev * (nCols / 2);

    if (accumulator[i * nCols + j] >= THRESHOLD && isLocalMaximum(i, j, nRows, nCols, accumulator)) {
        int insertPt = atomicAdd(lineCounter, 2);
        if (insertPt + 1 < 2 * MAX_NUM_LINES) {
            lines[insertPt] = THETA_A-THETA_VARIATION + (j * THETA_STEP_SIZE);
            lines[insertPt + 1] = (i - (nRows / 2)) * RHO_STEP_SIZE;
        }
    }
}

/**
 * Performs hough transform for given frame using CUDA and adds found lines
 * in 'lines' vector
 *
 * @param handle Handle tracking relevant info accross executions
 * @param frame Video frame on which hough transform is applied
 * @param lines Vector to which found lines are added to
 */
void houghTransformCuda(HoughTransformHandle *handle, cv::Mat frame, std::vector<Line> &lines) {
    CudaHandle *h = (CudaHandle *) handle;
    size_t accCount = h->nRows * h->nCols;
    size_t accSize = accCount * sizeof(int);
    size_t linesCount = 2 * MAX_NUM_LINES;
    size_t linesSize = linesCount * sizeof(int);

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemcpy2D(h->d_frame[dev], h->roiFrameWidth, frame.ptr() + h->frameOffset[dev], frame.cols,
            h->roiFrameWidth, h->roiFrameHeight, cudaMemcpyHostToDevice);
        cudaMemset(h->d_accumulator[dev], 0, accSize);
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        houghKernel<<<h->houghGridDim, h->houghBlockDim>>>(h->roiFrameWidth, h->roiFrameHeight, h->d_frame[dev],
            h->roiStartX, h->roiStartY, h->nRows, h->nCols, h->d_accumulator[dev], dev, h->splitStrategy);
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    if (h->splitStrategy != SplitStrategy::kNone) {
        ncclGroupStart();
        for (int dev = 0; dev < h->nDevs; dev++) {
            ncclAllReduce(h->d_accumulator[dev], h->d_accumulator[dev], accCount,
                ncclInt, ncclSum, h->comms[dev], cudaStreamDefault);
        }
        ncclGroupEnd();
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemset(h->d_lineCounter[dev], 0, sizeof(int));
        findLinesKernel<<<h->findLinesGridDim, h->findLinesBlockDim>>>(h->nRows, h->nCols,
            h->d_accumulator[dev], h->d_lines[dev], h->d_lineCounter[dev], dev);
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();

        cudaMemcpy(&h->lineCounter, h->d_lineCounter[dev], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h->lines, h->d_lines[dev], linesSize, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < h->lineCounter; i += 2) {
            lines.push_back(Line(h->lines[i], h->lines[i + 1]));
        }
    }
}

/**
 * Initializes handle object for given hough strategy
 *
 * @param handle Handle to be initialized
 * @param houghStrategy Strategy used to perform hough transform
 * @param nDevs Number of GPU devices
 */
void createHandle(HoughTransformHandle *&handle, int frameWidth, int frameHeight,
        HoughStrategy houghStrategy, SplitStrategy splitStrategy, int nDevs) {
    // additional param in order to crop roi, refering to regionOfInterest.
    int roiFrameWidth = frameWidth - (frameWidth / 9) - (frameWidth / 9); 
    int roiFrameHeight = frameHeight - (frameHeight / 2 + frameHeight / 10); 
    int roiStartX = frameWidth / 9;
    int roiStartY = frameHeight / 2 + frameHeight / 10;

    int nRows = (int) ceil(sqrt(roiFrameHeight * roiFrameHeight + roiFrameWidth * roiFrameWidth)) * 2 / RHO_STEP_SIZE;
    int nCols = (THETA_B -THETA_A + (2*THETA_VARIATION)) / THETA_STEP_SIZE;

    if (houghStrategy == HoughStrategy::kCuda) {
        CudaHandle *h = new CudaHandle();

        h->nDevs = nDevs;
        h->splitStrategy = splitStrategy;
        h->roiFrameSize = roiFrameWidth * roiFrameHeight * sizeof(unsigned char);
        // FIX: we assume device number divides global frame size
        if (splitStrategy != SplitStrategy::kNone) h->roiFrameSize /= nDevs;
        h->roiStart = roiStartY * frameWidth + roiStartX;  // offset in 1-dim

        // for copying roi frame
        h->frameOffset = new int[nDevs];
        h->frameOffset[0] = h->roiStart;

        switch (splitStrategy) {
        case SplitStrategy::kLeftRight:
            for (int dev = 0; dev < nDevs; dev++) {
                h->frameOffset[dev] = h->roiStart + dev * (roiFrameWidth / nDevs);
            }
            break;
        case SplitStrategy::kTopBottom:
            for (int dev = 0; dev < nDevs; dev++) {
                h->frameOffset[dev] = h->roiStart + dev * frameWidth * (roiFrameHeight / nDevs);
            }
            break;
        }

        // buffers
        cudaMallocHost(&(h->lines), 2 * MAX_NUM_LINES * sizeof(int));

        h->d_lines = new int *[nDevs];
        h->d_lineCounter = new int *[nDevs];
        h->d_frame = new unsigned char *[nDevs];
        h->d_accumulator = new int *[nDevs];

        for (int dev = 0; dev < nDevs; dev++) {
            cudaSetDevice(dev);
            cudaMalloc(h->d_lines + dev, 2 * MAX_NUM_LINES * sizeof(int));
            cudaMalloc(h->d_lineCounter + dev, sizeof(int));
            cudaMalloc(h->d_frame + dev, h->roiFrameSize);
            cudaMalloc(h->d_accumulator + dev, nRows * nCols * sizeof(int));
        }

        // nccl
        if (splitStrategy != SplitStrategy::kNone) {
            h->comms = new ncclComm_t[nDevs];
            h->devs = new int[nDevs];
            for (int i = 0; i < nDevs; i++) h->devs[i] = i;
            ncclCommInitAll(h->comms, nDevs, h->devs);
        }

        h->houghBlockDim = dim3(32, 5, 5);
        switch (splitStrategy) {
        case SplitStrategy::kLeftRight:
            h->houghGridDim = dim3(ceil(roiFrameHeight / 5), ceil(roiFrameWidth / nDevs / 5));
            break;
        case SplitStrategy::kTopBottom:
            h->houghGridDim = dim3(ceil(roiFrameHeight / nDevs / 5), ceil(roiFrameWidth / 5));
            break;
        default:
            h->houghGridDim = dim3(ceil(roiFrameHeight / 5), ceil(roiFrameWidth / 5));
        }

        h->findLinesBlockDim = dim3(32, 32);
        h->findLinesGridDim = dim3(ceil(nRows / 32), ceil(nCols / nDevs / 32));

        handle = (HoughTransformHandle *) h;
    } else if (houghStrategy == HoughStrategy::kSeq) {
        SeqHandle *h =  new SeqHandle();
        h->accumulator = new int[nRows * nCols];
        handle = (HoughTransformHandle *) h;
    }

    handle->roiFrameWidth = roiFrameWidth;
    handle->roiFrameHeight = roiFrameHeight;
    switch (splitStrategy) {
    case SplitStrategy::kLeftRight:
        handle->roiFrameWidth /= nDevs;
        break;
    case SplitStrategy::kTopBottom:
        handle->roiFrameHeight /= nDevs;
        break;
    }
    handle->roiStartX = roiStartX;
    handle->roiStartY = roiStartY;
    handle->nRows = nRows;
    handle->nCols = nCols;
}

/**
 * Frees memory on host and device that was allocated for the handle
 *
 * @param handle Handle to be destroyed
 * @param houghStrategy Hough strategy that was used to create the handle
 */
void destroyHandle(HoughTransformHandle *&handle, HoughStrategy houghStrategy) {
    if (houghStrategy == HoughStrategy::kCuda) {
        CudaHandle *h = (CudaHandle *) handle;

        // for copying roi frame
        delete[] h->frameOffset;

        // buffers
        for (int dev = 0; dev < h->nDevs; dev++) {
            cudaSetDevice(dev);
            cudaFree(h->d_lines[dev]);
            cudaFree(h->d_lineCounter[dev]);
            cudaFree(h->d_frame[dev]);
            cudaFree(h->d_accumulator[dev]);
        }

        delete[] h->d_lines;
        delete[] h->d_lineCounter;
        delete[] h->d_frame;
        delete[] h->d_accumulator;

        cudaFreeHost(h->lines);

        // nccl
        if (h->splitStrategy != SplitStrategy::kNone) {
            for (int i = 0; i < h->nDevs; i++) ncclCommDestroy(h->comms[i]);
            delete[] h->comms;
            delete[] h->devs;
        }
    } else if (houghStrategy == HoughStrategy::kSeq) {
        SeqHandle *h = (SeqHandle *) handle;
        delete[] h->accumulator;
    }

    delete handle;
    handle = NULL;
}
