#include "HoughTransform.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nccl.h"

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

    for(int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            if ((int) frame.at<uchar>(i, j) == 0)
                continue;

            // thetas of interest will be close to 45 and close to 135 (vertical lines)
            // we are doing 2 thetas at a time, 1 for each theta of Interest
            // we use thetas varying 15 degrees more and less
            for(int k = 0; k < 2 * THETA_VARIATION * (1 / THETA_STEP_SIZE); k++) {
                theta = THETA_A - THETA_VARIATION + ((double) k * THETA_STEP_SIZE);
                rho = calcRho(j, i, theta);
                h->accumulator[index(h->nRows, h->nCols, rho, theta)] += 1;

                theta = THETA_B-THETA_VARIATION + ((double) k * THETA_STEP_SIZE);
                rho = calcRho(j, i, theta);
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
__global__ void houghKernel(int frameWidth, int frameHeight, unsigned char* frame, int nRows, int nCols, int *accumulator, const int devIdx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double theta;
    int rho;

    if(i < frameHeight && j < frameWidth && ((int) frame[(i * frameWidth) + j]) != 0 && blockIdx.y % 2 == devIdx) {

        // thetas of interest will be close to 45 and close to 135 (vertical lines)
        // we are doing 2 thetas at a time, 1 for each theta of Interest
        // we use thetas varying 15 degrees more and less
        for(int k = threadIdx.z * (1 / THETA_STEP_SIZE); k < (threadIdx.z + 1) * (1 / THETA_STEP_SIZE); k++) {
            theta = THETA_A-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(j, i, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);

            theta = THETA_B-THETA_VARIATION + ((double)k*THETA_STEP_SIZE);
            rho = calcRho(j, i, theta);
            atomicAdd(&accumulator[index(nRows, nCols, rho, theta)], 1);
        }
    }
}

/**
 * CUDA kernel responsible for finding lines based on the number of votes for
 * every rho/theta combination
 */
__global__ void findLinesKernel(int nRows, int nCols, int *accumulator, int *lines, int *lineCounter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

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

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemcpy(h->d_frame[dev], frame.ptr(), h->frameSize, cudaMemcpyHostToDevice);
        cudaMemset(h->d_accumulator[dev], 0, h->nRows * h->nCols * sizeof(int));
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        houghKernel<<<h->houghGridDim, h->houghBlockDim>>>(frame.cols, frame.rows, h->d_frame[dev], h->nRows, h->nCols, h->d_accumulator[dev], dev);    
        cudaDeviceSynchronize();
    }
    
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;

    cudaMemset(h->d_lineCounter[0], 0, sizeof(int));
    findLinesKernel<<<h->findLinesGridDim, h->findLinesBlockDim>>>(h->nRows, h->nCols,
        h->d_accumulator[0], h->d_lines[0], h->d_lineCounter[0]);
    cudaDeviceSynchronize();

    cudaMemcpy(&h->lineCounter, h->d_lineCounter[0], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h->lines, h->d_lines[0], 2 * MAX_NUM_LINES * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h->lineCounter - 1; i += 2) {
        lines.push_back(Line(h->lines[i], h->lines[i + 1]));
    }
}

/**
 * Initializes handle object for given hough strategy
 *
 * @param handle Handle to be initialized
 * @param houghStrategy Strategy used to perform hough transform
 * @param nDevs Number of GPU devices
 */
void createHandle(HoughTransformHandle *&handle, int houghStrategy, int frameWidth, int frameHeight, int nDevs) {
    int nRows = (int) ceil(sqrt(frameHeight * frameHeight + frameWidth * frameWidth)) * 2 / RHO_STEP_SIZE;
    int nCols = (THETA_B -THETA_A + (2*THETA_VARIATION)) / THETA_STEP_SIZE;

    if (houghStrategy == CUDA) {
        CudaHandle *h = new CudaHandle();
        h->nDevs = nDevs;
        // FIX: we assume device number divides global frame size
        h->frameSize = frameWidth * frameHeight * sizeof(uchar) / nDevs;
        cudaMallocHost(&(h->lines), 2 * MAX_NUM_LINES * sizeof(int));
        h->lineCounter = 0;

        h->d_lines = new int *[nDevs];
        h->d_lineCounter = new int *[nDevs];
        h->d_frame = new uchar *[nDevs];
        h->d_accumulator = new int *[nDevs];

        for (int dev = 0; dev < nDevs; dev++) {
            cudaSetDevice(dev);
            cudaMalloc(h->d_lines + dev, 2 * MAX_NUM_LINES * sizeof(int));
            cudaMalloc(h->d_lineCounter + dev, sizeof(int));
            cudaMalloc(h->d_frame + dev, h->frameSize);
            cudaMalloc(h->d_accumulator + dev, nRows * nCols * sizeof(int));
        }

        h->houghBlockDim = dim3(32, 5, 5);
        h->houghGridDim = dim3(ceil(frameHeight / 5), ceil(frameWidth / 5));
        h->findLinesBlockDim = dim3(32, 32);
        h->findLinesGridDim = dim3(ceil(nRows / 32), ceil(nCols / 32));

        handle = (HoughTransformHandle *) h;
    } else if (houghStrategy == SEQUENTIAL) {
        SeqHandle *h =  new SeqHandle();
        h->accumulator = new int[nRows * nCols];
        handle = (HoughTransformHandle *) h;
    }

    handle->nRows = nRows;
    handle->nCols = nCols;
}

/**
 * Frees memory on host and device that was allocated for the handle
 *
 * @param handle Handle to be destroyed
 * @param houghStrategy Hough strategy that was used to create the handle
 */
void destroyHandle(HoughTransformHandle *&handle, int houghStrategy) {
    if (houghStrategy == CUDA) {
        CudaHandle *h = (CudaHandle *) handle;

        for (int dev = 0; dev < h->nDevs; dev++) {
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
    } else if (houghStrategy == SEQUENTIAL) {
        SeqHandle *h = (SeqHandle *) handle;
        delete[] h->accumulator;
    }

    delete handle;
    handle = NULL;
}