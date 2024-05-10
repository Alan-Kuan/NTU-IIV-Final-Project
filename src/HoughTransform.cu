#include "HoughTransform.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <nccl.h>

#include "Handle.hpp"
#include "Line.hpp"

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double theta;
    int rho;
    bool splitFlag = true;
    if (splitStrategy == SplitStrategy::kLeftRightCyclic) {
        splitFlag = blockIdx.y % 2 == dev;
    }else if (splitStrategy == SplitStrategy::kTopBottomCyclic) {
        splitFlag = blockIdx.x % 2 == dev;
    }

    if (i < roiFrameHeight && j < roiFrameWidth && ((int) roiFrame[(i * roiFrameWidth) + j]) != 0 && splitFlag) {
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
        for (int k = threadIdx.z * (1 / THETA_STEP_SIZE); k < (threadIdx.z + 1) * (1 / THETA_STEP_SIZE); k++) {
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
    int j = 2 * (blockIdx.y * blockDim.y + threadIdx.y) + dev;

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
 * @param lines Vector to which found lines are added to
 */
void houghTransformCuda(HoughTransformHandle *handle, std::vector<Line> &lines) {
    CudaHandle *h = (CudaHandle *) handle;

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemcpy2DAsync(h->d_frame[dev], h->roiFrameWidth, h->p_frame + h->frameOffset[dev], h->frameWidth,
            h->roiFrameWidth, h->roiFrameHeight, cudaMemcpyHostToDevice);
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemsetAsync(h->d_accumulator[dev], 0, h->accSize);
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        houghKernel<<<h->houghGridDim, h->houghBlockDim>>>(h->roiFrameWidth, h->roiFrameHeight, h->d_frame[dev],
            h->roiStartX, h->roiStartY, h->nRows, h->nCols, h->d_accumulator[dev], dev, h->splitStrategy);
    }

    if (h->splitStrategy != SplitStrategy::kNone) {
        ncclGroupStart();
        for (int dev = 0; dev < h->nDevs; dev++) {
            ncclAllReduce(h->d_accumulator[dev], h->d_accumulator[dev], h->accCount,
                ncclInt, ncclSum, h->comms[dev], cudaStreamDefault);
        }
        ncclGroupEnd();
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemsetAsync(h->d_lineCounter[dev], 0, sizeof(int));
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        findLinesKernel<<<h->findLinesGridDim, h->findLinesBlockDim>>>(h->nRows, h->nCols,
            h->d_accumulator[dev], h->d_lines[dev], h->d_lineCounter[dev], dev);
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemcpyAsync(h->lines[dev], h->d_lines[dev], h->linesSize, cudaMemcpyDeviceToHost);
    }
    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaMemcpyAsync(h->lineCounter + dev, h->d_lineCounter[dev], sizeof(int), cudaMemcpyDeviceToHost);
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }

    for (int dev = 0; dev < h->nDevs; dev++) {
        for (size_t i = 0; i < h->lineCounter[dev]; i += 2) {
            lines.push_back(Line(h->lines[dev][i], h->lines[dev][i + 1]));
        }
    }
}

// for generating the video of accumulator
void copyAccumulator(HoughTransformHandle *handle, int *accumulator) {
    CudaHandle *h = (CudaHandle *) handle;
    cudaMemcpy(accumulator, h->d_accumulator[0], h->accSize, cudaMemcpyDeviceToHost);
}
