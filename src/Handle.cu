#include "Handle.hpp"

#include <cmath>

#include <nccl.h>

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

        // -- attributes --
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

        h->accCount = nRows * nCols;
        h->accSize = h->accCount * sizeof(int);
        h->linesSize = 2 * MAX_NUM_LINES * sizeof(int);

        // -- buffers --
        // host
        h->lines = new int *[nDevs];
        h->lineCounter = new int[nDevs];
        // device
        h->d_lines = new int *[nDevs];
        h->d_lineCounter = new int *[nDevs];
        h->d_frame = new unsigned char *[nDevs];
        h->d_accumulator = new int *[nDevs];

        for (int dev = 0; dev < nDevs; dev++) {
            cudaSetDevice(dev);
            // host
            cudaMallocHost(h->lines + dev, h->linesSize);
            // device
            cudaMalloc(h->d_lines + dev, h->linesSize);
            cudaMalloc(h->d_lineCounter + dev, sizeof(int));
            cudaMalloc(h->d_frame + dev, h->roiFrameSize);
            cudaMalloc(h->d_accumulator + dev, h->accSize);
        }

        // -- nccl --
        if (splitStrategy != SplitStrategy::kNone) {
            h->comms = new ncclComm_t[nDevs];
            h->devs = new int[nDevs];
            for (int i = 0; i < nDevs; i++) h->devs[i] = i;
            ncclCommInitAll(h->comms, nDevs, h->devs);
        }

        // -- kernel params --
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

        delete[] h->frameOffset;

        // -- buffers --
        for (int dev = 0; dev < h->nDevs; dev++) {
            cudaSetDevice(dev);
            // host
            cudaFreeHost(h->lines[dev]);
            // device
            cudaFree(h->d_lines[dev]);
            cudaFree(h->d_lineCounter[dev]);
            cudaFree(h->d_frame[dev]);
            cudaFree(h->d_accumulator[dev]);
        }

        // host
        delete[] h->lines;
        delete[] h->lineCounter;
        // device
        delete[] h->d_lines;
        delete[] h->d_lineCounter;
        delete[] h->d_frame;
        delete[] h->d_accumulator;

        // -- nccl --
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