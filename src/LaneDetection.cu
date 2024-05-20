#include <ctime>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "HoughTransform.hpp"
#include "Line.hpp"
#include "Preprocessing.hpp"
#define MAX_NUM_LINES 10

extern void detectLanes(cv::VideoCapture inputVideo, cv::VideoWriter outputVideo, int houghStrategy);
extern void drawLines(cv::Mat &frame, std::vector<Line> lines);
extern cv::Mat plotAccumulator(int nRows, int nCols, int *accumulator);

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "usage LaneDetection inputVideo outputVideo [--cuda|--seq]" << std::endl << std::endl;
        std::cout << "Positional Arguments:" << std::endl;
        std::cout << " inputVideo    Input video for which lanes are detected" << std::endl;
        std::cout << " outputVideo   Name of resulting output video" << std::endl << std::endl;
        std::cout << "Optional Arguments:" << std::endl;
        std::cout << " --cuda        Perform hough transform using CUDA (default)" << std::endl;
        std::cout << " --seq         Perform hough transform sequentially on the CPU" << std::endl;
        return 1;
    }

    // Read input video
    cv::VideoCapture capture(argv[1]);
    // Check which strategy to use for hough transform (CUDA or sequential)
    int houghStrategy = argc > 3 && !strcmp(argv[3], "--seq") ? SEQUENTIAL : CUDA;
    int frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (!capture.isOpened()) {
        std::cout << "Unable to open video" << std::endl;
        return -1;
    }

    cv::VideoWriter video(argv[2], cv::VideoWriter::fourcc('M','J','P','G'), 30,
        cv::Size(frameWidth, frameHeight), true);

    detectLanes(capture, video, houghStrategy);

    return 0;
}

/**
 * Coordinates the lane detection using the specified hough strategy for the 
 * given input video and writes resulting video to output video
 * 
 * @param inputVideo Video for which lanes are detected
 * @param outputVideo Video where results are written to
 * @param houghStrategy Strategy which should be used to parform hough transform
 */
void detectLanes(cv::VideoCapture inputVideo, cv::VideoWriter outputVideo, int houghStrategy) {
    cv::Mat frames[2], preProcFrames[2];
    std::vector<Line> lines[2];

    clock_t readTime = 0;
	clock_t prepTime = 0;
	clock_t houghTime = 0;
    clock_t redundantTime = 0;
	clock_t writeTime = 0;
    clock_t totalTime = 0;

    std::cout << "Processing video " << (houghStrategy == CUDA ? "using CUDA" : "Sequentially") << std::endl;
    totalTime -= clock();

    int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

    HoughTransformHandle *handle;
    createHandle(handle, houghStrategy, frameWidth, frameHeight);
    int frameIndex = 0;

	for( ; ; ) {
        
        int gpuIndex = frameIndex % 2;
        
        // Read next frame
        readTime -= clock();
        inputVideo >> frames[gpuIndex];
        readTime += clock();
        if (frames[gpuIndex].empty())
            break;

        // Apply pre-processing steps
        prepTime -= clock();
        preProcFrames[gpuIndex] = filterLanes(frames[gpuIndex]);
        preProcFrames[gpuIndex] = applyGaussianBlur(preProcFrames[gpuIndex]);
        preProcFrames[gpuIndex] = applyCannyEdgeDetection(preProcFrames[gpuIndex]);
        preProcFrames[gpuIndex] = regionOfInterest(preProcFrames[gpuIndex]);
        prepTime += clock();

        // Perform hough transform
        houghTime -= clock();
        if(frameIndex > 1)
            redundantTime += clock();
        if (houghStrategy == CUDA)
            houghTransformCuda(handle, preProcFrames[gpuIndex], gpuIndex);
        else if (houghStrategy == SEQUENTIAL)
            houghTransformSeq(handle, preProcFrames[gpuIndex], lines[gpuIndex]);
     
        if (frameIndex > 0) {
            int prevGpuIndex = gpuIndex ^ 1;
            cudaSetDevice(prevGpuIndex);
            cudaDeviceSynchronize();
            
            lines[prevGpuIndex].clear();
            for (int i = 0; i < handle->lineCounter[prevGpuIndex] - 1; i += 2) {
                lines[prevGpuIndex].push_back(Line(handle->lines[prevGpuIndex][i], handle->lines[prevGpuIndex][i + 1]));
            }
            houghTime += clock();
            houghTime -= redundantTime;
            redundantTime = 0;
            redundantTime -= clock();
            
            // video
            writeTime -= clock();
            drawLines(frames[prevGpuIndex], lines[prevGpuIndex]);
            outputVideo << frames[prevGpuIndex];
            writeTime += clock();
        }

        frameIndex++;
    }
    
    int gpuIndex = frameIndex % 2;
    int prevGpuIndex = gpuIndex ^ 1;
    cudaSetDevice(prevGpuIndex);
    cudaDeviceSynchronize();
    
    lines[prevGpuIndex].clear();
    for (int i = 0; i < handle->lineCounter[prevGpuIndex] - 1; i += 2) {
        lines[prevGpuIndex].push_back(Line(handle->lines[prevGpuIndex][i], handle->lines[prevGpuIndex][i + 1]));
    }
    redundantTime += clock();
    houghTime += clock();
    houghTime -= redundantTime;
    // video
    writeTime -= clock();
    drawLines(frames[prevGpuIndex], lines[prevGpuIndex]);
    outputVideo << frames[prevGpuIndex];
    writeTime += clock();
    
    
    totalTime += clock();
    std::cout << "Read\tPrep\tHough\tWrite\tTotal" << std::endl;
    std::cout << std::setprecision(4) << (((float) readTime) / CLOCKS_PER_SEC) << "\t"
         << (((float) prepTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) houghTime) / CLOCKS_PER_SEC) << "\t"
		 << (((float) writeTime) / CLOCKS_PER_SEC) << "\t"
    	 << (((float) totalTime) / CLOCKS_PER_SEC) << std::endl;
}

/** Draws given lines onto frame */
void drawLines(cv::Mat &frame, std::vector<Line> lines) {
    for (size_t i = 0; i < lines.size(); i++) {
        int y1 = frame.rows;
        int y2 = (frame.rows / 2) + (frame.rows / 10);
        int x1 = (int) lines[i].getX(y1);
        int x2 = (int) lines[i].getX(y2);

        cv::line(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 5, 8, 0);
    }
}

/**
 * Helper function which plots the accumulator and returns result image (only 
 * for debugging purposes)
 */
cv::Mat plotAccumulator(int nRows, int nCols, int *accumulator) {
    cv::Mat plotImg(nRows, nCols, CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < nRows; i++) {
  		for (int j = 0; j < nCols; j++) {
			plotImg.at<uchar>(i, j) = std::min(accumulator[(i * nCols) + j] * 4, 255);
  		}
  	}

    return plotImg;
}
