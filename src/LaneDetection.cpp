#include <getopt.h>
#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "Handle.hpp"
#include "HoughTransform.hpp"
#include "Line.hpp"
#include "Preprocessing.hpp"

void showUsage(const char *arg0);
void detectLanes(cv::VideoCapture inputVideo, cv::VideoWriter outputVideo,
    HoughStrategy houghStrategy, SplitStrategy splitStrategy, int nDevs);
void drawLines(cv::Mat &frame, std::vector<Line> lines);
cv::Mat plotAccumulator(int nRows, int nCols, int *accumulator);

extern char *optarg;
extern int optind;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        showUsage(argv[0]);
        return 1;
    }

    HoughStrategy houghStrategy = kCuda;
    SplitStrategy splitStrategy = kNone;
    int nDevs = 1;

    struct option opts[] = {
        { "seq", no_argument, (int *) &houghStrategy, HoughStrategy::kSeq },
        { "ss", optional_argument, 0, 0 },
        { "nd", optional_argument, 0, 0 },
        { nullptr, 0, nullptr, 0 },
    };
    int optIdx;
    int val;

    while ((val = getopt_long_only(argc, argv, "", opts, &optIdx)) != -1) {
        if (val == 0) {
            switch (optIdx) {
            case 1:
                splitStrategy = (SplitStrategy) strtol(optarg, nullptr, 10);
                break;
            case 2:
                nDevs = strtol(optarg, nullptr, 10);
                break;
            }
        } else if (val == '?') {
            showUsage(argv[0]);
            return 1;
        }
    }

    const char* videoInput = argv[optind];
    const char* videoOutput = argv[optind + 1];

    // Read input video
    cv::VideoCapture capture(videoInput);
    int frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    if (!capture.isOpened()) {
        std::cerr << "Unable to open video" << std::endl;
        return -1;
    }

    cv::VideoWriter video(videoOutput, cv::VideoWriter::fourcc('M','J','P','G'), 30,
        cv::Size(frameWidth, frameHeight), true);

    detectLanes(capture, video, houghStrategy, splitStrategy, nDevs);

    return 0;
}

void showUsage(const char *arg0) {
    std::cout << "Usage: " << arg0 << " inputVideo outputVideo [options]" << std::endl << std::endl;
    std::cout << " inputVideo    Input video for which lanes are detected" << std::endl;
    std::cout << " outputVideo   Name of resulting output video" << std::endl << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << " --seq         Perform hough transform sequentially on the CPU (if omitted, CUDA is used)" << std::endl;
    std::cout << " --ss <num>    How to split the frame (default: 0, should not be used when --nd=1)" << std::endl;
    std::cout << "   0           no split" << std::endl;
    std::cout << "   1           left half & right half" << std::endl;
    std::cout << "   2           top half & bottom half" << std::endl;
    std::cout << "   3           cyclic split from left to right" << std::endl;
    std::cout << "   4           cyclic split from top to bottom" << std::endl;
    std::cout << " --nd <num>    Number of GPU devices (default: 1)" << std::endl;
}

/**
 * Coordinates the lane detection using the specified hough strategy for the 
 * given input video and writes resulting video to output video
 * 
 * @param inputVideo Video for which lanes are detected
 * @param outputVideo Video where results are written to
 * @param houghStrategy Strategy which should be used to parform hough transform
 * @param nDevs Number of GPU devices
 */
void detectLanes(cv::VideoCapture inputVideo, cv::VideoWriter outputVideo,
        HoughStrategy houghStrategy, SplitStrategy splitStrategy, int nDevs) {
    cv::Mat frame, preProcFrame;
    std::vector<Line> lines;

    clock_t readTime  = 0;
	clock_t prepTime  = 0;
	clock_t houghTime = 0;
	clock_t writeTime = 0;
    clock_t totalTime = 0;

    std::cout << "Processing video " << (houghStrategy == HoughStrategy::kCuda ? "using CUDA" : "Sequentially") << std::endl;
    totalTime -= clock();

    int frameWidth  = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);

    HoughTransformHandle *handle;
    createHandle(handle, frameWidth, frameHeight, houghStrategy, splitStrategy, nDevs);

	for (;;) {
        // Read next frame
        readTime -= clock();
		inputVideo >> frame;
        readTime += clock();
		if (frame.empty())
			break;

        // Apply pre-processing steps
        prepTime -= clock();
        preProcFrame = filterLanes(frame);
        preProcFrame = applyGaussianBlur(preProcFrame);
        preProcFrame = applyCannyEdgeDetection(preProcFrame);
        preProcFrame = regionOfInterest(preProcFrame);
        prepTime += clock();

        // Perform hough transform
        houghTime -= clock();
        lines.clear();
        if (houghStrategy == HoughStrategy::kCuda)
            houghTransformCuda(handle, preProcFrame, lines);
        else if (houghStrategy == HoughStrategy::kSeq)
            houghTransformSeq(handle, preProcFrame, lines);
        houghTime += clock();

        // Draw lines to frame and write to output video
        writeTime -= clock();
        drawLines(frame, lines);
        outputVideo << frame;
        writeTime += clock();
    }

    destroyHandle(handle, houghStrategy);

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
			plotImg.at<unsigned char>(i, j) = std::min(accumulator[(i * nCols) + j] * 4, 255);
  		}
  	}

    return plotImg;
}
