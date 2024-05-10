#ifndef HOUGH_TRANSFORM_H
#define HOUGH_TRANSFORM_H

#include <vector>

#include <opencv2/core.hpp>

#include "Handle.hpp"
#include "Line.hpp"

/**
 * Performs hough transform for given frame sequentially and adds found lines
 * in 'lines' vector
 * 
 * @param handle Handle tracking relevant info accross executions
 * @param frame Video frame on which hough transform is applied
 * @param lines Vector to which found lines are added to 
 */
void houghTransformSeq(HoughTransformHandle *handle, cv::Mat frame, std::vector<Line> &lines);

/**
 * Performs hough transform for given frame using CUDA and adds found lines
 * in 'lines' vector
 * 
 * @param handle Handle tracking relevant info accross executions
 * @param lines Vector to which found lines are added to 
 */
void houghTransformCuda(HoughTransformHandle *handle, std::vector<Line> &lines);

// for generating the video of accumulator
void copyAccumulator(HoughTransformHandle *handle, int *accumulator);

#endif  // HOUGH_TRANSFORM_H
