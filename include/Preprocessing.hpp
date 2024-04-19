#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/core.hpp>

/** Filters white and yellow lane markers from the image */
cv::Mat filterLanes(cv::Mat img);

/** Applys gaussian blur with kernel size 5 to image */
cv::Mat applyGaussianBlur(cv::Mat img);

/** Applyes canny edge detection no given image */
cv::Mat applyCannyEdgeDetection(cv::Mat img);

/** 
 * Crops out region of interest from image. The region of interest is the 
 * region which would usually contain the lane markers.
 */
cv::Mat regionOfInterest(cv::Mat img);

#endif  // PREPROCESSING_HPP