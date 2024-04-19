#include "Preprocessing.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define KERNEL_SIZE 5

/** Filters white and yellow lane markers from the image */
cv::Mat filterLanes(cv::Mat img) {
    cv::Mat hsvImg;
	cv::Mat grayImg;
    cv::cvtColor(img, hsvImg, cv::COLOR_BGR2HSV);
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);

    cv::Mat yellowHueRange;
	cv::Mat whiteHueRange;
	cv::Mat mask;
	cv::inRange(hsvImg, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), yellowHueRange);
	cv::inRange(img, cv::Scalar(120, 120, 120), cv::Scalar(255, 255, 255), whiteHueRange);
	cv::bitwise_or(yellowHueRange, whiteHueRange, mask);
	cv::bitwise_and(grayImg, mask, grayImg);

	return grayImg;
}

/** Applys gaussian blur with kernel size 5 to image */
cv::Mat applyGaussianBlur(cv::Mat img) {
    cv::GaussianBlur(img, img, cv::Size(KERNEL_SIZE, KERNEL_SIZE), 0);
	return img;
}

/** Applyes canny edge detection no given image */
cv::Mat applyCannyEdgeDetection(cv::Mat img) {
    cv::Canny(img, img, 50, 150);
	return img;
}

/** 
 * Crops out region of interest from image. The region of interest is the 
 * region which would usually contain the lane markers.
 */
cv::Mat regionOfInterest(cv::Mat img) {
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

    std::vector<cv::Point> vertices;
	vertices.push_back(cv::Point(img.cols / 9, img.rows));
	vertices.push_back(cv::Point(img.cols - (img.cols / 9), img.rows));
	vertices.push_back(cv::Point((img.cols / 2) + (img.cols / 8), (img.rows / 2) + (img.rows / 10)));
	vertices.push_back(cv::Point((img.cols / 2) - (img.cols / 8), (img.rows / 2) + (img.rows / 10)));

	// Create Polygon from vertices
    std::vector<cv::Point> ROI_Poly;
    cv::approxPolyDP(vertices, ROI_Poly, 1.0, true);

	// Fill polygon white
    cv::fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
	cv::bitwise_and(img, mask, img);

	return img;
}
