/* ------------------------------------------------------ /
*Image Proccessing with Deep Learning
* OpenCV : Filter Demo
* Created : 2021 - Spring
------------------------------------------------------ */

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void main()
{
	cv::Mat src, dst;
	src = cv::imread("../../../Images//coin.jpg", 0);

	int i = 3;
	Size kernelSize = cv::Size(i, i);

	/* Blur */
	cv::blur(src, dst, cv::Size(i, i), cv::Point(-1, -1));
	namedWindow("Blur", WINDOW_NORMAL);
	imshow("Blur", dst);

	/* Gaussian Filter */
	GaussianBlur(src, dst, Size(i,i), 0, 0);
	namedWindow("Gaussian", WINDOW_NORMAL);
	imshow("Gaussian", dst);


	/* Median Filter */
	medianBlur(src, dst, i);
	namedWindow("Median", CV_WINDOW_AUTOSIZE);
	imshow("Median", dst);

	/* Bilateral Filter */
	bilateralFilter(src, dst, i, i * 2, i / 2);
	namedWindow("Bilateral", CV_WINDOW_AUTOSIZE);
	imshow("Bilateral", dst);

	/* Laplacian Filter */
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	cv::Laplacian(src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
	src.convertTo(src, CV_16S);
	cv::Mat result_laplacian = src - dst;
	result_laplacian.convertTo(result_laplacian, CV_8U);
	namedWindow("Laplacian", CV_WINDOW_AUTOSIZE);
	cv::imshow("Laplacian", result_laplacian);

	

	/* 2D Convolution of a filter kernel */
	/* Design a normalized box filter kernel 5 by 5 */
	src.convertTo(src, CV_8UC1);

	Mat kernel;
	delta = 0;
	ddepth = -1;
	kernel_size = 5;
	Point anchor = Point(-1, -1);

	kernel = Mat::ones(Size(kernel_size, kernel_size), CV_32F) / (float)pow(kernel_size, 2);
	filter2D(src, dst, ddepth, kernel, anchor, delta);

	namedWindow("Conv2D", CV_WINDOW_AUTOSIZE);
	cv::imshow("Conv2D", dst);

	


	cv::waitKey(0);
}