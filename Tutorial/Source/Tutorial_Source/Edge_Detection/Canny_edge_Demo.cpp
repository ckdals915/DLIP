#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
const char* window_name = "Edge Map";

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	// Threshold1: 대체로 2보다 작다. 먼저 곁가지를 뽑아낸다. 높아질수록 곁가지가 없어진다.(겉가지)
	// Threshold2: 강한 edge만을 출력하는 것!! 곁가지가 없어진다.(줄기)
	// 즉 이 두개를 잘 조절해야 한다. 이어지는 정도가 얼마나 큰가?
	// Edge를 검출하는 기법 (코너 판단)
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}


/** @function main */
int main(int argc, char** argv)
{
	/// Load an image
	//const char* filename = "../images/pillsetc.png";
	const char* filename = "..//..//..//images/coins.png";
	//const char* filename = "../images/TrafficSign1.png";

	/// Read the image
	src = imread(filename, 1);

	if (!src.data)
	{
		return -1;
	}

	/// Create a matrix of the same type and size as src (for dst)
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

	/// Show the image
	CannyThreshold(0, 0);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}