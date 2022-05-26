
/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Threshold using Trackbar Demo
* Created: 2021-Spring
------------------------------------------------------*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Global variables for Threshold
int threshold_value = 0;
int threshold_type = 0;
int morphology_type = 0;

int const max_value = 255;
int const max_type = THRESH_TRIANGLE;
int const max_BINARY_value = 255;

// Global variables for Morphology
int element_shape = MORPH_RECT;		// MORPH_RECT, MORPH_ELIPSE, MORPH_CROSS
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

Mat src, src_gray, dst, dst_morph;


// Trackbar strings
String window_name = "Threshold & Morphology Demo";
String trackbar_type = "Thresh Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Invertd";
String trackbar_value = "Thresh Value";
String trackbar_morph = "Morph Type 0: None \n 1: erode \n 2: dilate \n 3: close \n 4: open";

// Function headers
void Threshold_Demo(int, void*);
void Morphology_Demo(int, void*);

int main()
{
	// Load an image
	src = imread("..//..//..//Images//170408580.jpg", IMREAD_COLOR);

	// Convert the image to Gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	// Create a window to display the results
	namedWindow(window_name, WINDOW_NORMAL);

	// Create trackbar to choose type of threshold
	createTrackbar(trackbar_type, window_name, &threshold_type, max_type, Threshold_Demo);
	createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
	createTrackbar(trackbar_morph, window_name, &morphology_type, max_type, Morphology_Demo);

	// Call the function to initialize
	Threshold_Demo(0, 0);
	Morphology_Demo(0, 0);

	// Wait until user finishes program
	while (true) {
		int c = waitKey(20);
		if (c == 27)
			break;
	}
}

void Threshold_Demo(int, void*)	// default form of callback function for trackbar
{
	/*
	* 0: Binary
	* 1: Threshold Truncated
	* 2: Threshold to Zero
	* 3: Threshold to Zero Inverted
	*/

	threshold(src_gray, dst, threshold_value, max_BINARY_value, threshold_type);
	imshow(window_name, dst);
}

void Morphology_Demo(int, void*)  // default form of callback function for trackbar
{
	/*
	* 0: None
	* 1: Erode
	* 2: Dilate
	* 3: Close
	* 4: Open
	*/
	switch (morphology_type) {
	case 0: dst.copyTo(dst_morph);	break;
	case 1: erode(dst, dst_morph, element); break;
	case 2: dilate(dst, dst_morph, element); break;
	case 3: morphologyEx(dst, dst_morph, CV_MOP_OPEN, element); break;
	case 4: morphologyEx(dst, dst_morph, CV_MOP_CLOSE, element); break;
	}
	imshow(window_name, dst_morph);
}


void plotHist(Mat src, string plotname, int width, int height) {

	/// Compute the histograms 
	Mat hist;
	/// Establish the number of bins (for uchar Mat type)
	int histSize = 256;
	/// Set the ranges (for uchar Mat type)
	float range[] = { 0, 256 };

	const float* histRange = { range };
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	double min_val, max_val;
	cv::minMaxLoc(hist, &min_val, &max_val);
	Mat hist_normed = hist * height / max_val; // max_val를 plot 이미지의 최고 높이에 맞춰지도록 함
	float bin_w = (float)width / histSize;	// plot이미지 가로크기에 맞추어 가로방향 축 간격값 지정
	Mat histImage(height, width, CV_8UC1, Scalar(0));	// plot용도로 사용될 width, height 크기의 8비트 흰색 이미지 생성
	for (int i = 0; i < histSize - 1; i++) {	// 그래프는 두 점을 잇는 형식으로 진행되므로 256 - 1번만큼 진행
		line(histImage,	// histImage에 라인을 그리는 함수
			Point((int)(bin_w * i), height - cvRound(hist_normed.at<float>(i, 0))),			// 시작점 : bin_w * k , normalized_val * n[k]
			Point((int)(bin_w * (i + 1)), height - cvRound(hist_normed.at<float>(i + 1, 0))),	// 종료점 : bin_w * (k+1) , normalized_val * n[k+1]
			Scalar(255), 2, 8, 0);													// 선의 색깔은 검정(0), 선두께는 linewidth로 지정
	}

	imshow(plotname, histImage);
}