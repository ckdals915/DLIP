/*
******************************************************************************
* @author	ChangMin An
* @Mod		2022-03-18
* @brief	Deep Learning Image Processing : LAB1 Grayscale Image Segmentation
*
******************************************************************************
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
using namespace cv;

// Define the variable
#define		ITERATION_SCALE		1

// Declare Function
void plotHist(Mat src, string plotname, int width, int height);
void Filter_Demo(int, void*);
void Threshold_Demo(int, void*);
void Morphology_Demo(int, void*);
void contour_Demo(const Mat& _src);
void print_Result(void);

int key = 0;
// Filter Variable
int i = 5;
bool contour_thred = false;

// Define Source Matrix 
Mat src, src_gray, src_filtered, src_equal, src_thred, src_morph, dst, hsv;
vector<Mat> hsv_split;
vector<Mat> rgb_split;

// Global variables for Threshold
int threshold_value = 0;
int threshold_type = 0;
int morphology_type = 0;
int filter_type = 0;
int kernel_size = 5;

const int max_value = 255;
const int max_type = 4;
const int max_thres_type = 16;
const int max_BINARY_value = 255;
const int max_kernel = 21;

// Morphology Variable
int erode_Iteration = 0;
int dilate_Iteration = 0;
int open_Iteration = 0;
int close_Iteration = 0;

// Laplacian value
int scale = 1;
int delta = 0;
int ddepth = CV_16S;

// Global variables for Morphology
int element_shape = MORPH_RECT;		// MORPH_RECT, MORPH_ELIPSE, MORPH_CROSS
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

// Trackbar strings
String window_name		= "Threshold Demo";
String trackbar_kernel	= "Kernel Value";
String trackbar_filter	= "Filter Type: \n 0: None \n 1: Blur \n 2: Gaussian \n 3: Median \n 4: Laplacian";
String trackbar_type	= "Thresh Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Invertd";
String trackbar_value	= "Thresh Value";
String trackbar_morph	= "Morph Type 0: None \n 1: erode \n 2: dilate \n 3: close \n 4: open";
String morphology_name	= "Mophology Demo";

// Counting Bolt and Nut
int Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;
int gray = 255;

int hmin = 0, hmax = 15, smin = 55, smax = 130, vmin = 145, vmax = 255;

// Main
void main(void)
{
	int hist_w = 512, hist_h = 400;
	// Load an image
	src = imread("..//..//..//Images//groupimage_crop.jpg", IMREAD_COLOR);

	// Convert the image to Gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/* Raw Data */
	namedWindow("Raw_Data", 1);
	imshow("Raw_Data", src_gray);

	/*============= Thresholding ==============*/
	// Create a window to display the results
	namedWindow(window_name, WINDOW_NORMAL);
	resizeWindow(window_name, Size(800, 800));

	// Create trackbar to choose type of threshold
	createTrackbar(trackbar_kernel, window_name, &kernel_size, max_kernel);
	createTrackbar(trackbar_filter, window_name, &filter_type, max_type, Filter_Demo);
	createTrackbar(trackbar_type, window_name, &threshold_type, max_thres_type, Threshold_Demo);
	createTrackbar(trackbar_value, window_name, &threshold_value, max_value, Threshold_Demo);
	
	// Call the function to initialize
	Filter_Demo(0, 0);
	Threshold_Demo(0, 0);
	Morphology_Demo(0, 0);
	//contour_Demo(dst);
	// Command for Thresholding, Morphology, and Contour
	
	while (true)
	{

		key = waitKey(20);

		if (key == 27)										break;
		else if (key == 'm' || key == 'M')					dst.copyTo(src_morph);
		else if (key == 'a' || key == 'A')					erode_Iteration += ITERATION_SCALE;
		else if (key == 's' || key == 'S')					dilate_Iteration += ITERATION_SCALE;
		else if (key == 'd' || key == 'D')					open_Iteration += ITERATION_SCALE;
		else if (key == 'f' || key == 'F')					close_Iteration += ITERATION_SCALE;
		else if (key == 'q' || key == 'Q')					erode_Iteration -= ITERATION_SCALE;
		else if (key == 'w' || key == 'W')					dilate_Iteration -= ITERATION_SCALE;
		else if (key == 'e' || key == 'E')					open_Iteration -= ITERATION_SCALE;
		else if (key == 'r' || key == 'R')					close_Iteration -= ITERATION_SCALE;
		else if (key == 'o' || key == 'O')					erode_Iteration = 0, dilate_Iteration = 0, open_Iteration = 0, close_Iteration = 0;
		else if (key == 'c' || key == 'C')					{ contour_thred = false; contour_Demo(src_morph); }
		else if (key == 't' || key == 'T')					{ contour_thred = true; contour_Demo(dst); }
		else if (key == 'p' || key == 'P')					print_Result();
		else if (key == 'h' || key == 'H')					{ plotHist(src_gray, "histplot_src", src_gray.cols, src_gray.rows); resizeWindow("histplot_src", Size(800, 800)); 
															  plotHist(src_filtered, "histplot_filtered", src_filtered.cols, src_filtered.rows); resizeWindow("histplot_filtered", Size(800, 800)); }
		else if (key == 'j' || key == 'J')					{ equalizeHist(src_gray, src_gray); }
		else												erode_Iteration		= erode_Iteration;

		erode(dst, src_morph, element, Point(-1, -1), erode_Iteration);
		dilate(src_morph, src_morph, element, Point(-1, -1), dilate_Iteration);
		morphologyEx(src_morph, src_morph, CV_MOP_OPEN, element, Point(-1, -1), open_Iteration);
		morphologyEx(src_morph, src_morph, CV_MOP_CLOSE, element, Point(-1, -1), close_Iteration);

		imshow(morphology_name, src_morph);

		
		
	}
	waitKey(0);
}

void Filter_Demo(int, void*)
{
	/*
	* 0: None
	* 1: Blur
	* 2: Gaussian
	* 3: Median
	* 4: Bilateral
	*/
	switch (filter_type) {
	case 0: src_gray.copyTo(src_filtered); break;
	case 1: blur(src_gray, src_filtered, Size(kernel_size, kernel_size), Point(-1, -1)); break;
	case 2: GaussianBlur(src_gray, src_filtered, Size(kernel_size, kernel_size), 0, 0); break;
	case 3: medianBlur(src_gray, src_filtered, kernel_size); break;
	case 4: bilateralFilter(src_gray, src_filtered, kernel_size, kernel_size * 2, kernel_size / 2);
	}
	imshow(window_name, src_filtered);
}

void Threshold_Demo(int, void*)	// default form of callback function for trackbar
{
	/*
	* 0: Binary
	* 1: Threshold Truncated
	* 2: Threshold to Zero
	* 3: Threshold to Zero Inverted
	* 7: Threshold Mask
	* 8: Threshold Otsu
	* 16: Threshold Triangle
	*/
	threshold(src_filtered, dst, threshold_value, max_BINARY_value, threshold_type);
	//adaptiveThreshold(src_filtered, dst, threshold_value, BORDER_REPLICATE, THRESH_BINARY, 3, 1);
	imshow(window_name, dst);
}

void contour_Demo(const Mat& _src)
{
	// Example code
	// src: binary image
	vector<vector<Point>> contours;

	// Contour 정보는 그 안에 어떤 일이 있다고 하더라도 상관없다. 최외곽 정보만 가져오는 것! : CV_RETR_EXTERNAL
	/// Find contours
	findContours(_src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw all contours excluding holes
	Mat drawing(_src.size(), CV_8U, Scalar(0));

	//cout << " Number of coins are =" << contours.size() << endl;
	Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;
	
	for (int i = 0; i < contours.size(); i++)
	{
		if (!contour_thred)
		{
			//printf(" * Contour[%d] -  Area OpenCV: %.2f - Length: %.2f \n", i, contourArea(contours[i]), arcLength(contours[i], true));
			if (contourArea(contours[i]) > 6000.0)
			{
				Bolt_M6++;
				gray = 255;
			}
			else if (contourArea(contours[i]) > 4000.0 && arcLength(contours[i], false) > 300.0)
			{
				Bolt_M5++;
				gray = 190;
			}
			else if (contourArea(contours[i]) > 4000.0 && arcLength(contours[i], false) <= 300.0)
			{
				Hexa_Nut_M6++;
				gray = 140;
			}
			else if (contourArea(contours[i]) > 2900.0)
			{
				Square_Nut_M5++;
				gray = 80;
			}
			else
			{
				Hexa_Nut_M5++;
				gray = 30;
			}
		}
		drawContours(drawing, contours, i, Scalar(gray), CV_FILLED, 8);
		
	}
	namedWindow("contour", 0);
	imshow("contour", drawing);
	drawing.copyTo(_src);
}

void Morphology_Demo(int, void*)  // default form of callback function for trackbar
{
	dst.copyTo(src_morph);
	namedWindow(morphology_name, WINDOW_NORMAL);
	resizeWindow(morphology_name, Size(800, 800));
	imshow(morphology_name, src_morph);
	//contour_Demo(src_morph);
}

void print_Result(void)
{
	cout << "# of \t\t Bolt \t\t M5 = " << Bolt_M5 << endl;
	cout << "# of \t\t Bolt \t\t M6 = " << Bolt_M6 << endl;
	cout << "# of \t\t Square Nut \t M5 = " << Square_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M5 = " << Hexa_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M6 = " << Hexa_Nut_M6 << endl;
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
	Mat hist_normed = hist * height / max_val;
	float bin_w = (float)width / histSize;
	Mat histImage(height, width, CV_8UC1, Scalar(0));
	for (int i = 0; i < histSize - 1; i++) {
		line(histImage,
			Point((int)(bin_w * i), height - cvRound(hist_normed.at<float>(i, 0))),
			Point((int)(bin_w * (i + 1)), height - cvRound(hist_normed.at<float>(i + 1, 0))),
			Scalar(255), 2, 8, 0);
	}

	imshow(plotname, histImage);
}

