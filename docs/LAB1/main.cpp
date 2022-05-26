/*
******************************************************************************
* @author	ChangMin An
* @Mod		2022-03-28
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

// Define the Value
#define KERNEL_SIZE 5
#define THRESHOLD_VALUE 116

enum Filter_Mode {
	BLUR = 1, GAUSSIAN, MEDIAN, LAPLACIAN, DEFAULT
};

// Declare Global Variable
Mat src, src_gray, src_filtered, dst, dst_morph;
int gray = 255;
int Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;

// Declare window-print name
String raw_data = "Raw_Data";
String result = "Result";

// Define Function
void Image_Processing(void);
void Convert_Gray(Mat& _src, Mat& _dst);
void Print_Window(String& _message, Mat& _src);
void Filter_Process(Mat& _src, Mat& _dst, int _filter, int _kernel);
void Threshold_Process(Mat& _src, Mat& _dst, int _thresValue);
void Contour_Process(Mat& _src, bool _flag);
void Morphology_Process(Mat& _src, Mat& _dst);
void Print_Result(void);

/*================= Main Loop ==================*/
void main(void)
{
	// Load an image
	src = imread("Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);

	// Process the image
	Image_Processing();

	// Print the result about Bolt & Nut
	Print_Result();

	// Print the Window about sorting result (break for ESC)
	while (true)
	{
		Print_Window(result, dst_morph);
		
		int key = waitKey(20);
		if (key == 27)	break;
	}
}

/*================= Function ==================*/

// Image Process about Bolt & Nut
void Image_Processing(void)
{

	Convert_Gray(src, src_gray);									// Convert the image to Gray
	Print_Window(raw_data, src_gray);								// Print the gray scale image
	Filter_Process(src_gray, src_filtered, MEDIAN, KERNEL_SIZE);	// Use Median-Filter & kernel_size = 5
	Threshold_Process(src_filtered, dst, THRESHOLD_VALUE);			// Threshold process in threshold-value = 116 					
	Contour_Process(dst, true);										// Fill in the empty space of an object
	Morphology_Process(dst, dst_morph);								// Morphology process (Erode: 5-times, Dirate: 2-times)
	Contour_Process(dst_morph, false);								// Counting each object and Distinguishing object to use brightness (Area, arcLength)
}

/* Convert the image to Gray
* _src: Input Matrix
* _dst: Output Matrix
*/
void Convert_Gray(Mat& _src, Mat& _dst)
{
	cvtColor(_src, _dst, CV_BGR2GRAY);
}

/* Print the gray scale image
* _message: Title
* _src: wanted to print matrix
*/
void Print_Window(String& _message, Mat& _src)
{
	namedWindow(_message, 0);
	resizeWindow(_message, Size(800, 800));
	imshow(_message, _src);
}

/* Filter Processing
* _src:		Input Matrix
* _dst:		Output Matrix
* _filter:	Type of Filter
* _kernel:	Kernel Size
*/
void Filter_Process(Mat& _src, Mat& _dst, int _filter, int _kernel)
{
	if		(_filter == BLUR)			blur(_src, _dst, cv::Size(_kernel, _kernel), cv::Point(-1, -1));
	else if (_filter == GAUSSIAN)		GaussianBlur(_src, _dst, cv::Size(_kernel, _kernel), 0, 0);
	else if (_filter == MEDIAN)			medianBlur(_src, _dst, _kernel);
}

/* Threshold Processing
* _src:			Input Matrix
* _dst:			Output Matrix
* _thresValue:	Threshold Value
*/
void Threshold_Process(Mat& _src, Mat& _dst, int _thresValue)
{
	threshold(_src, _dst, _thresValue, 255, 0);
}

/* Contour Processing
* _src:		Input Matrix
* _flag:	true(contour)/false(counting and color)
*/
void Contour_Process(Mat& _src, bool _flag)
{
	vector<vector<Point>> contours;
	int gray = 255;

	/// Find contours
	findContours(_src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw all contours excluding holes
	Mat drawing(_src.size(), CV_8U, Scalar(0));

	Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (!_flag)
		{
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
	drawing.copyTo(_src);
}

/* Morphology processing
* _src:	Input Matrix
* _dst:	Output Matrix
*/
void Morphology_Process(Mat& _src, Mat& _dst)
{
	int element_shape = MORPH_RECT;
	int n = 3;
	Mat element = getStructuringElement(element_shape, Size(n, n));
	erode(_src, _dst, element, Point(-1, -1));
	erode(_dst, _dst, element, Point(-1, -1), 4);
	dilate(_dst, _dst, element, Point(-1, -1), 2);
}
// Print the result of each object
void Print_Result(void)
{
	cout << "# of \t\t Bolt \t\t M5 = "		<< Bolt_M5 << endl;
	cout << "# of \t\t Bolt \t\t M6 = "		<< Bolt_M6 << endl;
	cout << "# of \t\t Square Nut \t M5 = " << Square_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M5 = "	<< Hexa_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M6 = "	<< Hexa_Nut_M6 << endl;
}