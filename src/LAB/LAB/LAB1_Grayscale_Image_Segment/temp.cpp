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


// Define the Value
#define KERNEL_SIZE 5
enum Filter_Mode {
	BLUR = 1, GAUSSIAN, MEDIAN, LAPLACIAN, DEFAULT
};

Mat src, src_gray, src_filtered, src_thred, dst, dst_morph;
int element_shape = MORPH_RECT;
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));
int gray = 255;
int Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;
String raw_data = "Raw_Data";

// Define Function
void Image_Processing(void);
void Convert_Gray(Mat& _src, Mat& _dst);
void Print_Raw(void);
void Filter_Process(void);
void Threshold_Process(void);
void Filled_Contour(void);
void Morphology_Process(void);
void Sorting(void);
void View_Result(void);
void Print_Result(void);

// Define Source Matrix


void main(void)
{
	// Load an image
	src = imread("..//..//..//Images//Lab_GrayScale_TestImage.jpg", IMREAD_COLOR);

	// Convert the image to Gray
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/* Print the Raw Data */
	namedWindow("Raw_Data", 0);
	resizeWindow("Raw_Data", Size(800, 800));
	imshow("Raw_Data", src_gray);

	/* Filter */
	medianBlur(src_gray, src_filtered, 5);

	/* Threshold */
	threshold(src_filtered, dst, 116, 255, 0);

	/* Contour for FILLED */
	vector<vector<Point>> contours;
	Mat drawing(dst.size(), CV_8U, Scalar(0));
	findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
		drawContours(drawing, contours, i, Scalar(255), CV_FILLED, 8);
	drawing.copyTo(dst);

	/* Mophology */
	erode(dst, dst_morph, element, Point(-1, -1));
	erode(dst_morph, dst_morph, element, Point(-1, -1), 4);
	dilate(dst_morph, dst_morph, element, Point(-1, -1), 2);


	Mat drawing2(dst_morph.size(), CV_8U, Scalar(0));
	findContours(dst_morph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


	for (int i = 0; i < contours.size(); i++)
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
		drawContours(drawing2, contours, i, Scalar(gray), CV_FILLED, 8);
	}
	drawing2.copyTo(dst_morph);

	while (true)
	{
		namedWindow("result", 0);
		imshow("result", dst_morph);
		resizeWindow("result", Size(800, 800));

		int key = waitKey(20);
		if (key == 27)	break;
	}

	cout << "# of \t\t Bolt \t\t M5 = " << Bolt_M5 << endl;
	cout << "# of \t\t Bolt \t\t M6 = " << Bolt_M6 << endl;
	cout << "# of \t\t Square Nut \t M5 = " << Square_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M5 = " << Hexa_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M6 = " << Hexa_Nut_M6 << endl;
}

void Convert_Gray(Mat& _src, Mat& _dst)
{
	cvtColor(_src, _dst, CV_BGR2GRAY);
}

void Print_Window(String& _message, Mat& _src)
{
	namedWindow(_message, 0);
	resizeWindow(_message, Size(800, 800));
	imshow(_message, _src);
}

void Filter_Process(Mat& _src, Mat& _dst, int _filter, int _kernel)
{
	if (_filter == BLUR)			blur(_src, _dst, cv::Size(_kernel, _kernel), cv::Point(-1, -1));
	else if (_filter == GAUSSIAN)		GaussianBlur(_src, _dst, cv::Size(_kernel, _kernel), 0, 0);
	else if (_filter == MEDIAN)			medianBlur(_src, _dst, _kernel);
}

void Threshold_Process(Mat& _src, Mat& _dst)
{
	threshold(src_filtered, dst, 116, 255, 0);
}

void Filled_Contour(void)
{
	vector<vector<Point>> contours;
	Mat drawing(dst.size(), CV_8U, Scalar(0));
	findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int i = 0; i < contours.size(); i++)
		drawContours(drawing, contours, i, Scalar(255), CV_FILLED, 8);
	drawing.copyTo(dst);
}

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

void Morphology_Process(Mat& _src, Mat& _dst)
{
	erode(_src, _dst, element, Point(-1, -1));
	erode(_dst, _dst, element, Point(-1, -1), 4);
	dilate(_dst, _dst, element, Point(-1, -1), 2);
}

void Sorting(void)
{
	vector<vector<Point>> contours;
	Mat drawing(dst_morph.size(), CV_8U, Scalar(0));
	findContours(dst_morph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int Bolt_M5 = 0, Bolt_M6 = 0, Square_Nut_M5 = 0, Hexa_Nut_M5 = 0, Hexa_Nut_M6 = 0;
	for (int i = 0; i < contours.size(); i++)
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
		drawContours(drawing, contours, i, Scalar(gray), CV_FILLED, 8);
	}
	drawing.copyTo(dst_morph);
}
void View_Result(Mat& _src)
{
	namedWindow("result", 0);
	imshow("result", dst_morph);
	resizeWindow("result", Size(800, 800));
}
void Print_Result(void)
{
	cout << "# of \t\t Bolt \t\t M5 = " << Bolt_M5 << endl;
	cout << "# of \t\t Bolt \t\t M6 = " << Bolt_M6 << endl;
	cout << "# of \t\t Square Nut \t M5 = " << Square_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M5 = " << Hexa_Nut_M5 << endl;
	cout << "# of \t\t Hexa Nut \t M6 = " << Hexa_Nut_M6 << endl;
}

void Image_Processing(void)
{
	Convert_Gray(src, src_gray);
	Print_Window(raw_data, src_gray);
	Filter_Process(src_gray, src_filtered, MEDIAN, KERNEL_SIZE);
	Threshold_Process(src_filtered, dst);
	Contour_Process(dst, true);
	Morphology_Process(dst, dst_morph);
	Contour_Process(dst_morph, false);

}

