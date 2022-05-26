/*
* *****************************************************************************
* @author	ChangMin An
* @Mod		2022 - 04 - 07
* @brief	LAB: Facial Temperature Measurement with IR images
******************************************************************************
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Function Declaration
void Video_Setting(void);
void Read_Video(void);
void Pre_Processing(void);
void Processing(void);
int	 Pixel2Cel(float _pixel);
void Print_Result(void);
void Flag_Update(void);
void View_Result(void);
void plotHist(Mat src, string plotname, int width, int height);

//=============== Global Variable =================
// Matrix
Mat src, dst, image_disp, image_gray, hsv, mask, roi_gray, dst_1Dsort;
vector<Mat> hsv_split;
vector<vector<Point>> contours;
VideoCapture cap("..//..//..//Images//IR_DEMO_cut.avi");

// Flag
bool warning_Flag = false;
bool text_Flag = false;
bool contourDraw_Flag = false;
bool breaking_Flag = false;

// hsv
#define HMIN 0
#define HMAX 55
#define SMIN 60
#define SMAX 255
#define VMIN 100
#define VMAX 255

// Video Source
int cam_W = 640;
int cam_H = 480;

// Contour Area Range
#define CONTOUR_AREA 13000

// Counting about sorting data to Temperature
float sum_Gray = 0.0;
float avg_Gray = 0.0;
float max_Gray = 0.0;
int avg_Cel = 0;
int max_Cel = 0;
int counting = 0;

int main()
{
	Video_Setting();			// Setting Video Source

	while (true)
	{
		Read_Video();			// Read Video Captured Image
		if (breaking_Flag == true)			break;


		Pre_Processing();		// InRange Processing & Filtering
		Processing();			// Get ROI to use Contour, Calculate Temperature mapping intensity into temperature
		Print_Result();			// Print the Result about Temperature
		View_Result();			// View Video Output
		Flag_Update();			// Warning and Text Flag Update to false

		char c = (char)waitKey(10);
		if (c == 27)						break;
	}
	return 0;
}

// Pixel to Celsius Temperature Mapping
int Pixel2Cel(float _pixel)
{
	return(int)round(15.0 / 255.0 * _pixel + 25.0);
}

// Setting Video Source
void Video_Setting(void)
{
	cap.set(CAP_PROP_FRAME_WIDTH, cam_W);
	cap.set(CAP_PROP_FRAME_HEIGHT, cam_H);

	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam\n";
		breaking_Flag == true;
	}
}

// Read Video Captured Image
void Read_Video(void)
{
	bool bSuccess = cap.read(src);
	if (!bSuccess)
	{
		cout << "Cannot find a frame from video stream\n";
		breaking_Flag = true;
	}
}

// InRange Processing & Filtering
void Pre_Processing(void)
{
	// Copy Source Image to image_disp and BGR2HSV
	src.copyTo(image_disp);
	cvtColor(src, hsv, COLOR_BGR2HSV);

	// Split matrix to make gray scale temperature
	split(hsv, hsv_split);
	hsv_split[2].copyTo(image_gray);

	// Plot histogram of gray scale
	//plotHist(image_gray, "hist_gray", image_gray.cols, image_gray.rows);


	// set dst as the output of InRange
	inRange(hsv, Scalar(MIN(HMIN, HMAX), MIN(SMIN, SMAX), MIN(VMIN, VMAX)),
		Scalar(MAX(HMIN, HMAX), MAX(SMIN, SMAX), MAX(VMIN, VMAX)), dst);
}

// Get ROI to use Contour, Calculate Temperature mapping intensity into temperature
void Processing(void)
{
	// Find All Contour
	findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0)
	{
		double maxArea = 0;
		int maxArea_idx = 0;

		for (int i = 0; i < contours.size(); i++)
		{
			// Set Contour Minimum Area
			if (contourArea(contours[i]) > CONTOUR_AREA)	contourDraw_Flag = true;
			else											contourDraw_Flag = false;

			// Set Contour Maximum Area
			if (contourArea(contours[i]) > maxArea)
			{
				maxArea = contourArea(contours[i]);
				maxArea_idx = i;
			}
		}

		// Draw Rectangular Area
		Rect boxPoint = boundingRect(contours[maxArea_idx]);
		Mat dst_out = Mat::zeros(dst.size(), CV_8UC3);
		mask = Mat::zeros(dst.size(), CV_8UC1);
		roi_gray = Mat::zeros(dst.size(), CV_8UC1);

		// Continue Drawing the Contour Box
		if (contourDraw_Flag)
		{
			// Recognition about Temperature Text Printing
			text_Flag = true;
			drawContours(dst_out, contours, maxArea_idx, Scalar(255, 255, 255), 2, 8);

			// Masking about Contours Area
			drawContours(mask, contours, maxArea_idx, Scalar(255, 255, 255), CV_FILLED, 8);

			image_disp += dst_out;

			// Draw the Contour Box on Original Image
			rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);

		}
		
		// Draw ONLY ROI Area to use Masking matrix
		bitwise_and(image_gray, image_gray, roi_gray, mask);

		// Sorting roi gray-scale image to descending 1D array
		dst_1Dsort = roi_gray.reshape(0, 1);
		cv::sort(dst_1Dsort, dst_1Dsort, SORT_DESCENDING);
	}

	// Calculate Temperature to use ROI Area value
	counting = 0;
	sum_Gray = 0.0;
	max_Gray = 0.0;

	for (int i = 0; i < (dst_1Dsort.cols)/400; i++)
	{
		if (dst_1Dsort.at<uchar>(0, i) != 0)
		{
			if (max_Gray <= dst_1Dsort.at<uchar>(0, i))	
				max_Gray = dst_1Dsort.at<uchar>(0, i);
			sum_Gray += dst_1Dsort.at<uchar>(0, i);
			counting++;
		}
	}

	if (counting != 0)		
		avg_Gray = (float)(sum_Gray / (float)counting);

	// Fitting from gray-scale to Celsius
	avg_Cel = Pixel2Cel(avg_Gray);
	max_Cel = Pixel2Cel(max_Gray);

	// Warning about User Temperature up to 38
	if (avg_Cel >= 38)	warning_Flag = true;
}

// Print the Result about Temperature
void Print_Result(void)
{
	String max_String = "Max: ";
	String max_Value = to_string(max_Cel);
	String avg_String = "AVG: ";
	String avg_Value = to_string(avg_Cel);
	String warning = "WARNING";
	String max = max_String + max_Value;
	String avg = avg_String + avg_Value;

	if (text_Flag)
	{
		if (warning_Flag)
		{
			putText(image_disp, max, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
			putText(image_disp, avg, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
			putText(image_disp, warning, Point(20, 70), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
		}
		else
		{
			putText(image_disp, max, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
			putText(image_disp, avg, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
		}
	}
}

// View Video Output
void View_Result(void)
{
	namedWindow("Display", 0);
	imshow("Display", image_disp);
}

// Warning and Text Flag Update to false
void Flag_Update(void)
{
	text_Flag = false;
	warning_Flag = false;
}

// Plot Histogram
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
