
/* 

	DLIP MIDTERM 2022  Submission


	NAME:ChangMin An

*/


#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



int hmin = 0, hmax = 15, smin = 55, smax = 130, vmin = 145, vmax = 255;
int counting = 0;

// Global variables for Morphology
int element_shape = MORPH_RECT;		// MORPH_RECT, MORPH_ELIPSE, MORPH_CROSS
int n = 3;
Mat element = getStructuringElement(element_shape, Size(n, n));

int main()
{
	Mat src, src2, hsv, src_gray, dstSegment, mask, roi, dst;
	vector<vector<Point> > contours;

	src = imread("groupimage_crop.jpg");
	src.copyTo(src2);
	src.copyTo(dst);
	namedWindow("src", 1);
	imshow("source", src);
	

	/******** Segmatation of Facial Area  ********/
	//  Segmentatation result is in B/W image, with white for facial area
	cvtColor(src, hsv, COLOR_BGR2HSV);

	inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
		Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), src_gray);

	erode(src_gray, src_gray, element, Point(-1, -1));
	dilate(src_gray, src_gray, element, Point(-1, -1));
	erode(src_gray, src_gray, element, Point(-1, -1), 2);
	dilate(src_gray, src_gray, element, Point(-1, -1), 2);
	
	findContours(src_gray, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Mat drawing(src_gray.size(), CV_8U, Scalar(0));
	for (int i = 0; i < contours.size(); i++)
		if (contourArea(contours[i]) > 2000)
		{
			drawContours(drawing, contours, i, Scalar(255), CV_FILLED, 8);
			counting++;
		}
		else
			drawContours(drawing, contours, i, Scalar(0), CV_FILLED, 8);

	drawing.copyTo(src_gray);
	dilate(src_gray, src_gray, element, Point(-1, -1), 4);
	erode(src_gray, src_gray, element, Point(-1, -1), 2);

	src_gray.copyTo(dstSegment);

	imshow("dstSegment", dstSegment);


	/******** Count Number of Faces  ********/
	//  Print the text on source image
	String Human_String = "Human: ";
	String Human_Value = to_string(counting);
	String Human = Human_String + Human_Value;
	putText(src, Human, Point(40, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
	imshow("Part_B_Put_text", src);
	

	/******** Draw Rectangles around Faces  ********/
	// Show Tight Fit Rectangle on both  source image(src) and on (dstSegment)
	findContours(dstSegment, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Rect boxPoint = boundingRect(contours[0]);
	roi = Mat::zeros(src.size(), CV_8UC3);
	mask = Mat::zeros(src.size(), CV_8UC1);
	Mat blur_img = Mat::zeros(src.size(), CV_8UC3);
	src.copyTo(roi);

	for (int i = 0; i < counting; i++)
	{
		boxPoint = boundingRect(contours[i]);
		rectangle(dstSegment, boxPoint, Scalar(255, 0, 0), 3);
		rectangle(src, boxPoint, Scalar(255, 0, 0), 3);
		rectangle(mask, boxPoint, Scalar(255, 0, 0), 3);
	}

	imshow("Part_C_dstSegment2", dstSegment);
	imshow("Part_C_src", src);

	/******** Blur Faces  ********/
	// Show Blurred faces on  source image(src)
	findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
		drawContours(mask, contours, i, Scalar(255), CV_FILLED, 8);
	
	// Masking Face
	bitwise_and(roi, roi, blur_img, mask);

	//Blurring
	blur(blur_img, blur_img, Size(11, 11));

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < blur_img.cols*3; j++)
		{
			if (blur_img.at<char>(i, j) != 0)
				dst.at<char>(i, j) = blur_img.at<char>(i, j);
			else
				dst.at<char>(i, j) = dst.at<char>(i, j);
		}
	}
	
	imshow("mask", mask);
	imshow("blur_img", blur_img);
	imshow("Part_D_Result", dst);



	
	waitKey(0);
	return 0;
}

