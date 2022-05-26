//#include "opencv2/video/tracking.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <ctype.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat src, dst;
Point origin;
Rect selection;

bool selectObject = false;
bool trackObject = false;
int hmin = 1, hmax = 179, smin = 0, smax = 0, vmin = 0, vmax = 0;
bool contourDraw_FLAG = false;

/// On mouse event 
static void onMouse(int event, int x, int y, int, void*);

int main()
{
	Mat image_disp, hsv, hue, mask;
	vector<vector<Point> > contours;

	VideoCapture cap(0);
	int cam_W = 640;
	int cam_H = 480;
	cap.set(CAP_PROP_FRAME_WIDTH, cam_W);
	cap.set(CAP_PROP_FRAME_HEIGHT, cam_H);

	Mat dst_track = Mat::zeros(cam_H, cam_W, CV_8UC3);

	

	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	// TrackBar 설정
	namedWindow("Source", 0);
	setMouseCallback("Source", onMouse, 0);
	createTrackbar("Hmin", "Source", &hmin, 179, 0);
	createTrackbar("Hmax", "Source", &hmax, 179, 0);
	createTrackbar("Smin", "Source", &smin, 255, 0);
	createTrackbar("Smax", "Source", &smax, 255, 0);
	createTrackbar("Vmin", "Source", &vmin, 255, 0);
	createTrackbar("Vmax", "Source", &vmax, 255, 0);

	

	while (true)
	{
		

		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		src.copyTo(image_disp);
		

		imshow("Source", src);
		/******** Convert BGR to HSV ********/
		// input mat: image
		// output mat: hsv
		cvtColor(src, hsv, COLOR_BGR2HSV);


		/******** Add Pre-Processing such as filtering etc  ********/
		// YOUR CODE GOES HERE
		// YOUR CODE GOES HERE



		/// set dst as the output of InRange
		inRange(hsv, Scalar(MIN(hmin, hmax), MIN(smin, smax), MIN(vmin, vmax)),
			Scalar(MAX(hmin, hmax), MAX(smin, smax), MAX(vmin, vmax)), dst);


		/******** Add Post-Processing such as morphology etc  ********/
		// YOUR CODE GOES HERE
		// YOUR CODE GOES HERE



		namedWindow("InRange", 0);
		imshow("InRange", dst);

		/// once mouse has selected an area bigger than 0
		if (trackObject)
		{
			trackObject = false;					// Terminate the next Analysis loop
			Mat roi_HSV(hsv, selection); 			// Set ROI by the selection box		
			Scalar means, stddev;
			meanStdDev(roi_HSV, means, stddev);
			cout << "\n Selected ROI Means= " << means << " \n stddev= " << stddev;

			// Change the value in the trackbar according to Mean and STD //
			hmin = MAX((means[0] - stddev[0]), 0);
			hmax = MIN((means[0] + stddev[0]), 179);
			setTrackbarPos("Hmin", "Source", hmin);
			setTrackbarPos("Hmax", "Source", hmax);

			/******** Repeat for S and V trackbar ********/
			smin = MAX((means[1] - stddev[1]), 0);
			smax = MIN((means[1] + stddev[1]), 255);
			setTrackbarPos("Smin", "Source", smin);
			setTrackbarPos("Smax", "Source", smax);

			vmin = MAX((means[2] - stddev[2]), 0);
			vmax = MIN((means[2] + stddev[2]), 255);
			setTrackbarPos("Vmin", "Source", vmin);
			setTrackbarPos("Vmax", "Source", vmax);

		}


		if (selectObject && selection.area() > 0)  // Left Mouse is being clicked and dragged
		{
			// Mouse Drag을 화면에 보여주기 위함
			Mat roi_RGB(image_disp, selection);
			bitwise_not(roi_RGB, roi_RGB);
			imshow("Source", image_disp);
		}
		src.copyTo(image_disp);



		///  Find All Contour   ///
		findContours(dst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if (contours.size() > 0)
		{
			/// Find the Contour with the largest area ///
			double maxArea = 0;
			int maxArea_idx = 0;

			for (int i = 0; i < contours.size(); i++)
				if (contourArea(contours[i]) > maxArea) {
					maxArea = contourArea(contours[i]);
					maxArea_idx = i;
				}

			///  Draw the max Contour on Black-background  Image ///
			Mat dst_out = Mat::zeros(dst.size(), CV_8UC3);
			drawContours(dst_out, contours, maxArea_idx, Scalar(0, 0, 255), 2, 8);
			namedWindow("Contour", 0);
			imshow("Contour", dst_out);


			/// Draw the Contour Box on Original Image ///
			Rect boxPoint = boundingRect(contours[maxArea_idx]);
			rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);
			namedWindow("Contour_Box", 0);
			imshow("Contour_Box", image_disp);


			/// Continue Drawing the Contour Box  ///
			if (contourDraw_FLAG)
			{
				rectangle(dst_track, boxPoint, Scalar(255, 0, 255), 3);
				namedWindow("Contour_Track", 0);
				imshow("Contour_Track", dst_track);
				dst_track *= 0.95;		//잔상: 시간이 지날수록 점점 검은색으로 가게 하는것
			}
			
		}

		char c = (char)waitKey(10);
		if (c == 27)						break;
		else if (c == 'c' || c == 'C')		contourDraw_FLAG = true;
		else if (c == 'o' || c == 'O')		dst_track = Mat::zeros(cam_H, cam_W, CV_8UC3);
	} // end of for(;;)

	return 0;
}



/// On mouse event 
static void onMouse(int event, int x, int y, int, void*)
{
	if (selectObject)  // for any mouse motion
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x) + 1;
		selection.height = abs(y - origin.y) + 1;
		selection &= Rect(0, 0, src.cols, src.rows);  /// Bitwise AND  check selectin is within the image coordinate 안전장치(image 내에서만 드래그 할 수 있도록)
	}

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
		selectObject = true;
		origin = Point(x, y);
		break;
	case CV_EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.area())
			trackObject = true;
		break;
	}
}