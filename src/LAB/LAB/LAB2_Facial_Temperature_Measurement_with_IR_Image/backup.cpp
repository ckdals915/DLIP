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
bool warning_Flag = false;
bool recognition = false;

int hmin = 0, hmax = 55, smin = 60, smax = 255, vmin = 100, vmax = 255;

bool contourDraw_FLAG = false;




/// On mouse event 
static void onMouse(int event, int x, int y, int, void*);
void plotHist(Mat src, string plotname, int width, int height);

int main()
{
	Mat image_disp, hsv, hue, mask, image_gray, roi_gray;
	vector<Mat> hsv_split;
	vector<vector<Point> > contours;

	VideoCapture cap("..//..//..//Images//IR_DEMO_cut.avi");
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

	namedWindow("VideoSource", CV_WINDOW_AUTOSIZE);

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
		// split the Matrix
		split(hsv, hsv_split);
		/*imshow("H", hsv_split[0]);
		imshow("S", hsv_split[1]);
		imshow("V", hsv_split[2]);*/

		// Median filter about V, kernel_Size: 9
		//medianBlur(hsv_split[2], image_gray, 5);
		hsv_split[2].copyTo(image_gray);
		imshow("Gray_Scale", image_gray);

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
			{
				if (contourArea(contours[i]) > 13000)
					contourDraw_FLAG = true;
				else
					contourDraw_FLAG = false;

				if (contourArea(contours[i]) > maxArea) {

					//printf("%.2f\n", contourArea(contours[i]));
					maxArea = contourArea(contours[i]);
					maxArea_idx = i;
				}
			}


			Rect boxPoint = boundingRect(contours[maxArea_idx]);
			Mat dst_out = Mat::zeros(dst.size(), CV_8UC3);
			mask = Mat::zeros(dst.size(), CV_8UC1);
			roi_gray = Mat::zeros(dst.size(), CV_8UC1);

			/// Continue Drawing the Contour Box  ///
			if (contourDraw_FLAG)
			{
				///  Draw the max Contour on Black-background  Image ///
				recognition = true;
				drawContours(dst_out, contours, maxArea_idx, Scalar(255, 255, 255), 2, 8);

				// Masking
				drawContours(mask, contours, maxArea_idx, Scalar(255, 255, 255), CV_FILLED, 8);


				image_disp += dst_out;

				/// Draw the Contour Box on Original Image ///

				rectangle(image_disp, boxPoint, Scalar(255, 0, 255), 3);

				rectangle(dst_track, boxPoint, Scalar(255, 0, 255), 3);



				dst_track *= 0.5;		//잔상: 시간이 지날수록 점점 검은색으로 가게 하는것
			}

			// bitwise_and
			bitwise_and(image_gray, image_gray, roi_gray, mask);

			
			imshow("roi_gray", roi_gray);
			//resizeWindow("roi_gray", Size(800, 800));
			// Sorting
			Mat dst_sort = roi_gray.reshape(0, 1);
			cv::sort(dst_sort, dst_sort, SORT_DESCENDING);


			// 상위 5% 값 더하기

			float sum_gray = 0.0;
			int counting = 0;
			float user_scale = 0.0;
			float max_gray = 0.0;


			for (int i = 0; i < (dst_sort.cols) / 400; i++)
			{
				if (dst_sort.at<uchar>(0, i) != 0)
				{
					if (max_gray <= dst_sort.at<uchar>(0, i))
						max_gray = dst_sort.at<uchar>(0, i);
					sum_gray += dst_sort.at<uchar>(0, i);
					counting++;
				}
			}


			if (counting != 0)
				user_scale = (float)(sum_gray / (float)counting);

			// Fitting from gray-scale to Celsius
			int temp = (15.0 / 255.0 * user_scale + 25.0);
			int max_temp = (15.0 / 255.0 * max_gray + 25.0);
			//printf("max Temp = %d\t avg = %d\n", max_temp, temp);

			if (temp >= 38)
				warning_Flag = true;

			String Text_max = "Max: ";
			String max = to_string(max_temp);
			String avg = to_string(temp);
			String Text_avg = "AVG: ";
			String Warning = "WARNING";

			String Text1 = Text_max + max;
			String Text2 = Text_avg + avg;

			if (recognition == true)
			{
				if (warning_Flag == true)
				{
					putText(image_disp, Text1, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
					putText(image_disp, Text2, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
					putText(image_disp, Warning, Point(20, 70), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
				}
				else
				{
					putText(image_disp, Text1, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
					putText(image_disp, Text2, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
				}
			}

			namedWindow("roi_gray", 0);
			imshow("roi_gray", roi_gray);
			imshow("image_gray", image_gray);

			imshow("mask", mask);
			namedWindow("Contour_Track", 0);
			imshow("Contour_Track", dst_track);
			namedWindow("Contour", 0);
			imshow("Contour", dst_out);
			namedWindow("Contour_Box", 0);
			imshow("Contour_Box", image_disp);
			namedWindow("Src", 0);

			recognition = false;
			warning_Flag = false;

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