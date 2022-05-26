/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV Tutorial: Video Open/Display
* Created: 2021-Spring
------------------------------------------------------*/
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	bool flip_flag = false;
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);


	while (1)
	{
		Mat frame;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(frame);


		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}

		// input key in
		int inKey = waitKey(30);

		// Operating
		if (inKey == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}

		else if (inKey == 'h' || inKey == 'H')
			flip_flag = !flip_flag;
	
		if (flip_flag)
			flip(frame, frame, 1);
		
		imshow("myVideo", frame);
	}
}