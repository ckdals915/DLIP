/*------------------------------------------------------/
* Image Proccessing with Deep Learning
* OpenCV : Filter Demo - Video
* Created: 2021-Spring
------------------------------------------------------*/

#include <opencv2/opencv.hpp>
#include <iostream>

#define KERNEL_SCALE 2
#define KERNEL_MAX   11
#define KERNEL_MIN   3

enum Filter_Mode {
	BLUR = 1, GAUSSIAN, MEDIAN, LAPLACIAN, DEFAULT
};



using namespace std;
using namespace cv;

int main()
{
	/*  open the video camera no.0  */
	VideoCapture cap(0);

	if (!cap.isOpened())	// if not success, exit the programm
	{
		cout << "Cannot open the video cam\n";
		return -1;
	}

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	int key = 0;
	int kernel_size = 3;	//default
	int filter_type = 0;

	// Laplacian Variable
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	while (1)
	{
		Mat src, dst;

		/*  read a new frame from video  */
		bool bSuccess = cap.read(src);

		if (!bSuccess)	// if not success, break loop
		{
			cout << "Cannot find a frame from  video stream\n";
			break;
		}


		key = waitKey(30);
		if (key == 27) // wait for 'ESC' press for 30ms. If 'ESC' is pressed, break loop
		{
			cout << "ESC key is pressed by user\n";
			break;
		}

		else if (key == 'b' || key == 'B')		filter_type = BLUR;			// blur

		else if (key == 'g' || key == 'G')		filter_type = GAUSSIAN;		// gaussian

		else if (key == 'm' || key == 'M')		filter_type = MEDIAN;		// median

		else if (key == 'l' || key == 'L')		filter_type = LAPLACIAN;	// Laplacian

		else if (key == 'o' || key == 'O')		filter_type = DEFAULT;

		else if (key == 'w' || key == 'W')		kernel_size += KERNEL_SCALE;

		else if (key == 's' || key == 'S')		kernel_size -= KERNEL_SCALE;

		else									filter_type = filter_type;

		// Saturation of kernel_size 1~11
		if		(kernel_size >= KERNEL_MAX)		kernel_size = KERNEL_MAX;
		else if (kernel_size <= KERNEL_MIN)		kernel_size = KERNEL_MIN;
		else									kernel_size = kernel_size;

		if (filter_type == BLUR)				blur(src, dst, cv::Size(kernel_size, kernel_size), cv::Point(-1, -1));

		else if (filter_type == GAUSSIAN)		GaussianBlur(src, dst, cv::Size(kernel_size, kernel_size), 0, 0);

		else if (filter_type == MEDIAN)			medianBlur(src, dst, kernel_size);

		else if (filter_type == LAPLACIAN)
		{
			Laplacian(src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
			src.convertTo(src, CV_16S);
			dst = src - dst;
			dst.convertTo(dst, CV_8U);
		}

		else if (filter_type == DEFAULT)		src.copyTo(dst);

		else									src.copyTo(dst);

		imshow("MyVideo", dst);
	}
	return 0;
}