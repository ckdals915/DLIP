#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat src, gray;

	String filename = "..//..//..//Images//TrafficSign1.png";

	/* Read the image */
	src = imread(filename, 1);

	if (!src.data)
	{
		printf(" Error opening image\n");
		return -1;
	}

	cvtColor(src, gray, COLOR_BGR2GRAY);

	/* smooth it, otherwise a lot of false circles may be detected */
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	//(N, 3) 2D Array
	vector<Vec3f> circles;
	HoughCircles(gray, circles, 3, 2, gray.rows / 4, 150, 200, 200, 300);
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		/* draw the circle center */
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		/* draw the circle outline */
		circle(src, center, radius, Scalar(255, 0, 0), 3, 8, 0);
	}

	vector<Vec3f> circles2;
	HoughCircles(gray, circles2, 3, 2, gray.rows / 4, 100, 150, 100, 200);
	for (size_t i = 0; i < circles2.size(); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);

		/* draw the circle center */
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);

		/* draw the circle outline */
		circle(src, center, radius, Scalar(255, 0, 0), 3, 8, 0);
	}

	namedWindow("circles", 1);
	imshow("circles", src);

	/* Wait and Exit */
	waitKey();
	return 0;
}