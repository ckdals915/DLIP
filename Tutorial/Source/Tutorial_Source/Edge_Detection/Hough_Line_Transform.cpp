#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// Declare the output variables
	Mat dst, cdst, cdstP;

	// Loads an image
	const char* filename = "../..//..//images/track_gray.JPG";
	Mat src = imread(filename, IMREAD_GRAYSCALE);

	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		return -1;
	}

	Mat dst_temp;
	// Edge detection
	Canny(src, dst, 200, 200, 3);
	
	Canny(src, dst_temp, 200, 200, 3);

	// Copy edge results to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cdstP = cdst.clone();

	// (Option 1) Standard Hough Line Transform
	//Vec2f: float단위의 2D Array (N, 2) (rho, theta)
	vector<Vec2f> lines;
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0);

	// Draw the detected lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);		// 이미지에 선을 그리는 함수(print하는것)
	}

	// (Option 2) Probabilistic Line Transform
	vector<Vec4i> linesP;
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // 10: GAP 하나의 라인으로 볼건지 따로 볼건지, 뒷 50: 최소 선 길이

	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}

	// Show results
	imshow("Source", src);
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);

	// Wait and Exit
	waitKey();
	return 0;
}