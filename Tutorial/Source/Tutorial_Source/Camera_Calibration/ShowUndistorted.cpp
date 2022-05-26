#include <iostream>
#include <opencv.hpp>



int main()
{
	cv::Mat src, result, cameraMatrix, distCoeffs;
	src = cv::imread("calibTest.jpg");
 
	double fx, fy, cx, cy, k1, k2, p1, p2;


	fx = 834.58755288917052;
	fy = 847.42845271147621;
	cx = 593.71226204720074;
	cy = 355.08244223377216;
	k1 = -0.4079357365071829;
	k2 = 0.14181447141681008;
	p1 = -0.001915596395994537;
	p2 = 0.00723345381229942;
	
	cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
	cameraMatrix.at<double>(0, 0) = fx;
	cameraMatrix.at<double>(0, 2) = cx;
	cameraMatrix.at<double>(1, 1) = fy;
	cameraMatrix.at<double>(1, 2) = cy;


	distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
	distCoeffs.at<double>(0, 0) = k1;
	distCoeffs.at<double>(1, 0) = k2;
	distCoeffs.at<double>(2, 0) = p1;
	distCoeffs.at<double>(3, 0) = p2;



	cv::undistort(src, result, cameraMatrix, distCoeffs);



	cv::imshow("SRC",	src);
	cv::imshow("result", result);
	cv::waitKey(0);
	return 0;
}
 