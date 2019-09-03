/*2019-09-03 Alex
项目说明：该项目主要是通过读取Mat图像，对OpenCV的函数进行重写，
之后调试正常后，把函数用于图像大师的函数中。
*/

#include "pch.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include"MyOpenCV.h"

using namespace std;
using namespace cv;

int main()
{
	Mat src,dst;
	src = imread("D:/CODE/MyOpenCV3/1.jpg");
	MyOpencv m;
	dst = m.M_resize_zjl(src,dst,800,600);
	
	imshow("show",dst);

	waitKey(0);
}
