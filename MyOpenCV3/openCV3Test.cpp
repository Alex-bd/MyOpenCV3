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
	//src = imread("D:/CODE/MyOpenCV3/1.jpg");
	src = imread("D:/CODE/ImageDashi/ImageDashi/ImageDashi/Image/edit/img_black.png");
	//经上面的测试，说明.jpg格式和.png格式的文件都能正常运行
	MyOpencv m;
	//dst = m.M_resize_zjl(src,800,600);	//最近邻	
	//dst = m.M_resize_sxx(src,400,400);	//双线性内插
	//dst = m.fangshe(src,-45);			//仿射变换
	//dst = m.flip(src , -1);				//镜像
	//dst = m.rotateImage(src,180,0);		//任意角旋转
	//dst = m.threshold(src,127);			//二值化
	//dst = m.gamma(src,100,255);				//伽马变换
	dst = m.Log(src,178);					//对数变换
	imshow("show",dst);

	waitKey(0);
}
