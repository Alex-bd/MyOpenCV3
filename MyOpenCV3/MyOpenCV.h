/*2019-09-03 Alex
项目说明：该项目主要是通过读取Mat图像，对OpenCV的函数进行重写，
之后调试正常后，把函数用于图像大师的函数中。
*/
#pragma once
//重写OpenCV的头文件
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
class MyOpencv
{
public:
	MyOpencv();
	~MyOpencv();
	//函数声明：
	Mat M_resize_zjl(Mat src,Mat dst,int x,int y);	//最近邻 重置大小

private:

};

MyOpencv::MyOpencv()
{
}

MyOpencv::~MyOpencv()
{
}
//最近邻 重置大小
/*参数1：源图像
  参数2：目标图像
  参数3：weight
  参数4：height
*/
Mat MyOpencv::M_resize_zjl(Mat src,Mat dst,int x,int y)
{
	//下面的这几行都是一致的
	Mat matSrc, matDst1, matDst2;			//声明图像 源变量和目标变量
	matSrc = src;							//获取源
	//Mat  (大小，类型，三原色数值)
	matDst1 = Mat(Size(x,y),matSrc.type(),Scalar::all(0));	//把目标像素矩阵元素全设置为0
	//公式：srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double) matSrc.cols / matDst1.cols;
	double scale_y = (double) matSrc.rows / matDst1.cols;
	//最近邻关键步骤
	//整个双层循环就是再利用上述公式对目标像素矩阵的每个像素点进行赋值
	for (int i=0; i < matDst1.cols; ++i)	//遍历列像素点
	{
		int sx = cvFloor(i*scale_x);	//cvFloor:返回不大于参数的最大整数值
		sx = min(sx, matSrc.cols - 1);
		for (int  j = 0; j < matDst1.rows; ++j)
		{
			int sy = cvFloor(j*scale_y);
			sy = min(sy, matSrc.rows-1);
			matDst1.at<Vec3b>(j, i) = matSrc.at<Vec3b>(sy,sx);//赋值
		}
	}
	//imwrite("nearest_1.jpg",matDst1);

	return matDst1;
}