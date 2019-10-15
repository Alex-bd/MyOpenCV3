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
	Mat M_resize_zjl(Mat src,int x,int y);	//最近邻 重置大小
	Mat M_resize_sxx(Mat src,int x,int y);	//双线性
	Mat xuanzhuan(Mat src, int angle);		//旋转 仿射变换
	Mat flip(Mat src,int flag);
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
Mat MyOpencv::M_resize_zjl(Mat src,int x,int y)
{
	//下面的这几行都是一致的
	Mat matSrc, matDst;			//声明图像 源变量和目标变量
	Mat matDst2;
	matSrc = src;							//获取源
	//Mat  (大小，类型，三原色数值)
	matDst = Mat(Size(x,y),matSrc.type(),Scalar::all(0));	//把目标像素矩阵元素全设置为0
	matDst2 = Mat(matDst.size(),matSrc.type(),Scalar::all(0));
															//公式：srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double) matSrc.cols / matDst.cols;
	double scale_y = (double) matSrc.rows / matDst.cols;
	//最近邻关键步骤
	//整个双层循环就是再利用上述公式对目标像素矩阵的每个像素点进行赋值
	for (int i=0; i < matDst.cols; ++i)	//遍历列像素点
	{
		int sx = cvFloor(i*scale_x);	//cvFloor:返回不大于参数的最大整数值
		sx = min(sx, matSrc.cols - 1);
		for (int  j = 0; j < matDst.rows; ++j)
		{
			int sy = cvFloor(j*scale_y);
			sy = min(sy, matSrc.rows-1);
			matDst.at<Vec3b>(j, i) = matSrc.at<Vec3b>(sy,sx);//赋值
		}
	}
	//return matDst;
	resize(matSrc,matDst2,matDst.size(),0,0,1);
	return matDst2;
}
//双线性插值
Mat MyOpencv::M_resize_sxx(Mat src,int x,int y)
{
	Mat matSrc, matDst;			//声明图像 源变量和目标变量
	//Mat matDst2;	//调用resize用于对比的目标图像
	matSrc = src;							//获取源
	//Mat  (大小，类型，三原色数值)
	matDst = Mat(Size(x, y), matSrc.type(), Scalar::all(0));	//把目标像素矩阵元素全设置为0
	//matDst2 = Mat(matDst.size(), matSrc.type(), Scalar::all(0));
	//公式：srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double)matSrc.cols / matDst.cols;
	double scale_y = (double)matSrc.rows / matDst.cols;

	uchar * dataDst = matDst.data;		//定义一个指针指向目标图像的数据域
	int stepDst = matDst.step;
	uchar * dataSrc = matSrc.data;		//指针指向源图像的数据域
	int stepSrc = matSrc.step;
	int iWidthSrc = matSrc.cols;	//源图像列数
	int iHeightSrc = matSrc.rows;	//源图像行数

	for(int j = 0;j < matDst.rows; j++)
	{
		float fy = (float)((j + 0.5)*scale_y - 0.5);	//浮点数
		int sy = cvFloor(fy);		//求y 整数
		fy -= sy;					//只保留小数
		sy = min(sy,iHeightSrc -2);
		sy = max(0,sy);				//求得坐标
		
		short cbufy[2];				//定义一个数组	
		cbufy[0] = saturate_cast<short>((1.f - fy) * 2048);
		cbufy[1] = 2048;		

		for (int i = 0; i < matDst.cols; i++)
		{
			float fx = (float)((i + 0.5)*scale_x - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 0)
			{
				fx = 0, sx = 0;
			}
			if (sx >= iWidthSrc -1)
			{
				fx = 0, sx = iWidthSrc - 2;
			}

			short cbufx[2];
			cbufx[0] = saturate_cast<short>((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];

			for (int  k = 0; k < matSrc.channels(); k++)
			{
				*(dataDst + j * stepDst + 3 * i + k) = (*(dataSrc + sy * stepSrc + 3 * sx + k) * cbufx[0] * cbufy[0] +
					*(dataSrc + (sy + 1) * stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy * stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;
			}
		}
	}
	return matDst;
	//resize(matSrc,matDst2,matDst.size(),0,0,1);
	//return matDst2;
}

//最近邻 仿射变换
Mat MyOpencv::xuanzhuan(Mat src, int angle)
{
	Mat matSrc, matDst;
	matSrc = src;
	double degree = angle;	//旋转角度
	double angles = degree* CV_PI / 180;
	double alpha = cos(angles);
	double beta = sin(angles);
	int iwidth = matSrc.cols;
	int iheight = matSrc.rows;
	int iNewWidth = cvRound(iwidth * fabs(alpha)+ iheight *fabs(beta));
	int iNewHeight = cvRound(iheight * fabs(alpha) + iwidth * fabs(beta));

	double m[6];			//变换矩阵  
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1 - alpha) * iwidth / 2 - beta * iheight / 2;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = beta * iwidth / 2 + (1 - alpha)* iheight / 2;

	Mat M = Mat(2,3,CV_64F,m);
	matDst = Mat(Size(iNewWidth,iNewHeight),matSrc.type(),Scalar::all(0));		//全赋值为0的图像

	double D = m[0] * m[4] - m[1] * m[3];
	D = D != 0 ? 1. / D : 0;
	double A11 = m[4] * D, A22 = m[0] * D;
	m[0] = A11; m[1] *= -D;
	m[3] *= -D; m[4] = A22;
	double b1 = -m[0] * m[2] - m[1] * m[5];
	double b2 = -m[3] * m[2] - m[4] * m[5];
	m[2] = b1; m[5] = b2;

	int round_delta = 512;
	for (int  y = 0; y < iNewHeight; y++)
	{
		for (int  x = 0;  x < iNewWidth;  x++)
		{
			int adelta = saturate_cast<int>(m[0]* x * 1024);
			int bdelta = saturate_cast<int>(m[3]* y * 1024);
			int X0 = saturate_cast<int>((m[1] * y + m[2]) * 1024) + round_delta;
			int Y0 = saturate_cast<int>((m[4] * y + m[5]) * 1024) + round_delta;
			int X = (X0 + adelta) >> 10;
			int Y = (Y0 + bdelta) >> 10;

			if ((unsigned)X < iwidth && (unsigned)Y < iheight)
			{
				matDst.at<cv::Vec3b>(y, x) = matSrc.at<cv::Vec3b>(Y, X);
			}
		}
	}
	return matDst;
}

//镜像
Mat MyOpencv::flip(Mat src,int flag)
{
	Mat matSrc, matDst;
	matSrc = src;
	int width = matSrc.cols;
	int height = matSrc.rows;
	matDst = Mat(Size(width,height), matSrc.type(), Scalar::all(0));
	if (flag == 0)//水平镜像
	{
		for (int y = 0; y < matSrc.cols; y++)
		{
			for (int x = 0; x < matSrc.rows; x++)
			{
				matDst.at<cv::Vec3b>(y,x) = matSrc.at<cv::Vec3b>(y,width-x-1);
			}
		}
	}
	else if (flag > 0)	//垂直镜像
	{

	}
	else
	{

	}
	return matDst;
}