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
					*(dataSrc + (sy + 1)*stepSrc + 3 * sx + k) * cbufx[0] * cbufy[1] +
					*(dataSrc + sy * stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
					*(dataSrc + (sy + 1)*stepSrc + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;
			}
		}
	}
	return matDst;
	//resize(matSrc,matDst2,matDst.size(),0,0,1);
	//return matDst2;
}
