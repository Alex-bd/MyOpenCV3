/*2019-09-03 Alex
��Ŀ˵��������Ŀ��Ҫ��ͨ����ȡMatͼ�񣬶�OpenCV�ĺ���������д��
֮����������󣬰Ѻ�������ͼ���ʦ�ĺ����С�
*/
#pragma once
//��дOpenCV��ͷ�ļ�
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
	//����������
	Mat M_resize_zjl(Mat src,int x,int y);	//����� ���ô�С
	Mat M_resize_sxx(Mat src,int x,int y);	//˫����
	Mat xuanzhuan(Mat src, int angle);		//��ת ����任
	Mat flip(Mat src,int flag);
private:

};

MyOpencv::MyOpencv()
{
}

MyOpencv::~MyOpencv()
{
}
//����� ���ô�С
/*����1��Դͼ��
  ����2��Ŀ��ͼ��
  ����3��weight
  ����4��height
*/
Mat MyOpencv::M_resize_zjl(Mat src,int x,int y)
{
	//������⼸�ж���һ�µ�
	Mat matSrc, matDst;			//����ͼ�� Դ������Ŀ�����
	Mat matDst2;
	matSrc = src;							//��ȡԴ
	//Mat  (��С�����ͣ���ԭɫ��ֵ)
	matDst = Mat(Size(x,y),matSrc.type(),Scalar::all(0));	//��Ŀ�����ؾ���Ԫ��ȫ����Ϊ0
	matDst2 = Mat(matDst.size(),matSrc.type(),Scalar::all(0));
															//��ʽ��srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double) matSrc.cols / matDst.cols;
	double scale_y = (double) matSrc.rows / matDst.cols;
	//����ڹؼ�����
	//����˫��ѭ������������������ʽ��Ŀ�����ؾ����ÿ�����ص���и�ֵ
	for (int i=0; i < matDst.cols; ++i)	//���������ص�
	{
		int sx = cvFloor(i*scale_x);	//cvFloor:���ز����ڲ������������ֵ
		sx = min(sx, matSrc.cols - 1);
		for (int  j = 0; j < matDst.rows; ++j)
		{
			int sy = cvFloor(j*scale_y);
			sy = min(sy, matSrc.rows-1);
			matDst.at<Vec3b>(j, i) = matSrc.at<Vec3b>(sy,sx);//��ֵ
		}
	}
	//return matDst;
	resize(matSrc,matDst2,matDst.size(),0,0,1);
	return matDst2;
}
//˫���Բ�ֵ
Mat MyOpencv::M_resize_sxx(Mat src,int x,int y)
{
	Mat matSrc, matDst;			//����ͼ�� Դ������Ŀ�����
	//Mat matDst2;	//����resize���ڶԱȵ�Ŀ��ͼ��
	matSrc = src;							//��ȡԴ
	//Mat  (��С�����ͣ���ԭɫ��ֵ)
	matDst = Mat(Size(x, y), matSrc.type(), Scalar::all(0));	//��Ŀ�����ؾ���Ԫ��ȫ����Ϊ0
	//matDst2 = Mat(matDst.size(), matSrc.type(), Scalar::all(0));
	//��ʽ��srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double)matSrc.cols / matDst.cols;
	double scale_y = (double)matSrc.rows / matDst.cols;

	uchar * dataDst = matDst.data;		//����һ��ָ��ָ��Ŀ��ͼ���������
	int stepDst = matDst.step;
	uchar * dataSrc = matSrc.data;		//ָ��ָ��Դͼ���������
	int stepSrc = matSrc.step;
	int iWidthSrc = matSrc.cols;	//Դͼ������
	int iHeightSrc = matSrc.rows;	//Դͼ������

	for(int j = 0;j < matDst.rows; j++)
	{
		float fy = (float)((j + 0.5)*scale_y - 0.5);	//������
		int sy = cvFloor(fy);		//��y ����
		fy -= sy;					//ֻ����С��
		sy = min(sy,iHeightSrc -2);
		sy = max(0,sy);				//�������
		
		short cbufy[2];				//����һ������	
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

//����� ����任
Mat MyOpencv::xuanzhuan(Mat src, int angle)
{
	Mat matSrc, matDst;
	matSrc = src;
	double degree = angle;	//��ת�Ƕ�
	double angles = degree* CV_PI / 180;
	double alpha = cos(angles);
	double beta = sin(angles);
	int iwidth = matSrc.cols;
	int iheight = matSrc.rows;
	int iNewWidth = cvRound(iwidth * fabs(alpha)+ iheight *fabs(beta));
	int iNewHeight = cvRound(iheight * fabs(alpha) + iwidth * fabs(beta));

	double m[6];			//�任����  
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1 - alpha) * iwidth / 2 - beta * iheight / 2;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = beta * iwidth / 2 + (1 - alpha)* iheight / 2;

	Mat M = Mat(2,3,CV_64F,m);
	matDst = Mat(Size(iNewWidth,iNewHeight),matSrc.type(),Scalar::all(0));		//ȫ��ֵΪ0��ͼ��

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

//����
Mat MyOpencv::flip(Mat src,int flag)
{
	Mat matSrc, matDst;
	matSrc = src;
	int width = matSrc.cols;
	int height = matSrc.rows;
	matDst = Mat(Size(width,height), matSrc.type(), Scalar::all(0));
	if (flag == 0)//ˮƽ����
	{
		for (int y = 0; y < matSrc.cols; y++)
		{
			for (int x = 0; x < matSrc.rows; x++)
			{
				matDst.at<cv::Vec3b>(y,x) = matSrc.at<cv::Vec3b>(y,width-x-1);
			}
		}
	}
	else if (flag > 0)	//��ֱ����
	{

	}
	else
	{

	}
	return matDst;
}