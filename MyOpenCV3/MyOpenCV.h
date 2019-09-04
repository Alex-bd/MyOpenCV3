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
