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
	Mat M_resize_zjl(Mat src,Mat dst,int x,int y);	//����� ���ô�С

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
Mat MyOpencv::M_resize_zjl(Mat src,Mat dst,int x,int y)
{
	//������⼸�ж���һ�µ�
	Mat matSrc, matDst1, matDst2;			//����ͼ�� Դ������Ŀ�����
	matSrc = src;							//��ȡԴ
	//Mat  (��С�����ͣ���ԭɫ��ֵ)
	matDst1 = Mat(Size(x,y),matSrc.type(),Scalar::all(0));	//��Ŀ�����ؾ���Ԫ��ȫ����Ϊ0
	//��ʽ��srcX=dstX* (srcWidth/dstWidth) , srcY = dstY * (srcHeight/dstHeight)
	double scale_x = (double) matSrc.cols / matDst1.cols;
	double scale_y = (double) matSrc.rows / matDst1.cols;
	//����ڹؼ�����
	//����˫��ѭ������������������ʽ��Ŀ�����ؾ����ÿ�����ص���и�ֵ
	for (int i=0; i < matDst1.cols; ++i)	//���������ص�
	{
		int sx = cvFloor(i*scale_x);	//cvFloor:���ز����ڲ������������ֵ
		sx = min(sx, matSrc.cols - 1);
		for (int  j = 0; j < matDst1.rows; ++j)
		{
			int sy = cvFloor(j*scale_y);
			sy = min(sy, matSrc.rows-1);
			matDst1.at<Vec3b>(j, i) = matSrc.at<Vec3b>(sy,sx);//��ֵ
		}
	}
	//imwrite("nearest_1.jpg",matDst1);

	return matDst1;
}