/*2019-09-03 Alex
��Ŀ˵��������Ŀ��Ҫ��ͨ����ȡMatͼ�񣬶�OpenCV�ĺ���������д��
֮����������󣬰Ѻ�������ͼ���ʦ�ĺ����С�
*/
#pragma once
//��дOpenCV��ͷ�ļ�
#include <iostream>
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
	Mat fangshe(Mat src, int angle);		// ����任
	Mat flip(Mat src,int flag);				//����
	Mat rotateImage(Mat src, int degree, int border_value);	//��תͼ��
	Mat threshold(Mat src,unsigned char threshold);
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
//angleΪ��ת�Ƕ�
Mat MyOpencv::fangshe(Mat src, int angle)
{
	Mat matSrc, matDst;
	matSrc = src;
	double degree = -angle;	//��ת�Ƕ�
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
/* flag = 0 ��ʾˮƽ����
   flag > 0 ��ʾ��ֱ����
   flag < 0 ��ʾˮƽ��ֱ����
*/
Mat MyOpencv::flip(Mat src,int flag)
{
	Mat matSrc, matDst,matDst2;
	matSrc = src;
	int width = matSrc.cols;
	int height = matSrc.rows;
	matDst = Mat(Size(width,height), matSrc.type(), Scalar::all(0));
	
	if (flag == 0)//ˮƽ����
	{
		for (int i = 0; i < matSrc.rows; i++)
		{
			for (int j = 0; j < matSrc.cols; j++)
			{
				matDst.at<Vec3b>(i, j)[0] = matSrc.at<Vec3b>(i, src.cols - 1 - j)[0];
				matDst.at<Vec3b>(i, j)[1] = matSrc.at<Vec3b>(i, src.cols - 1 - j)[1];
				matDst.at<Vec3b>(i, j)[2] = matSrc.at<Vec3b>(i, src.cols - 1 - j)[2];
			}
		}
	}
	else if (flag > 0)	//��ֱ����
	{
		for (int i = 0; i < matSrc.rows; i++)
		{
			for (int j = 0; j < matSrc.cols; j++)
			{
				matDst.at<Vec3b>(i, j)[0] = matSrc.at<Vec3b>(src.rows -1 -i, j)[0];
				matDst.at<Vec3b>(i, j)[1] = matSrc.at<Vec3b>(src.rows - 1 - i, j)[1];
				matDst.at<Vec3b>(i, j)[2] = matSrc.at<Vec3b>(src.rows - 1 - i, j)[2];
			}
		}
	}
	else
	{
		for (int i = 0; i < matSrc.rows; i++)
		{
			for (int j = 0; j < matSrc.cols; j++)
			{
				matDst.at<Vec3b>(i, j)[0] = matSrc.at<Vec3b>(src.rows - 1 - i, src.cols - 1 - j)[0];
				matDst.at<Vec3b>(i, j)[1] = matSrc.at<Vec3b>(src.rows - 1 - i, src.cols - 1 - j)[1];
				matDst.at<Vec3b>(i, j)[2] = matSrc.at<Vec3b>(src.rows - 1 - i, src.cols - 1 - j)[2];
			}
		}
		
	}
	return matDst;
}


//ͼ����ת����ת����ȡͼ��Crop ����ȡͼ�������ڽӾ���
//         Mat img ��ͼ�����룬��ͨ��������ͨ��
//         Mat & imgout ��ͼ�����
//         int degree ��ͼ��Ҫ��ת�ĽǶ�
//         int border_value��ͼ����ת���ֵ(0-255)
Mat MyOpencv::rotateImage(Mat src, int degree, int border_value)
{
	Mat matDst;
	degree = -degree;//warpAffineĬ�ϵ���ת��������ʱ�룬���ԼӸ��ű�ʾת��Ϊ˳ʱ��
	double angle = degree * CV_PI / 180.; // ����  
	double a = sin(angle), b = cos(angle);
	int width = src.cols;
	int height = src.rows;
	int width_rotate = int(width * fabs(b) - height * fabs(a));//height * fabs(a) + 
	int height_rotate = int(height * fabs(b) - width * fabs(a));//width * fabs(a) + 
	if (width_rotate <= 20 || height_rotate <= 20)
	{
		width_rotate = 20;
		height_rotate = 20;
	}
	//��ת����map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6];
	Mat map_matrix = Mat(2, 3, CV_32F, map);	//����ת����ת��Mat��2 x 3 ����
	// ��ת����
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 =  map_matrix;
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);//�����ά��ת�ķ���任����   ����������Ϊ�����������
	map[2] += (width_rotate - width) / 2;		//b1
	map[5] += (height_rotate - height) / 2;		//b2
	//Mat img_rotate;
	//��ͼ��������任
	//CV_WARP_FILL_OUTLIERS - ����������ͼ������ء�
	//�������������������ͼ��ı߽��⣬��ô���ǵ�ֵ�趨Ϊ fillval.
	//CV_WARP_INVERSE_MAP - ָ�� map_matrix �����ͼ������ͼ��ķ��任��
	int chnnel = src.channels();
	if (chnnel == 3)
		warpAffine(src, matDst, map_matrix, Size(width_rotate, height_rotate), 1, 0, Scalar(border_value, border_value, border_value));//BGR
	else
		warpAffine(src, matDst, map_matrix, Size(width_rotate, height_rotate), 1, 0, border_value);
	return matDst;
}

Mat MyOpencv::threshold(Mat src, unsigned char threshold)
{
	//��ȡһ��Ӱ��
	Mat matSrc = src;
	/*Mat matDst;*/
	//���Ӱ��Ϊ�յĻ���ֱ�ӷ���
	if (matSrc.empty())
	{
		cout << "����ͼ��Ϊ��" << endl;
	}
	//��ȡӰ����к���
	int iWidth = matSrc.cols;
	int iHeight = matSrc.rows;
	//�������Ӱ��,��Ϊ�Ƕ�ֵ��Ӱ�����Դ������ǻҶ�Ӱ��
	Mat Dst = Mat(iHeight, iWidth, CV_8UC1);
	//����ɫӰ��ת��Ϊ�Ҷ�Ӱ��
	//ת����ʽΪ
	//Y = 0.2126 R + 0.7152 G + 0.0722 B.
	unsigned char Threshold = threshold;
	for (int i = 0; i < iHeight; i++)
	{
		for (int j = 0; j < iWidth; j++)
		{
			//��Ҫע���һ���ط���OpenCV��ȡ��ɫӰ�����BGR��˳��
			//�Ƚ���ɫӰ��ת��Ϊ�Ҷ�Ӱ��
			unsigned char B = matSrc.at<Vec3b>(i, j)[0];
			unsigned char G = matSrc.at<Vec3b>(i, j)[1];
			unsigned char R = matSrc.at<Vec3b>(i, j)[2];
			unsigned char iGray = 0.2126 * R + 0.7152 * G + 0.0722 * B;
			//���лҶȶ�ֵ������
			if (iGray > Threshold)
			{
				Dst.at<unsigned char>(i, j) = 255;
			}
			else
			{
				Dst.at<unsigned char>(i, j) = 0;
			}
		}
	}
	return Dst;
}