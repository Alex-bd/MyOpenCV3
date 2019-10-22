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
#include <cmath>
#define  CV_CAST_8U(t)  (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0) //���ǰ�tǿ��ת��Ϊuchar���ͣ����t>=255 ��t=255 ���t<=0 ��t=0;

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
	Mat threshold(Mat src,unsigned char threshold);		//��ֵ��
	Mat gamma(Mat src,int gamma,int c);			//٤��任
	Mat Log(Mat src, int c);					//�����任
	IplImage* myDrawHistogram(int* hist_cal);	//��ʾֱ��ͼ
	void myEqualizeHist(CvArr* srcarr, CvArr* dstarr);//ֱ��ͼ���⻯
	
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

/*٤��任*/
//�涨��gamma�ͽ�������Ҫ����100
//c:Ϊ����c��Ĭ��Ϊ255
Mat MyOpencv::gamma(Mat src,int gamma,int c)
{
	Mat matDst;
	if (gamma < 0)
	{
		cout << "gammaȡֵ����"<<endl;
		return src;
	}
	/*��ͨ��*/
	Mat lookuptable(1, 256,CV_8U);	//���ұ�
	uchar* p = lookuptable.ptr();
	for (int  i = 0; i < 256; i++)
	{
		p[i] = saturate_cast<uchar>(pow( i /255.0,gamma/100) * c);
	}
	LUT(src,lookuptable,matDst);
	////��ͨ��(ÿ��ͨ������һ�����ұ�)
	//uchar lookuptable[256 * 3];
	//for (int i = 0; i < 256; i++)
	//{
	//	if (i <= 100)
	//	{
	//		lookuptable[i * 3] = 0;
	//		lookuptable[i * 3 + 1] = 50;
	//		lookuptable[i * 3 + 2] = 50;
	//	}
	//	if (i > 100 && i <= 200)
	//	{
	//		lookuptable[i * 3] = 100;
	//		lookuptable[i * 3 + 1] = 10;
	//		lookuptable[i * 3 + 2] = 200;
	//	}
	//	if (i > 200)
	//	{
	//		lookuptable[i * 3] = 255;
	//		lookuptable[i * 3 + 1] = 200;
	//		lookuptable[i * 3 + 2] = 100;
	//	}
	//}
	//Mat lut(1,256,CV_8UC3,lookuptable);
	//LUT(src, lut, matDst);
	return matDst;
}
//�����任
Mat MyOpencv::Log(Mat src, int c)
{
	Mat  matDst;
	//�������ұ�
	Mat lookupTable(1,256,CV_8U);
	uchar *p = lookupTable.ptr();
	for (int  i = 0; i < 256; ++i)
	{
		//p[i] = saturate_cast<uchar>((c / 100.0) * log(1 + i / 255.0) * 255.0);//������ʽ
		p[i] = saturate_cast<uchar>((c / 100.0)* log(1 + i / 255.0) * 255.0);
	}
	LUT(src,lookupTable,matDst);

	return matDst;
}
//ֱ��ͼ��ʾ
IplImage* MyOpencv::myDrawHistogram(int * hist_cal)
{
	//�ҳ�ֱ��ͼ�����ֵ���Ա���й�һ��
	int hist_max = 0;
	for (int  i = 0; i < 256; i++)
	{
		if (hist_cal[i] > hist_max)
		{
			hist_max = hist_cal[i];		//�ҵ����ֵ
		}
	}
	//����һ���հ׵�ͼ��������ʾֱ��ͼ
	IplImage* hist_image = cvCreateImage(cvSize(256*3,64*3),8,1);
	//Mat hist_image(256*3,64*3,CV_8UC3,Scalar(0,0,0));
	cvZero(hist_image);
	
	for (int  j = 0; j < 256; j++)
	{
		CvPoint p0 = cvPoint(j * 3, 0);
		CvPoint p1 = cvPoint((j + 1) * 3, cvRound(((hist_max - hist_cal[j]) * 64) / hist_max * 3));//�Խ��ߵĶ���
		/* 
		 ������������color 
		������ɫ (RGB) �����ȣ��Ҷ�ͼ�� ��(grayscale image���� 
		���ĸ�������thickness 
		��ɾ��ε������Ĵ�ϸ�̶ȡ�ȡ��ֵʱ���� CV_FILLED���������������ɫ�ʵľ��Ρ� 
		�����������line_type 
		���������͡���cvLine������ 
		������������shift 
		������С����λ���� */
		cvRectangle(hist_image,p0,p1,cvScalar(255,255,255),-1,8,0);
		//����Mat����ͼ�ϻ��ƾ���ʱ��ѡ��cv::trctangle()
		//rectangle(hist_image,p0,p1,cvScalar(255,255,255),-1,8,0);
	}
	return hist_image;
}


int p[256];			//Դ���ұ�����
int dst_p[256];		//Ŀ����ұ�����
//ֱ��ͼ���⻯
void myEqualizeHist(CvArr* srcarr, CvArr* dstarr)
{
	CvMat sstub;
	CvMat dstub;
	CvMat* src = cvGetMat(srcarr, &sstub);//convert CvArr to CvMat
	CvMat* dst = cvGetMat(dstarr, &dstub);
					
	CvSize size = cvSize(src->rows,src->cols);	//���src�ߴ�

	int x, y;
	const int hist_size = 256;	//ֱ��ͼ�Ҷȷ�Χ
	//int p[hist_size];//p���鳤��Ϊͼ��ĻҶȵȼ���һ��Ϊ256��
	fill(p, p + hist_size, 0);	//��ʼ��p[256]����ȫ��Ϊ0

	//ɨ��ͼ���ÿһ�����ص㣬����ֵΪk��hist[k]++
	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step * y;		//һ��һ�е�ָ��
		for (x = 0; x < size.width; x++)
		{
			p[sptr[x]]++;	//�൱��[x][y]������Ӧ������ֵ++   sptr[x]���Ǹ�ֵ����ֵ����ĳ��ĻҶ�ֵ
		}
	}

	int c[hist_size];
	c[0] = p[0];
	//�ۼƺ��������ͼ����Ҷȼ����ۼƷֲ�
	for (int i = 1; i < hist_size; i++)
	{
		c[i] = c[i - 1] + p[i];
	}

	uchar lut[hist_size];   //���� ���ұ�
	//����ӳ�亯��������look up table
	for (int i = 0; i < hist_size; i++)
	{
		int val = cvRound(c[i] * (255.f / (src->cols * src->rows)));	//�ۼƺ���*255/N
		lut[i] =CV_CAST_8U(val);//����ֵiӳ��֮��ֵΪlut[i]	  //��int���תΪuchar ,������궨��
	}
	
	//����look up table���ı�ͼ������ֵ���Ѽ�������⻯���ͼ���������ֵ
	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step * y;
		uchar* dptr = dst->data.ptr + dst->step * y;
		for (x = 0; x < size.width; x++)
		{
			dptr[x] = lut[sptr[x]];			
		}
	}

	//����ֱ��ͼ���⻯֮���ͼ������طֲ�
	//int dst_p[hist_size];
	fill(dst_p, dst_p + hist_size, 0);
	for (y = 0; y < size.height; y++)
	{
		const uchar* dst_sptr = dst->data.ptr + dst->step * y;
		for (x = 0; x < size.width; x++)
		{
			dst_p[dst_sptr[x]]++;//�൱��[x][y]������Ӧ������ֵ++
		}
	}
}