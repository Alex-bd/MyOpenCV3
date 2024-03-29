/*2019-09-03 Alex
项目说明：该项目主要是通过读取Mat图像，对OpenCV的函数进行重写，
之后调试正常后，把函数用于图像大师的函数中。
*/
#pragma once
//重写OpenCV的头文件
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>
#define  CV_CAST_8U(t)  (uchar)(!((t) & ~255) ? (t) : (t) > 0 ? 255 : 0) //就是把t强制转化为uchar类型，如果t>=255 则t=255 如果t<=0 则t=0;

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
	Mat fangshe(Mat src, int angle);		// 仿射变换
	Mat flip(Mat src,int flag);				//镜像
	Mat rotateImage(Mat src, int degree, int border_value);	//旋转图像
	Mat threshold(Mat src,unsigned char threshold);		//二值化
	Mat gamma(Mat src,int gamma,int c);			//伽马变换
	Mat Log(Mat src, int c);					//对数变换
	IplImage* myDrawHistogram(int* hist_cal);	//显示直方图
	void myEqualizeHist(CvArr* srcarr, CvArr* dstarr);//直方图均衡化
	
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
//angle为旋转角度
Mat MyOpencv::fangshe(Mat src, int angle)
{
	Mat matSrc, matDst;
	matSrc = src;
	double degree = -angle;	//旋转角度
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
/* flag = 0 表示水平镜像
   flag > 0 表示垂直镜像
   flag < 0 表示水平垂直镜像
*/
Mat MyOpencv::flip(Mat src,int flag)
{
	Mat matSrc, matDst,matDst2;
	matSrc = src;
	int width = matSrc.cols;
	int height = matSrc.rows;
	matDst = Mat(Size(width,height), matSrc.type(), Scalar::all(0));
	
	if (flag == 0)//水平镜像
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
	else if (flag > 0)	//垂直镜像
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


//图像旋转：旋转（截取图像）Crop ，截取图像最大的内接矩形
//         Mat img ：图像输入，单通道或者三通道
//         Mat & imgout ：图像输出
//         int degree ：图像要旋转的角度
//         int border_value：图像旋转填充值(0-255)
Mat MyOpencv::rotateImage(Mat src, int degree, int border_value)
{
	Mat matDst;
	degree = -degree;//warpAffine默认的旋转方向是逆时针，所以加负号表示转化为顺时针
	double angle = degree * CV_PI / 180.; // 弧度  
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
	//旋转数组map
	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]
	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]
	float map[6];
	Mat map_matrix = Mat(2, 3, CV_32F, map);	//把旋转矩阵转成Mat的2 x 3 矩阵
	// 旋转中心
	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);
	CvMat map_matrix2 =  map_matrix;
	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);//计算二维旋转的仿射变换矩阵   第三个参数为等向比例因子
	map[2] += (width_rotate - width) / 2;		//b1
	map[5] += (height_rotate - height) / 2;		//b2
	//Mat img_rotate;
	//对图像做仿射变换
	//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。
	//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.
	//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，
	int chnnel = src.channels();
	if (chnnel == 3)
		warpAffine(src, matDst, map_matrix, Size(width_rotate, height_rotate), 1, 0, Scalar(border_value, border_value, border_value));//BGR
	else
		warpAffine(src, matDst, map_matrix, Size(width_rotate, height_rotate), 1, 0, border_value);
	return matDst;
}

Mat MyOpencv::threshold(Mat src, unsigned char threshold)
{
	//读取一幅影像
	Mat matSrc = src;
	/*Mat matDst;*/
	//如果影像为空的话，直接返回
	if (matSrc.empty())
	{
		cout << "输入图像为空" << endl;
	}
	//获取影像的行和列
	int iWidth = matSrc.cols;
	int iHeight = matSrc.rows;
	//创建结果影像,因为是二值化影像，所以创建的是灰度影像
	Mat Dst = Mat(iHeight, iWidth, CV_8UC1);
	//将彩色影像转换为灰度影像
	//转换公式为
	//Y = 0.2126 R + 0.7152 G + 0.0722 B.
	unsigned char Threshold = threshold;
	for (int i = 0; i < iHeight; i++)
	{
		for (int j = 0; j < iWidth; j++)
		{
			//需要注意的一个地方是OpenCV读取彩色影像的是BGR的顺序
			//先将彩色影像转换为灰度影像
			unsigned char B = matSrc.at<Vec3b>(i, j)[0];
			unsigned char G = matSrc.at<Vec3b>(i, j)[1];
			unsigned char R = matSrc.at<Vec3b>(i, j)[2];
			unsigned char iGray = 0.2126 * R + 0.7152 * G + 0.0722 * B;
			//进行灰度二值化处理
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

/*伽马变换*/
//规定：gamma送进来的数要除以100
//c:为常数c，默认为255
Mat MyOpencv::gamma(Mat src,int gamma,int c)
{
	Mat matDst;
	if (gamma < 0)
	{
		cout << "gamma取值错误"<<endl;
		return src;
	}
	/*单通道*/
	Mat lookuptable(1, 256,CV_8U);	//查找表
	uchar* p = lookuptable.ptr();
	for (int  i = 0; i < 256; i++)
	{
		p[i] = saturate_cast<uchar>(pow( i /255.0,gamma/100) * c);
	}
	LUT(src,lookuptable,matDst);
	////三通道(每个通道单独一个查找表)
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
//对数变换
Mat MyOpencv::Log(Mat src, int c)
{
	Mat  matDst;
	//建立查找表
	Mat lookupTable(1,256,CV_8U);
	uchar *p = lookupTable.ptr();
	for (int  i = 0; i < 256; ++i)
	{
		//p[i] = saturate_cast<uchar>((c / 100.0) * log(1 + i / 255.0) * 255.0);//对数公式
		p[i] = saturate_cast<uchar>((c / 100.0)* log(1 + i / 255.0) * 255.0);
	}
	LUT(src,lookupTable,matDst);

	return matDst;
}
//直方图显示
IplImage* MyOpencv::myDrawHistogram(int * hist_cal)
{
	//找出直方图中最大值，以便进行归一化
	int hist_max = 0;
	for (int  i = 0; i < 256; i++)
	{
		if (hist_cal[i] > hist_max)
		{
			hist_max = hist_cal[i];		//找到最大值
		}
	}
	//创建一幅空白的图像，用来显示直方图
	IplImage* hist_image = cvCreateImage(cvSize(256*3,64*3),8,1);
	//Mat hist_image(256*3,64*3,CV_8UC3,Scalar(0,0,0));
	cvZero(hist_image);
	
	for (int  j = 0; j < 256; j++)
	{
		CvPoint p0 = cvPoint(j * 3, 0);
		CvPoint p1 = cvPoint((j + 1) * 3, cvRound(((hist_max - hist_cal[j]) * 64) / hist_max * 3));//对角线的顶点
		/* 
		 第三个参数：color 
		线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）。 
		第四个参数：thickness 
		组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩的矩形。 
		第五个参数：line_type 
		线条的类型。见cvLine的描述 
		第六个参数：shift 
		坐标点的小数点位数。 */
		cvRectangle(hist_image,p0,p1,cvScalar(255,255,255),-1,8,0);
		//当在Mat类型图上绘制矩形时，选择cv::trctangle()
		//rectangle(hist_image,p0,p1,cvScalar(255,255,255),-1,8,0);
	}
	return hist_image;
}


int p[256];			//源查找表数组
int dst_p[256];		//目标查找表数组
//直方图均衡化
void myEqualizeHist(CvArr* srcarr, CvArr* dstarr)
{
	CvMat sstub;
	CvMat dstub;
	CvMat* src = cvGetMat(srcarr, &sstub);//convert CvArr to CvMat
	CvMat* dst = cvGetMat(dstarr, &dstub);
					
	CvSize size = cvSize(src->rows,src->cols);	//求出src尺寸

	int x, y;
	const int hist_size = 256;	//直方图灰度范围
	//int p[hist_size];//p数组长度为图像的灰度等级（一般为256）
	fill(p, p + hist_size, 0);	//初始化p[256]数组全部为0

	//扫描图像的每一个像素点，像素值为k则hist[k]++
	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step * y;		//一行一行的指向
		for (x = 0; x < size.width; x++)
		{
			p[sptr[x]]++;	//相当于[x][y]这个点对应的像素值++   sptr[x]就是个值，该值代表某点的灰度值
		}
	}

	int c[hist_size];
	c[0] = p[0];
	//累计函数，求出图像各灰度级的累计分布
	for (int i = 1; i < hist_size; i++)
	{
		c[i] = c[i - 1] + p[i];
	}

	uchar lut[hist_size];   //定义 查找表
	//根据映射函数，建立look up table
	for (int i = 0; i < hist_size; i++)
	{
		int val = cvRound(c[i] * (255.f / (src->cols * src->rows)));	//累计函数*255/N
		lut[i] =CV_CAST_8U(val);//像素值i映射之后值为lut[i]	  //把int结果转为uchar ,规则见宏定义
	}
	
	//根据look up table，改变图像像素值，已计算出均衡化后的图像各点像素值
	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step * y;
		uchar* dptr = dst->data.ptr + dst->step * y;
		for (x = 0; x < size.width; x++)
		{
			dptr[x] = lut[sptr[x]];			
		}
	}

	//计算直方图均衡化之后的图像的像素分布
	//int dst_p[hist_size];
	fill(dst_p, dst_p + hist_size, 0);
	for (y = 0; y < size.height; y++)
	{
		const uchar* dst_sptr = dst->data.ptr + dst->step * y;
		for (x = 0; x < size.width; x++)
		{
			dst_p[dst_sptr[x]]++;//相当于[x][y]这个点对应的像素值++
		}
	}
}