#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#define REOURCEFOLDER "C:\\Github\\ShangChuang\\OpenCVTest\\Debug\\"

using namespace cv;
using namespace std;

int main()
{
	VideoCapture capture(REOURCEFOLDER"If.mp4");
	double totalFrame = capture.get(CAP_PROP_FRAME_COUNT);
	printf("整个视频共%d帧", totalFrame);

	double frameToStart = 0;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	printf("从%d帧开始读", frameToStart);

	//设置结束帧
	double frameToStop = totalFrame;

	if (frameToStop < frameToStart)
	{
		cout << "结束帧小于开始帧，程序错误，即将退出！" << endl;
		return -1;
	}
	else
	{
		cout << "结束帧为：第" << frameToStop << "帧" << endl;
	}

	//获取帧率
	double rate = capture.get(CAP_PROP_FPS);
	cout << "帧率为:" << rate << endl;

	//定义一个用来控制读取视频循环结束的变量
	//	bool stop = false;

	//承载每一帧的图像
	Mat frame;
	//显示每一帧的窗口
	namedWindow("Extracted frame");
	//两帧间的间隔时间:
	int delay = 1000 / rate;

	//利用while循环读取帧
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
	double currentFrame = frameToStart;
	
	//滤波器的核
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size,kernel_size,CV_32F)/(float)(kernel_size*kernel_size);
	
	Mat output;
	while (currentFrame <= totalFrame)
	{
		//读取下一帧
		if (!capture.read(frame))
		{
			cout << "读取视频失败" << endl;
			return -1;
		}
	
		imshow("1.Origin", frame);
		Mat origin(frame);
			
		cout << "正在读取第" << currentFrame << "帧" << endl;

		cvtColor(frame, frame, COLOR_RGB2GRAY);
		imshow("2.RGB2GRAY", frame);
		Mat gray;
		frame.copyTo(gray);
		threshold(frame, frame, 250, 255, THRESH_BINARY);
		imshow("3.threshold",frame);

		Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
		cv::morphologyEx(frame, frame, MORPH_CLOSE, element);
		imshow("4.morph", frame);

		//variables
		vector<vector<Point> > contours;
		findContours(frame, contours, 0, 1);
		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (uint i = 0; i < contours.size(); i++)
			if (contours[i].size()>240)
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
				Rect appRect(boundingRect(Mat(contours_poly[i])));
				if (appRect.width>appRect.height)
					boundRect.push_back(appRect);
			}
		Mat region;
		origin.copyTo(region);

		Mat msk = Mat::zeros(frame.size(), CV_8U);
		for (uint i = 0; i < boundRect.size(); i++)
		{
			rectangle(region, boundRect[i], Scalar(255, 255, 255), 2);
			msk(boundRect[i]) = 1;
		}
		imshow("5.area", region);
		//imshow("msk", msk);

		//Mat tmp;
		//inpaint(origin, msk, tmp, 3, INPAINT_TELEA);

		Mat processed;
		threshold(gray, gray, 230, 255, THRESH_BINARY);
		gray.copyTo(processed, msk);


		//have to dilate the mask region
		element = getStructuringElement(MORPH_RECT,
			Size(2 * 5, 2 * 5 + 1),
			Point(5, 5));
		dilate(processed, processed, element);

		//element = getStructuringElement(MORPH_RECT, Size(9, 9));
		//cv::morphologyEx(processed, processed, MORPH_CLOSE, element);

		imshow("6.Processed mask",processed);
		
		Mat tmp;
		inpaint(origin, processed, tmp, 1, INPAINT_NS);
		imshow("7.Final Result", tmp);
		



		waitKey(delay);
		//按下ESC或者到达指定的结束帧后退出读取视频
		/*		if((char) c == 27 || currentFrame > frameToStop)
		{
		stop = true;
		}
		//按下按键后会停留在当前帧，等待下一次按键
		if( c >= 0)
		{
		waitKey(0);
		}
		waitKey(0);
		*/		currentFrame++;
	}
	//关闭视频文件
	capture.release();
	waitKey(0);
	return 0;
}