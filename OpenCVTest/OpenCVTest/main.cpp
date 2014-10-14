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
	printf("������Ƶ��%d֡", totalFrame);

	double frameToStart = 0;
	capture.set(CAP_PROP_POS_FRAMES, frameToStart);
	printf("��%d֡��ʼ��", frameToStart);

	//���ý���֡
	double frameToStop = totalFrame;

	if (frameToStop < frameToStart)
	{
		cout << "����֡С�ڿ�ʼ֡��������󣬼����˳���" << endl;
		return -1;
	}
	else
	{
		cout << "����֡Ϊ����" << frameToStop << "֡" << endl;
	}

	//��ȡ֡��
	double rate = capture.get(CAP_PROP_FPS);
	cout << "֡��Ϊ:" << rate << endl;

	//����һ���������ƶ�ȡ��Ƶѭ�������ı���
	//	bool stop = false;

	//����ÿһ֡��ͼ��
	Mat frame;
	//��ʾÿһ֡�Ĵ���
	namedWindow("Extracted frame");
	//��֡��ļ��ʱ��:
	int delay = 1000 / rate;

	//����whileѭ����ȡ֡
	//currentFrame����ѭ�����п��ƶ�ȡ��ָ����֡��ѭ�������ı���
	double currentFrame = frameToStart;
	
	//�˲����ĺ�
	int kernel_size = 3;
	Mat kernel = Mat::ones(kernel_size,kernel_size,CV_32F)/(float)(kernel_size*kernel_size);
	
	Mat output;
	while (currentFrame <= totalFrame)
	{
		//��ȡ��һ֡
		if (!capture.read(frame))
		{
			cout << "��ȡ��Ƶʧ��" << endl;
			return -1;
		}
	
		imshow("1.Origin", frame);
		Mat origin(frame);
			
		cout << "���ڶ�ȡ��" << currentFrame << "֡" << endl;

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
		//����ESC���ߵ���ָ���Ľ���֡���˳���ȡ��Ƶ
		/*		if((char) c == 27 || currentFrame > frameToStop)
		{
		stop = true;
		}
		//���°������ͣ���ڵ�ǰ֡���ȴ���һ�ΰ���
		if( c >= 0)
		{
		waitKey(0);
		}
		waitKey(0);
		*/		currentFrame++;
	}
	//�ر���Ƶ�ļ�
	capture.release();
	waitKey(0);
	return 0;
}