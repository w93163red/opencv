#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <iostream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace cv;
using namespace std;

void calc(Mat src)
{
	vector<Mat> rgb_planes;
	split(src, rgb_planes);

	/// 设定bin数目
	int histSize = 255;

	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat r_hist, g_hist, b_hist;

	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 1280; int hist_h = 720;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// 显示直方图
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
}

void depart(Mat card,Mat src)
{
	vector<Vec2f> lines; 
	Mat rot_mat;
	Mat ans; 
	Mat temp;
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	Mat paintx = Mat::zeros(src.size(),src.type());

	cout << card.rows << " " << card.cols << endl;
	
	cout << lines.size() << endl;
	Point center = (card.cols / 2, card.rows / 2);
	HoughLines(card, lines, 1, CV_PI / 180, card.cols*0.75, 0, 0);
	double angle = lines[0][1];
	double scale = 1.05;
	rot_mat = getRotationMatrix2D(center, angle, scale);	
	warpAffine(src, ans, rot_mat, src.size());
	bitwise_not(ans, ans);
	resize(ans, ans, Size(100, 30));
	imshow("card", card);
	imshow("rot", ans);
	//calc(ans);
}

int main(int argc, char** argv)
{
	VideoCapture vid = VideoCapture("e:\\cssj\\191901_20145710536.mp4");
	if (!vid.isOpened())
		abort();
	Mat src;
	for (int i = 0; i < 2; i++)
	{
		vid >> src;
	}
	Mat mid,dst,src_color;

	src_color = src.clone();
	imshow("src", src);

	cvtColor(src, src, CV_RGB2GRAY);
	blur(src, src, Size(3, 3));
	Canny(src, mid, 100, 255, 3); //需要随图片进行调整的参数1
	
	cvtColor(mid, dst, CV_GRAY2BGR);

	vector<Vec2f> lines;
	//HoughLines(mid, lines, 1, CV_PI / 180, 150, 0, 0);
	Mat temp = mid.clone();
	Mat temp1;
	Mat temp2 = Mat::zeros(mid.size(), mid.type());
	Mat srcROI;
	Mat card;

	imshow("mid", mid);


	int dilation_size = 4;
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));    //需要随图片调整的参数2
	dilate(temp, temp1, element);

	imshow("1", temp1);
	vector<vector<Point>> contours;
	
	findContours(temp1, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(temp2, contours, -1, Scalar(255), 1);

	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Rect> ans(contours.size());
	imshow("2", temp2);
	cout << src.cols << " " << src.rows << endl;
	int n = contours.size();

	for (int i = 0; i < n; i++)
	{
		vector<Mat> mt;
		Mat src_colorROI;
		Range rowrange;
		Range colrange;
		int flag = 0;

		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		double a = abs(boundRect[i].tl().y - boundRect[i].br().y);
		double b = abs(boundRect[i].tl().x - boundRect[i].br().x);
		double area = contourArea(contours_poly[i]);
		double percent = area / ((double)src.rows * (double)src.cols);		
		src_colorROI = src_color(Rect(boundRect[i].tl(), boundRect[i].br()));
		cvtColor(src_colorROI, src_colorROI, CV_BGR2HSV);
		srand((unsigned)time(NULL));
		//cout << "-------------------" << endl;
		//cout << boundRect[i].tl() << endl;
		if (b <= a && boundRect[i].tl().y > src.rows / 2)
		{
			continue;
		}
		for (int j = 0; j < 50; j++)
		{
			int test_row = rand() % src_colorROI.rows;
			int test_col = rand() % src_colorROI.cols;
			Vec3b point = src_colorROI.at<Vec3b>(test_row, test_col);
			//cout << test_row << " " << test_col << " " << point << endl;
			if (point.val[0]>75 && point.val[0]<130 && point.val[1]>50 && point.val[2]>50)
			{
				flag++;
			}
		}
		cout << flag << endl;
		if (flag>=25)//判断条件需要修改
		{
			ans.push_back(boundRect[i]);
			cout << boundRect[i].tl() << " " << percent << endl;
			srcROI = mid(Rect(boundRect[i].tl(), boundRect[i].br()));//可能需要调整的部分3
			card = src(Rect(boundRect[i].tl(), boundRect[i].br()));
			imshow("ROI", srcROI);
			imshow("colorroi", src_colorROI);
			//card = srcROI.clone();
			imshow("card", card);
			//depart(srcROI, card);
		}
	}
	Mat temp3(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < ans.size(); i++)
	{
		rectangle(temp3, ans[i].tl(), ans[i].br(), cvScalar(255), 2, 8, 0);
	}
	imshow("3", temp3);

	waitKey(0);
	return 0;

}
