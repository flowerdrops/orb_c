#include "stdafx.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>  
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
	Mat frame, img;
	std::vector<KeyPoint> key, key_frame;
	Mat des, des_frame;
	Ptr<ORB> orb = ORB::create(5, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	Ptr<ORB> orb_frame = ORB::create(200, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
	//////////////////// 对识别标准的初始化  //////////////////
	img = imread("test1.jpg", CV_LOAD_IMAGE_COLOR);		
	orb->detect(img, key);
	orb->compute(img, key, des);

	///////////////////  这段好像不是必须的？  //////////////
	Mat outimg;
	drawKeypoints(img, key, outimg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB 特征点", outimg);

	/////////////////////////    视频的初始化  ///////////////////////////////////
	VideoCapture cap(0);

	/////////////////////////    初始化匹配参数  ///////////////////////
	std::vector<DMatch> matches;
	vector<DMatch> good_matches;
	BFMatcher matcher(NORM_HAMMING);

	while (true) {
		cap >> frame;
		if (frame.empty())
		{
			cerr << "ERROR: Can't grab camera frame." << endl;
			break;
		}
		/////// 对frame的识别  ///////
		orb_frame->detect(frame, key_frame);
		orb_frame->compute(frame, key_frame, des_frame);

		if (des_frame.empty())
			continue;
		///////// 好了，可以开始匹配了!  ////////
		matches.clear();	good_matches.clear();
		matcher.match(des, des_frame, matches);
		double min_dist = 10000, max_dist = 0;
		for (int i = 0; i < des.rows; i++) {
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		for (int i = 0; i < des.rows; i++) {
			if (matches[i].distance <= max(2 * min_dist, 30.0)) {
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_goodmatch;
//		key_frame.clear();
		drawMatches(img, key, frame, key_frame, good_matches, img_goodmatch);
		imshow("优化后的匹配点对", img_goodmatch);
		int key = waitKey(1);
		if (key == 27/*ESC*/)
			break;
	}

	return 0;
}