#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv/highgui.h>
#include <sstream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2\video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
using namespace cv;
using namespace std;

const int gauss_slider_max = 100;
const int mean_slider_max = 100;
int GaussSize=4;


int sp = 8;
int sr = 8;
int maxLevel = 6;

int meanBlock = 4;
int C = 5;

int meanBlock1 = 7;
int C1 = 2;

Mat img,img1;
Mat gray,gray1, output, output1, output2;
Mat blurred;
Mat dil, ero, dil1, ero1;
Mat hist,hist1;

int dilation_size = 1;
int erosion_size = 1;

int dilation_size1 = 1;
int erosion_size1 = 1;


/*void trackDilationandErosion(int, void*)
{
	dilate(output, dil, getStructuringElement(
		MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(-1,-1)));

	erode(dil, ero, getStructuringElement(
		MORPH_RECT,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(-1, -1)));
	imshow("Morph", ero);
	imwrite("meanmorph.png", ero);

}*/

void trackDilationandErosion1(int, void*)
{
	dilate(output1, dil1, getStructuringElement(
		MORPH_RECT,
		Size(2 * dilation_size1 + 1, 2 * dilation_size1 + 1),
		Point(-1, -1)));

	erode(dil1, ero1, getStructuringElement(
		MORPH_RECT,
		Size(2 * erosion_size1 + 1, 2 * erosion_size1 + 1),
		Point(-1, -1)));
	imshow("Morph1", ero1);
	imwrite("gaussmorph.jpg", ero1);

}


void trackAdapGauss(int, void*)
{
	cvtColor(blurred, gray1, CV_BGR2GRAY);
	//equalizeHist(hist1, gray1);
	//imshow("Equalized1", gray1);
	adaptiveThreshold(gray1, output1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, (meanBlock1*2)+1, C1);
	imshow("AdaptiveGaussian", output1);
	imwrite("gaussthresh.jpg", output1);
	trackDilationandErosion1(0, 0);
}


/*void trackAdapMean(int, void*)
{
	cvtColor(blurred, gray, CV_BGR2GRAY);
	//equalizeHist(hist, gray);
	//imshow("Equalized", gray);
	adaptiveThreshold(gray, output, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, (meanBlock*2)+1, C);
	imshow("AdaptiveMean", output);
	imwrite("meanthresh.jpg", output);
	trackDilationandErosion(0, 0);
}*/

void trackMeanShift(int, void*)
{
	pyrMeanShiftFiltering(img1, blurred, sp, sr, maxLevel);
	imshow("MeanShift", blurred);
	//trackAdapMean(0, 0);
	imwrite("meanshiftfilter.jpg", blurred);
	trackAdapGauss(0, 0);
}

void trackGaussian(int, void*)
{
	GaussianBlur(img, img1, Size((GaussSize*2)+1, (GaussSize*2)+1), 0, 0);
	imshow("Gaussian",img1);
	imwrite("gaussblur.jpg", img1);
	trackMeanShift(0, 0);
}






int main()
{
	img = imread("11.jpg");
	//Mat gray,output,output1,output2;
	//Mat blur;
	//cvtColor(img, gray, CV_BGR2GRAY);
	//for (int i = 1; i < 31; i = i + 2)
	//medianBlur(gray, blur, i);
	//blur(gray, blur, Size(i, i), Point(-1, -1));
	
	namedWindow("Gaussian",1);
	createTrackbar("GaussianKerSize", "Gaussian", &GaussSize, gauss_slider_max, trackGaussian);
	trackGaussian(GaussSize, 0);


	namedWindow("MeanShift", 1);
	createTrackbar("sp", "MeanShift", &sp, 10, trackMeanShift);
	trackMeanShift(sp, 0);
	createTrackbar("sr", "MeanShift", &sr, 10, trackMeanShift);
	trackMeanShift(sr, 0);
	createTrackbar("maxLevel", "MeanShift", &maxLevel, 10, trackMeanShift);
	trackMeanShift(maxLevel, 0);

	
	//cvtColor(blurred, blurred, CV_BGR2GRAY);

	/*namedWindow("AdaptiveMean", 1);
	createTrackbar("blockSize", "AdaptiveMean", &meanBlock, 10, trackAdapMean);
	trackAdapMean(meanBlock, 0);
	createTrackbar("C value", "AdaptiveMean", &C, 10, trackAdapMean);
	trackAdapMean(C, 0);*/

	namedWindow("AdaptiveGaussian", 1);
	createTrackbar("blockSize", "AdaptiveGaussian", &meanBlock1, 10, trackAdapGauss);
	trackAdapGauss(meanBlock1, 0);
	createTrackbar("C value", "AdaptiveGaussian", &C1, 10, trackAdapGauss);
	trackAdapGauss(C1, 0);

	
	/*namedWindow("Morph", 1);
	createTrackbar("Dilation Size", "Morph", &dilation_size, 10, trackDilationandErosion);
	trackDilationandErosion(dilation_size, 0);
	createTrackbar("Erosion Size", "Morph", &erosion_size, 10, trackDilationandErosion);
	trackDilationandErosion(erosion_size, 0);*/

	namedWindow("Morph1", 1);
	createTrackbar("Dilation Size", "Morph1", &dilation_size1, 10, trackDilationandErosion1);
	trackDilationandErosion1(dilation_size1, 0);
	createTrackbar("Erosion Size", "Morph1", &erosion_size1, 10, trackDilationandErosion1);
	trackDilationandErosion1(erosion_size1, 0);
	
	//use histogram n
	waitKey(0);
	//bilateralFilter(gray, blur, i, i * 2, i / 2);
	//threshold(blur, output2,0,255, THRESH_OTSU);
	//adaptiveThreshold(blurred, output, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 5);
	//adaptiveThreshold(blurred, output1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, 2);

	//threshold(output, output2, 10, 255, THRESH_OTSU);

	
	/*

	int dilation_size = 1;
	int erosion_size = 1;
	dilate(output, output, getStructuringElement(
		MORPH_RECT,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size)));

	erode(output, output, getStructuringElement(
		MORPH_RECT,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size)));
		*/

	/*imshow("mean", output);
	imshow("gaussain", output1);
	//imshow("otsu", output2);
	waitKey(0);

	uchar* p;
	uchar* q;
	int k = 0;
	q = output.ptr<uchar>(k);
	//uchar B, G, R;
	for (int j = 0; j < gray.rows; j++)
	{
		p = gray.ptr<uchar>(j);
		for (int i = 0; i < gray.cols; i++)
		{
			if (*q ==255)
				*p = 255;
			q++;
			p++;
			
		}
		q = output.ptr<uchar>(k++);
	}


	namedWindow("display");
	namedWindow("gray");
	imshow("display", output);
	imshow("gray", gray);
	waitKey(0);
	*/
}