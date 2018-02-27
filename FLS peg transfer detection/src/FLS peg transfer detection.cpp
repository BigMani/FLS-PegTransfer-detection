#pragma once
#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <stdio.h>
#include "cv.h"
#include "cvaux.h"
#include <cxcore.h>
#include "highgui.h"
#include <videoInput.h>

//Initializations
static CvMemStorage *storage = cvCreateMemStorage(0);
int screenshot = 555;
int key = 0;
CvSeq *contour, *lines, *blocks;
CvPoint2D32f boxPoints[4];
CvPoint point1, point2;

// Take a screenshot
void captureScreenshot(IplImage *img)
{
	std::stringstream out;
	out << screenshot;
	string s = out.str();
	string file = "positive" + s + ".jpg";
	char *a = new char[file.size() + 1];
	strcpy(a, file.c_str());
	cvSaveImage(a, img);
	screenshot += 1;
}

// Filter image (erosion + dilate)
void filterNoise(IplImage *img, int filteriterations)
{
	cvErode(img, img, 0, filteriterations);
	cvDilate(img, img, 0, filteriterations);
}

// Draw minimum red area box around the input contour
void boxContour(IplImage *in, const CvArr* contour, CvMemStorage* storage)
{
	CvBox2D box = cvMinAreaRect2(contour, storage);
	cvBoxPoints(box, boxPoints);
	point1 = cvPoint((int)boxPoints[1].x, (int)boxPoints[1].y);
	cvLine(in, cvPoint((int)boxPoints[0].x, (int)boxPoints[0].y), cvPoint((int)boxPoints[1].x, (int)boxPoints[1].y),
		CV_RGB(255, 0, 0), 3);
	cvLine(in, cvPoint((int)boxPoints[1].x, (int)boxPoints[1].y), cvPoint((int)boxPoints[2].x, (int)boxPoints[2].y),
		CV_RGB(255, 0, 0), 3);
	cvLine(in, cvPoint((int)boxPoints[2].x, (int)boxPoints[2].y), cvPoint((int)boxPoints[3].x, (int)boxPoints[3].y),
		CV_RGB(255, 0, 0), 3);
	cvLine(in, cvPoint((int)boxPoints[3].x, (int)boxPoints[3].y), cvPoint((int)boxPoints[0].x, (int)boxPoints[0].y),
		CV_RGB(255, 0, 0), 3);
}

// Convert image to thresholded binary image (inverse)
void convertToBinary(IplImage *in, IplImage *out, int min, int max, const char* type)
{
	if (type == "INV")
	{
		cvCvtColor(in, out, CV_RGB2GRAY);
		cvThreshold(out, out, min, max, CV_THRESH_BINARY_INV);
	}
	else
	{
		cvCvtColor(in, out, CV_RGB2GRAY);
		cvThreshold(out, out, min, max, CV_THRESH_BINARY);
	}
}

// Detect blocks using object detection via AdaBoost
void findBlocks(IplImage *in, CvHaarClassifierCascade* cascade)
{
	IplImage* temp_img = in;
	int scale = 2;
	temp_img = cvCreateImage(cvSize(in->width / 2, in->height / 2), IPL_DEPTH_8U, 3);
	cvPyrDown(in, temp_img, CV_GAUSSIAN_5x5);
	blocks = cvHaarDetectObjects(temp_img, cascade, storage, 1.2, 4, CV_HAAR_DO_CANNY_PRUNING);
	// draw all the rectangles
	for (int i = 0; i < blocks->total; i++)
	{
		CvRect block_rect = *(CvRect*)cvGetSeqElem(blocks, i);
		point2 = cvPoint(block_rect.x*scale, block_rect.y*scale);
		cvRectangle(in, cvPoint(block_rect.x*scale, block_rect.y*scale), cvPoint((block_rect.x + block_rect.width)*scale,
			(block_rect.y + block_rect.height)*scale), CV_RGB(0, 255, 0), 4);
	}
}

// Detect Tools and mark the endpoints
void findTools(IplImage *in, IplImage *out)
{
	convertToBinary(in, out, 40, 250, "INV");
	filterNoise(out, 4);
	cvFindContours(out, storage, &contour);
	for (; contour; contour = contour->h_next)
	{
		double area = fabs(cvContourArea(contour, CV_WHOLE_SEQ));
		if (area > 1000)
		{
			boxContour(in, contour, storage);
		}
	}
}

// Main Function
int _tmain(int argc, _TCHAR* argv[])
{
	//Capture from camera
	int device1 = 1;
	videoInput VI;
	int numDevices = VI.listDevices();
	VI.setupDevice(device1);
	int width = VI.getWidth(device1);
	int height = VI.getHeight(device1);
	unsigned char* yourBuffer = new unsigned char[VI.getSize(device1)];
	//Create variables for image storage
	int fps = 1000 / 300; //Denominator is the desired fps
	IplImage *imgOriginal = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *imgResult = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	IplImage *imgThresh = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	//Load trainined xml file
	CvHaarClassifierCascade* blockcascade = cvLoadHaarClassifierCascade("haarcascade6.xml", cvSize(20, 20));
	//Main algorithm loop

	while (key != 27)
	{
		VI.getPixels(device1, yourBuffer, false, false);
		imgOriginal->imageData = (char*)yourBuffer;
		cvConvertImage(imgOriginal, imgOriginal, CV_CVTIMG_FLIP);
		findBlocks(imgOriginal, blockcascade);
		findTools(imgOriginal, imgThresh);
		cvShowImage("Final", imgOriginal);
		if (key == 99)
		{
			captureScreenshot(imgOriginal);
		}
		key = cvWaitKey(fps);
	}
	//Memory handling
	VI.stopDevice(device1);
	cvDestroyWindow("Original");
	cvDestroyWindow("Thresh");
	cvDestroyWindow("Final");
	cvReleaseImage(&imgOriginal);
	return 0;
}