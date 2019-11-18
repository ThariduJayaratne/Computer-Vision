
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;



/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load thfile:///usr/share/doc/HTML/index.htmle Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	//Real dartboards
	std::vector<Rect> img0;
	img0.push_back(Rect(442, 12, 151, 169));

	std::vector<Rect> img1;
	img1.push_back(Rect(191, 130, 203, 193));

	std::vector<Rect> img2;
	img2.push_back(Rect(106, 98, 84, 84));

	std::vector<Rect> img3;
	img3.push_back(Rect(326, 149, 64, 67));

	std::vector<Rect> img4;
	img4.push_back(Rect(186, 96, 182, 212));

	std::vector<Rect> img5;
	img5.push_back(Rect(430, 139, 114, 117));

	std::vector<Rect> img6;
	img6.push_back(Rect(212, 117, 63, 64));

	std::vector<Rect> img7;
	img7.push_back(Rect(257, 174, 133, 138));

	std::vector<Rect> img8;
	img8.push_back(Rect(63, 251, 63, 97));
	img8.push_back(Rect(845, 218, 110, 116));

	std::vector<Rect> img9;
	img9.push_back(Rect(210, 49, 228, 226));

	std::vector<Rect> img10;
	img10.push_back(Rect(90, 106, 99, 109));
	img10.push_back(Rect(588, 137, 55, 77));
	img10.push_back(Rect(919, 150, 31, 65));


	std::vector<Rect> img11;
	img11.push_back(Rect(179, 108, 51, 73));
	img11.push_back(Rect(214, 255, 15, 19));
	img11.push_back(Rect(441, 113, 56, 75));

	std::vector<Rect> img12;
	img12.push_back(Rect(165, 77, 51, 141));

	std::vector<Rect> img13;
	img13.push_back(Rect(282, 120, 118, 129));

	std::vector<Rect> img14;
	img14.push_back(Rect(124, 103, 120, 122));
	img14.push_back(Rect(991, 100, 119, 116));

	std::vector<Rect> img15;
	img15.push_back(Rect(158, 59, 128, 138));

	//Detected dartboards
	std::vector<Rect> dartboards;
	Mat frame_gray;
	int width,height;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500) );

       // 3. Print number of Faces found
	std::cout << dartboards.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(frame, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}

	float truePositives = 0;

	for( int i = 0; i < img1.size(); i++ )
	{
		rectangle(frame, Point(img1[i].x, img1[i].y), Point(img1[i].x + img1[i].width, img1[i].y + img1[i].height), Scalar( 0, 0, 255 ), 2);
	}

	float falsePositives = 0;
	for (int i = 0; i < dartboards.size(); i++) {
		int correct = 0;
		for (int j = 0; j < img1.size(); j++) {
			Rect r1(dartboards[i].x, dartboards[i].y,dartboards[i].width,dartboards[i].height);
			Rect r2(img1[j].x, img1[j].y,img1[j].width,img1[j].height);
			Rect intersection = r1 & r2;
			Rect unionA = r1 | r2;
			float iWidth = intersection.width;
			float iHeight = intersection.height;
			float uWidth = unionA.width;
			float uHeight = unionA.height;
			float areaRatio = (iWidth * iHeight) / (uWidth * uHeight);
			if (areaRatio > 0.5f) {
				correct = 1;
				truePositives += 1;
				printf("IOU is %f\n", i, j, areaRatio);
			 }
		}
		if (correct == 0) {
			falsePositives++;
		}
	}
	float tpr = truePositives/dartboards.size();
	float precision = truePositives/(truePositives+falsePositives);
	float f1score = (2*(tpr*precision))/(tpr+precision);
	printf("True positives: %f\n", truePositives);
	printf("False positives: %f\n", falsePositives);
	printf("Tpr: %f\n", tpr);
	printf("F1 score is: %f\n", f1score);

	imwrite("dart1T.jpg", frame);

}
