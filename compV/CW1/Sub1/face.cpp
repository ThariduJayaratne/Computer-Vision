
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
String cascade_name = "frontalface.xml";
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
	std::vector<Rect> img0;
	img0.push_back(Rect(176,208,95,92));

	std::vector<Rect> img6;
	img6.push_back(Rect(282,96,47,64));

	std::vector<Rect> img7;
	img7.push_back(Rect(346,183,87,113));

  std::vector<Rect> img11;
  img11.push_back(Rect(322,66,68,92));

	std::vector<Rect> img9;
  img9.push_back(Rect(86,220,120,125));

  std::vector<Rect> img8;
  img8.push_back(Rect(164,274,64,82));

	std::vector<Rect> img4;
	img4.push_back(Rect(350,101,127,159));

	std::vector<Rect> img5;
	img5.push_back(Rect(67,147,51,45));
	img5.push_back(Rect(59,258,52,53));
	img5.push_back(Rect(200,226,49,50));
	img5.push_back(Rect(255,172,46,50));
	img5.push_back(Rect(299,244,43,54));
	img5.push_back(Rect(439,241,41,55));
	img5.push_back(Rect(384,195,50,45));
	img5.push_back(Rect(518,186,47,46));
	img5.push_back(Rect(566,249,47,59));
	img5.push_back(Rect(682,255,44,54));
	img5.push_back(Rect(651,192,46,45));

	std::vector<Rect> img13;
	img13.push_back(Rect(427,123,94,121));

	std::vector<Rect> img14;
	img14.push_back(Rect(732,196,83,89));
	img14.push_back(Rect(474,217,76,98));

	std::vector<Rect> img15;
	img15.push_back(Rect(73,132,58,82));
	img15.push_back(Rect(370,114,59,78));
	img15.push_back(Rect(539,131,69,85));

	std::vector<Rect> faces;
	Mat frame_gray;
	int width,height;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, faces[i], Scalar( 0, 255, 0 ), 2);
	}
	float truePositives = 0;

	for( int i = 0; i < img6.size(); i++ )
	{
		rectangle(frame, img6[i], Scalar( 0, 0, 255 ), 2);
	}

	float falsePositives = 0;
	float falseNegatives = 0;

	for (int i = 0; i < faces.size(); i++) {
		int correct = 0;
		for (int j = 0; j < img6.size(); j++) {
			Rect r1(faces[i].x, faces[i].y,faces[i].width,faces[i].height);
			Rect r2(img6[j].x, img6[j].y,img6[j].width,img6[j].height);
			Rect intersection = r1 & r2;
			Rect unionA = r1 | r2;
			float iWidth = intersection.width;
			float iHeight = intersection.height;
			float uWidth = unionA.width;
			float uHeight = unionA.height;
			float areaRatio = (iWidth * iHeight) / (uWidth * uHeight);
			if (areaRatio >= 0.5f) {
				correct = 1;
				truePositives += 1;
				printf("IOU is %f\n",areaRatio);
			 }
		}
		if (correct == 0) {
			falsePositives++;
		}
	  falseNegatives = img6.size()-truePositives;
	}
	// float tpr = truePositives/faces.size();
	float recall = truePositives/(truePositives+falseNegatives);
	float precision = truePositives/(truePositives+falsePositives);
	float f1score = 2*((recall*precision)/(recall+precision));
	printf("Tpr: %f\n", recall);

	printf("True positives: %f\n", truePositives);
	printf("precision: %f\n", precision);
	printf("False positives: %f\n", falsePositives);
	printf("False negatives: %f\n", falseNegatives);
	printf("F1 score is: %f\n", f1score);

	imwrite("dart6T.jpg", frame);

}
