/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

Mat img4 = (Mat_<int>(1,4) << 362, 128, 460, 243);

Mat img5 = (Mat_<int>(11,4) << 67,147,118,192,
															 59,258,111,311,
														 	 200,226,249,276,
															 255,172,301,222,
															 299,244,342,298,
															 439,241,480,296,
															 384,195,434,240,
															 518,186,565,232,
															 566,249,613,308,
															 682,255,726,309,
															 651,192,697,237);

Mat img13 = (Mat_<int>(1,4) << 427,123,521,244);

Mat img14 = (Mat_<int>(2,4) << 732,196,825,285,
															 474,217,550,315);

Mat img15 = (Mat_<int>(3,4) << 73,132,131,214,
															 370,114,429,192,
															 539,131,608,216);

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

	// for (int i = 0; i < img15.rows; i++) {
	// 	rectangle(frame, Point(img15.at<int>(i,0), img15.at<int>(i,1)), Point(img15.at<int>(i,2), img15.at<int>(i,3)), Scalar(0, 0, 255), 2);
	// }
	// rectangle(frame, Point(362, 128), Point(460, 243), Scalar(0, 0, 255), 2);
	// 4. Save Result Image
	// imwrite( "dart15T.jpg", frame );

// int x,y = widthAndHeight(img5);
// printf("%f",x);

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
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
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
		printf("Area: %d\n", faces[i].width * faces[i].height);
	}
		//Draw box around true Faces
		for (int i = 0; i < img4.rows; i++) {
			rectangle(frame, Point(img4.at<int>(i,0), img4.at<int>(i,1)), Point(img4.at<int>(i,2), img4.at<int>(i,3)), Scalar(0, 0, 255), 2);
			width = (img4.at<int>(i,2)) - (img4.at<int>(i,0));
			height = (img4.at<int>(i,3)) - (img4.at<int>(i,1));
			int area = width * height;
			printf("Area of truth: %d\n", area);
		}

}
