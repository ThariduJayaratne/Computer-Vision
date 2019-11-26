#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;

String cascade_name = "cascade.xml";
CascadeClassifier cascade;

Mat sobel (Mat input/*, Mat dx, Mat dy, Mat magnitude, Mat direction*/) {

  Mat dx(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat dy(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat magnitude(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat direction(input.rows, input.cols, CV_32FC1, Scalar(0));

  Mat kernelx = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat kernely = kernelx.t();

  for (int x = 1; x < input.rows - 1; x++) {
    for (int y = 1; y < input.cols - 1; y++) {

      float pixelx = 0;
      float pixely = 0;

      for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {

          pixelx += input.at<uchar>(x+i, y+j) * kernelx.at<int>(i+1, j+1);
          pixely += input.at<uchar>(x+i, y+j) * kernely.at<int>(i+1, j+1);

        }
      }

      dx.at<float>(x,y) = pixelx;
      dy.at<float>(x,y) = pixely;
      magnitude.at<float>(x, y) = sqrt((pixelx * pixelx) + (pixely * pixely));
      direction.at<float>(x, y) = atan2(pixely,pixelx);
    }
  }

  normalize(dx, dx, 0, 255, NORM_MINMAX);
  normalize(dy, dy, 0, 255, NORM_MINMAX);
  normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
  // normalize(direction, direction, 0 , 255, NORM_MINMAX);

  imwrite("dx.jpg", dx);
  imwrite("dy.jpg", dy);
  imwrite("magnitude.jpg", magnitude);

  return direction;
}

void threshold(Mat input, int val) {

  Mat output(input.rows, input.cols, CV_8UC1, Scalar(0));

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {

      if (input.at<uchar>(i, j) > val) {
        output.at<uchar>(i, j) = 255;
      }
      else {
        output.at<uchar>(i, j) = 0;
      }
    }
  }
  imwrite("threshold.jpg", output);
}

// void houghline(Mat threshold, Mat direction){
//   int diagonal = sqrt((threshold.rows*threshold.rows)+(threshold.cols*threshold.cols));
//   Mat output(threshold.rows, threshold.cols, CV_32FC1, Scalar(0));
//   int accumulator[180][diagonal];
//   for (int i = 0; i < diagonal; i++) {
//     for (int j = 0; j < 180 ; j++) {
//         accumulator[j][i] = 0;
//       }
//     }
//     for (int x = 0; x < threshold.rows; x++) {
//       for (int y = 0; y < threshold.cols; y++) {
//           float angle = (direction.at<float>(x,y)*180)/3.1415926;
//           int variation = 5;
//           if (threshold.at<uchar>(x, y) == 255) {
//             float minA,maxA =0;
//             if (angle<0) minA += 180+variation;
//             else if(angle>180) maxA -= 180+variation;
//             for(int z =0;z<180;z++){
//               if(angle>minA && angle<maxA){
//               float perpenD = x*cos(z)+ y*sin(z);
//               if(perpenD<=diagonal){
//                 accumulator[z][(int)perpenD] +=1;
//               }
//             }
//           }
//         }
//       }
//     }
//     normalize(output, output, 0, 255, NORM_MINMAX);
//     imwrite("dart3lines.jpg", output);
// }


int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));
    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	for (j = 0; j < dim2; j++) {
  	    array[i][j] = (int *) malloc(dim3 * sizeof(int));
	}
    }
    return array;
}

vector<Rect> hough(Mat threshold, Mat direction, int radius) {

  Mat output(threshold.rows, threshold.cols, CV_32FC1, Scalar(0));
  int ***accumulator = malloc3dArray(threshold.rows,threshold.cols,radius);
  for (int i = 0; i < threshold.rows; i++) {
    for (int j = 0; j < threshold.cols; j++) {
      for (int r = 0; r < radius; r++) {
        accumulator[i][j][r] = 0;
      }
    }
  }

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at<uchar>(x, y) >= 255) {
        for (int r = 5; r <= radius; r++) {

          int xo = x + r * sin(direction.at<float>(x, y));
          int yo = y + r * cos(direction.at<float>(x, y));

          if (xo >= 0 && yo >= 0 && xo < threshold.rows && yo < threshold.cols) {
            output.at<float>(xo, yo) += 1;
            accumulator[xo][yo][r] += 1;
          }

          int xon = x - r * sin(direction.at<float>(x, y));
          int yon = y - r * cos(direction.at<float>(x, y));

          if (xon >= 0 && yon >= 0 && xon < threshold.rows && yon < threshold.cols) {
            output.at<float>(xon, yon) += 1;
            accumulator[xon][yon][r] += 1;
          }
        }
      }
    }
  }

  normalize(output, output, 0, 255, NORM_MINMAX);
  std::vector<Rect> img;
  int xdivisions = output.rows/4;
  int ydivisions = output.cols/4;
  for (int x = 0; x < xdivisions; x++) {
    for (int y = 0; y < ydivisions; y++) {
      int found = 0;
      int completed = 0;
      while (found == 0 && completed == 0) {
        for (int i = x * xdivisions; (i < (x + 1) * xdivisions) && (i < output.rows); i++) {
          for (int j = y * ydivisions; (j < (y + 1) * ydivisions) && (j < output.cols); j++) {
            if (output.at<float>(i, j) >= 220) {
              int maxRadius = 0;
              int temp = 0;
              for (int r = 0; r < radius; r++) {
                if (accumulator[i][j][r] > temp) {
                  temp = accumulator[i][j][r];
                  maxRadius = r;
                }
              }
              circle(output, Point(j, i), maxRadius, Scalar(0, 0, 255), 2);
              img.push_back(Rect((j-maxRadius) % output.cols, (i-maxRadius) % output.rows, 2*maxRadius % output.cols, 2*maxRadius % output.rows));
              found = 1;
              break;
            }
            if (found == 1) break;
          }
        }
        completed = 1;
      }
    }
  }

  // imwrite("dart0hough.jpg", output);
  return img;
}


void detectAndDisplay( Mat input )
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

  int width,height;

  Mat gray_image;
  cvtColor( input, gray_image, CV_BGR2GRAY );

  Mat direction = sobel(gray_image);

  Mat magnitude = imread("magnitude.jpg", 0);
  threshold(magnitude, 30);

  Mat threshold = imread("threshold.jpg", 0);

  std::vector<Rect> imgDetected = hough(threshold, direction, 150);
  // houghline(magnitude,direction);

  equalizeHist( gray_image, gray_image );
  std::vector<Rect> dartboards;

  cascade.detectMultiScale( gray_image, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500) );

  //Viola-Jones
  for( int i = 0; i < dartboards.size(); i++ )
  {
    rectangle(input,dartboards[i], Scalar( 255, 0, 0 ), 2);
  }
  std::cout << dartboards.size() << std::endl;

  //Hough
  for( int k = 0; k < imgDetected.size(); k++ )
  {
    rectangle(input, imgDetected[k], Scalar( 0, 255, 0 ), 2);
  }
  std::cout << imgDetected.size() << std::endl;

  std::vector<Rect> accurateViola;
  for (int i = 0; i < imgDetected.size(); i++) {
    int found = 0;
		for (int j = 0; j < dartboards.size(); j++) {
			Rect r1(imgDetected[i].x, imgDetected[i].y,imgDetected[i].width,imgDetected[i].height);
			Rect r2(dartboards[j].x, dartboards[j].y,dartboards[j].width,dartboards[j].height);
			Rect intersection = r1 & r2;
			Rect unionA = r1 | r2;
			float iWidth = intersection.width;
			float iHeight = intersection.height;
			float uWidth = unionA.width;
			float uHeight = unionA.height;
			float areaRatio = (iWidth * iHeight) / (uWidth * uHeight);
			if (areaRatio >= 0.3f) {
        accurateViola.push_back(Rect(dartboards[j].x, dartboards[j].y,dartboards[j].width,dartboards[j].height));
        found = 1;
        break;
			 }
       if (found == 1) break;
		}

    //Combined
    for( int i = 0; i < accurateViola.size(); i++ )
    {
      rectangle(input, accurateViola[i], Scalar( 0, 0, 255 ), 2);
    }
    std::cout << accurateViola.size() << std::endl;
	// 	if (correct == 0) {
	// 		falsePositives++;
	// 	}
	// 	falseNegatives = img4.size()-truePositives;
	// }
	// float recall = truePositives/(truePositives+falseNegatives);
	// float precision = truePositives/(truePositives+falsePositives);
	// float f1score = 2*((recall*precision)/(recall+precision));
	// printf("Tpr: %f\n", recall);
	// printf("True positives: %f\n", truePositives);
	// printf("False positives: %f\n", falsePositives);
  // printf("False negatives: %f\n", falseNegatives);
	// printf("F1 score is: %f\n", f1score);
	imwrite("dart5Comb.jpg", input);

  // float truePositives = 0;
  // float falsePositives = 0;
  // float falseNegatives = 0;
  // for (int i = 0; i < accurateViola.size(); i++) {
  //   int correct = 0;
  //   for (int j = 0; j < img4.size(); j++) {
  //
  //   }
  // }
}
}

int main() {
  Mat input = imread("dart5.jpg", 1);
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  detectAndDisplay(input);
  return 0;
}
