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
//           int variation = 50;
//           if (threshold.at<uchar>(x, y) == 255) {
//             float minA,maxA;
//             if (angle<0) minA += 180+variation;
//             else if(angle>180) maxA -= 180+variation;
//             else{
//               minA = angle - variation;
//               maxA = angle + variation;
//             }
//             for(int z =minA;z<maxA;z++){
//               float radians = (z*3.1415926)/180;
//               float perpenD = x*cos(radians)+ y*sin(radians);
//               if(perpenD<=diagonal){
//                 accumulator[z][(int)perpenD] +=1;
//                 output.at<float>(z,(int)perpenD) += 1;
//               }
//           }
//         }
//       }
//     }
//     normalize(output, output, 0, 255, NORM_MINMAX);
//     imwrite("dart4lines.jpg", output);
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

vector<Point> hough(Mat threshold, Mat direction, int radius) {

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
  std::vector<Point> img;
  int xdivisions = output.rows/4;
  int ydivisions = output.cols/4;
  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 4; y++) {
      int found = 0;
      int completed = 0;
      while (found == 0 && completed == 0) {
        for (int i = x * xdivisions; (i < (x + 1) * xdivisions) && (i < output.rows); i++) {
          for (int j = y * ydivisions; (j < (y + 1) * ydivisions) && (j < output.cols); j++) {
            if (output.at<float>(i, j) >= 230) {
              int maxRadius = 0;
              int temp = 0;
              for (int r = 0; r < radius; r++) {
                if (accumulator[i][j][r] > temp) {
                  temp = accumulator[i][j][r];
                  maxRadius = r;
                }
              }
              img.push_back(Point(j, i));
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
  return img;
  }

vector<Rect> houghsquares(Mat threshold, Mat direction, int radius) {

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
  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 4; y++) {
      int found = 0;
      int completed = 0;
      while (found == 0 && completed == 0) {
        for (int i = x * xdivisions; (i < (x + 1) * xdivisions) && (i < output.rows); i++) {
          for (int j = y * ydivisions; (j < (y + 1) * ydivisions) && (j < output.cols); j++) {
            if (output.at<float>(i, j) >= 230) {
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
  return img;
}

void detectAndDisplay( Mat input )
{
  int width,height;

  Mat gray_image;
  cvtColor( input, gray_image, CV_BGR2GRAY );

  Mat direction = sobel(gray_image);

  Mat magnitude = imread("magnitude.jpg", 0);
  threshold(magnitude, 30);

  Mat threshold = imread("threshold.jpg", 0);

  std::vector<Point> imgDetected = hough(threshold, direction, 150);
  std::vector<Rect> imgsquares = houghsquares(threshold, direction, 150);

  equalizeHist( gray_image, gray_image );
  std::vector<Rect> dartboards;

  cascade.detectMultiScale( gray_image, dartboards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500) );

  std::cout << dartboards.size() << std::endl;

  for( int k = 0; k < imgsquares.size(); k++ )
  {
    rectangle(input, imgsquares[k], Scalar( 255, 0, 0 ), 2);
  }
  std::cout << imgDetected.size() << std::endl;

  std::vector<Rect> accurateViola;
  for (int i = 0; i < imgDetected.size(); i++) {
    int found = 0;
		for (int j = 0; j < dartboards.size(); j++) {
			Rect r1(imgsquares[i].x, imgsquares[i].y,imgsquares[i].width,imgsquares[i].height);
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
    for( int i = 0; i < accurateViola.size(); i++ )
    {
      rectangle(input, accurateViola[i], Scalar( 0, 255, 0 ), 2);
      rectangle(input, Point(imgDetected[i].x - (accurateViola[i].width)/2 , imgDetected[i].y - (accurateViola[i].height)/2), Point(imgDetected[i].x + (accurateViola[i].width)/2, imgDetected[i].y + (accurateViola[i].height)/2), Scalar( 0, 0, 255 ), 2);
    }
    std::cout << accurateViola.size() << std::endl;
	  imwrite("dart4final.jpg", input);
  }
}

int main() {
  Mat input = imread("dart4.jpg", 1);
  if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  detectAndDisplay(input);
  return 0;
}
