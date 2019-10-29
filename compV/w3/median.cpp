
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat image = imread("car2.png", 1);
  
  for(int x=0;x<image.rows;x++){
    for(int y =0;y<image.cols;y++){
        uchar neighbours[9] = {
            image.at<uchar>(x,y),
            image.at<uchar>(x+1,y),
            image.at<uchar>(x-1,y),
            image.at<uchar>(x,y+1),
            image.at<uchar>(x,y-1),
            image.at<uchar>(x+1,y-1),
            image.at<uchar>(x+1,y+1),
            image.at<uchar>(x-1,y-1),
            image.at<uchar>(x,y),
        }
    }
  }
  imwrite("corrected3.jpg", image2);

  return 0;
}