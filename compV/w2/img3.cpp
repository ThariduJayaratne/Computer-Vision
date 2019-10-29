
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat image = imread("mandrill1.jpg", 1);
  Mat image2 = imread("mandrill1.jpg", 1);
  
  for(int y=0;y<image.rows;y++){
    for(int x =0;x<image.cols;x++){
      image2.at<Vec3b>((y+30)%image.rows,(x+30)%image.cols)[2] = image.at<Vec3b>(y,x)[2];
    }
  }
  imwrite("corrected3.jpg", image2);

  return 0;
}
