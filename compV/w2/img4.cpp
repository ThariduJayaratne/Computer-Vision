
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat image = imread("mandrill3.jpg", 1);

  bitwise_not(image,image);
  imwrite("corrected4.jpg", image);

  return 0;
}