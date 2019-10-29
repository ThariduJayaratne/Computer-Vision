
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat image = imread("mandrill.jpg", 0);
    Mat result(image.rows, image.cols, CV_8UC1);
    int egg[3][3] = {
      1,1,1,
      1,1,1,
      1,1,1
    };

  for(int y=1;y<image.rows-1;y++){
    for(int x =1;x<image.cols-1;x++){
    int pixel = 0;
      for(int i=-1;i<2;i++){
          for(int j=-1;j<2;j++){
              pixel +=  image.at<uchar>(y-i,x-j) * egg[i+1][j+1];
          }
      }
          result.at<uchar>(y,x) = pixel/9;

    }
  }
  imwrite("convolutionpic.jpg", result);

  return 0;
}
