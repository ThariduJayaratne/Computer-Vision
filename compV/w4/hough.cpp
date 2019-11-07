#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

Mat sobel()
{
  Mat input = imread("coin1.png",0);
  Mat output1(input.rows,input.cols, CV_32FC1, Scalar(0));
  Mat output2(input.rows,input.cols, CV_32FC1, Scalar(0));
  Mat kernelx = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat kernely = kernelx.t();

 for(int y=1;y<input.rows-1;y++){
    for(int x =1;x<input.cols-1;x++){
       float pixelx = 0;
     float pixely = 0;
      for(int i=-1;i<2;i++){
          for(int j=-1;j<2;j++){
              pixelx +=  input.at<uchar>(y+i,x+j) * kernelx.at<int>(i+1,j+1);
             pixely +=  input.at<uchar>(y+i,x+j) * kernely.at<int>(i+1,j+1);
          }

      }
     output1.at<float>(y, x) = sqrt((pixelx*pixelx)+(pixely*pixely));
     output2.at<float>(y, x) = atan2(pixely,pixelx);
     }
   }
   cv::normalize(output1,output1,0,255,NORM_MINMAX);
   return output1, output2;
}


Mat threshold(int val){
  Mat image = imread("magnitude.jpg",0);
  Mat result(image.rows, image.cols, CV_8UC1,Scalar(0));
  for(int x=0; x<image.rows; x++) {
   for(int y=0; y<image.cols; y++) {
     if (image.at<uchar>(x,y)>val) {
       result.at<uchar>(x,y)=255;
     }
     else {
       result.at<uchar>(x,y)=0;
     }
   }
 }
 return result;
}

void hough(Mat threholded){
  thresholded = threshold(100);
  Mat mag,dir = sobel();
  int xo, yo = 0;
  for(int x=0;x<thresholded.rows,x++){
    for(int y =0;y<thresholded.cols,y++){
      if(thresholded.at<uchar>(i,j)==255){
        //xo = x + r cos(direction val)
      }
    }
  }
}



int main(int argc, char** argv) {
  // char* imageName = argv[1];
  // Mat image = imread(imageName,1);
  imwrite("thresholded.jpg",threshold(110));


  return 0;
}
