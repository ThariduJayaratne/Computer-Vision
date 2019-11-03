#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

void sobel(
 cv::Mat &input,
 cv::Mat &output1);

void sobel(cv::Mat &input, cv::Mat &output1)
{
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
     }
   }
   cv::normalize(output1,output1,0,255,NORM_MINMAX);

   imwrite("thresholded.jpg", output1);
}

int main(int argc, char** argv) {
  char* imageName = argv[1];
  Mat image = imread( imageName, 1 );
  Mat result(image.rows, image.cols, CV_32FC1,Scalar(0));
  Mat output(image.rows,image.cols, CV_32FC1, Scalar(0));

  for(int x=0; x<image.rows; x++) {
   for(int y=0; y<image.cols; y++) {
     if (image.at<uchar>(x,y)>100) {
       result.at<uchar>(x,y)=255;
     }
     else {
       result.at<uchar>(x,y)=0;
     }
   }
 }
  sobel(result,output);
  return 0;
}
