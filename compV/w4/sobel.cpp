
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath.h>

using namespace cv;
int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 Mat image5(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image2(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image3(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image4(image.rows,image.cols, CV_8UC1, Scalar(0));

 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 sobel(gray_image,gray_image.size,image2,image3,image4,image5);

 return 0;
}

void sobel(
	cv::Mat &input,
	int size,
	cv::Mat &output,cv::Mat &output2,cv::Mat &output3,cv::Mat &output4);

void sobel(cv::Mat &input, int size, cv::Mat &output1,cv::Mat &output2,cv::Mat &output3,cv::Mat &output4)
{
  output1.create(input.size(), input.type());
  output2.create(input.size(), input.type());
  output3.create(input.size(), input.type());
  output4.create(input.size(), input.type());


  cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput,
	1, 1, 1, 1,
	cv::BORDER_REPLICATE );
  int xdir[3][3] = {
    -1,0,1,
    2,0,2,
    -1,0,1
  };
  int ydir[3][3] = {
    -1,2,-1,
    0,0,0,
    1,2,1
  };
  for ( int i = 0; i < input.rows; i++ )
  {
    for( int j = 0; j < input.cols; j++ )
    {
      double sum = 0.0;
      for( int m = -1; m <= 1; m++ )
      {
        for( int n = -1; n <= 1; n++ )
        {
          // find the correct indices we are using
          int imagex = i + m + 1;
          int imagey = j + n + 1;
          int kernelx = m + 1;
          int kernely = n + 1;

          // get the values from the padded image and the kernel
          int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
          double dx = xdir[kernelx][kernely];
          double dy = ydir[kernelx][kernely];

          // do the multiplication
          sum1 += imageval * dx;
          sum2 += imageval * dy;

        }
      }
      // set the output value as the sum of the convolution
      output1.at<uchar>(i, j) = (uchar) sum1;
      output2.at<uchar>(i, j) = (uchar) sum2;
      output3.at<uchar>(i, j) = (uchar) sqrt((sum1*sum1)+(sum2*sum2));
      output4.at<uchar>(i, j) = (uchar) atan2(sum2/sum1);
    }
  }
}
