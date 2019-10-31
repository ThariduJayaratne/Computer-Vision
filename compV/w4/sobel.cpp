
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;

void sobel(
	cv::Mat &input,
	int size,
	cv::Mat &output1,cv::Mat &output2,cv::Mat &output3,cv::Mat &output4);

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
  // int xdir[3][3] = {
  //   -1,0,1,
  //   2,0,2,
  //   -1,0,1
  // };
  // int ydir[3][3] = {
  //   -1,2,-1,
  //   0,0,0,
  //   1,2,1
  // };
  Mat kernelx(3,3, CV_8UC1, Scalar(0));
  Mat kernely(3,3, CV_8UC1, Scalar(0));

  kernelx.at<uchar>(0,0) = -1;
  kernelx.at<uchar>(0,1) = 0;
  kernelx.at<uchar>(0,2) = 1;
  kernelx.at<uchar>(1,0) = 2;
  kernelx.at<uchar>(1,1) = 0;
  kernelx.at<uchar>(1,2) = 2;
  kernelx.at<uchar>(2,0) = -1;
  kernelx.at<uchar>(2,1) = 0;
  kernelx.at<uchar>(2,2) = -1;

  kernely = kernelx.t();

  int dx = 0;
  int dy = 0;

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
          int xd = m + 1;
          int yd = n + 1;

          // get the values from the padded image and the kernel
          int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
          double x = kernelx.at<double>( xd, yd );
          double y = kernely.at<double>( xd, yd );

          // do the multiplication
           dx += imageval * x;
           dy += imageval * y;

        }
      }
      // set the output value as the sum of the convolution
      output1.at<uchar>(i, j) = (uchar) dx;
      output2.at<uchar>(i, j) = (uchar) dy;
      output3.at<uchar>(i, j) = (uchar) sqrt((dx*dx)+(dy*dy));
      output4.at<uchar>(i, j) = (uchar) atan2(dy,dx);
    }
  }
}

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];
 Mat image = imread( imageName, 1 );
 Mat image5(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image2(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image3(image.rows,image.cols, CV_8UC1, Scalar(0));
 Mat image4(image.rows,image.cols, CV_8UC1, Scalar(0));


 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 // CONVERT COLOUR, BLUR AND SAVE
 // CONVERT COLOUR, BLUR AND SAVE
 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 sobel(gray_image,gray_image.rows,image2,image3,image4,image5);
 imwrite("dx.jpg", image2);
 imwrite("dy.jpg", image3);
 imwrite("magnitude.jpg", image4);
 imwrite("arctan.jpg", image5);

 return 0;
}
