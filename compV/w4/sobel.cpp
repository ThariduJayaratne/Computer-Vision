#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <math.h>

using namespace cv;

void sobel(
	cv::Mat &input,
	cv::Mat &output1,cv::Mat &output2,cv::Mat &output3,cv::Mat &output4);

void sobel(cv::Mat &input, cv::Mat &output1,cv::Mat &output2,cv::Mat &output3,cv::Mat &output4)
{
  Mat kernelx = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
	Mat kernely = kernelx.t();

	for(int y=1;y<input.rows-1;y++){
    for(int x =1;x<input.cols-1;x++){
   		float pixelx = 0;
			float pixely = 0;
      for(int i=-1;i<2;i++){
          for(int j=-1;j<2;j++){
              pixelx +=  input.at<uchar>(y-i,x-j) * kernelx.at<int>(i+1,j+1);
							pixely +=  input.at<uchar>(y-i,x-j) * kernely.at<int>(i+1,j+1);
          }

      }
			output1.at<float>(y, x) = pixelx;
			output2.at<float>(y, x) = pixely;
			output3.at<float>(y, x) = sqrt((pixelx*pixelx)+(pixely*pixely));
			output4.at<float>(y, x) = atan2(pixely,pixelx);

			}
		}
		cv::normalize(output1,output1,0,255,NORM_MINMAX);
		cv::normalize(output2,output2,0,255,NORM_MINMAX);
		cv::normalize(output3,output3,0,255,NORM_MINMAX);
		cv::normalize(output4,output4,0,255,NORM_MINMAX);
		imwrite("dx.jpg", output1);
		imwrite("dy.jpg", output2);
		imwrite("magnitude.jpg", output3);
		imwrite("arctan.jpg", output4);
}

int main( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];
 Mat image = imread( imageName, 1 );
 Mat image2(image.rows,image.cols, CV_32FC1, Scalar(0));
 Mat image3(image.rows,image.cols, CV_32FC1, Scalar(0));
 Mat image4(image.rows,image.cols, CV_32FC1, Scalar(0));
 Mat image5(image.rows,image.cols, CV_32FC1, Scalar(0));

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 Mat gray_image;
 cvtColor( image, gray_image, CV_BGR2GRAY );

 sobel(gray_image,image2,image3,image4,image5);


 return 0;
}
