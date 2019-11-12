#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

Mat sobel(Mat input) {

  Mat magnitude(input.rows,input.cols, CV_32FC1, Scalar(0));
  Mat direction(input.rows,input.cols, CV_32FC1, Scalar(0));
  Mat kernelx = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat kernely = kernelx.t();

 for(int y=1;y<input.rows-1;y++) {
    for(int x =1;x<input.cols-1;x++) {
       float pixelx = 0;
       float pixely = 0;
       for(int i=-1;i<2;i++){
          for(int j=-1;j<2;j++){
              pixelx +=  input.at<uchar>(y+i,x+j) * kernelx.at<int>(i+1,j+1);
              pixely +=  input.at<uchar>(y+i,x+j) * kernely.at<int>(i+1,j+1);
          }
        }
        magnitude.at<float>(y, x) = sqrt((pixelx*pixelx)+(pixely*pixely));
        direction.at<float>(y, x) = atan2(pixely,pixelx);
     }
   }
   Mat unnorm = direction;
   cv::normalize(magnitude,magnitude,0,255,NORM_MINMAX);
   cv::normalize(direction,direction,0,255,NORM_MINMAX);


   imwrite("magnitude.jpg", magnitude);
   imwrite("direction.jpg", direction);
   return unnorm;
}

void threshold(Mat input, int val){

  Mat result(input.rows, input.cols, CV_8UC1, Scalar(0));
  for(int x=0; x<input.rows; x++) {
    for(int y=0; y<input.cols; y++) {
      if (input.at<uchar>(x,y)>val) {
        result.at<uchar>(x,y)=255;
      }
      else {
        result.at<uchar>(x,y)=0;
      }
    }
  }
  imwrite("thresholded.jpg", result);
}

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

void hough(Mat thresholded, Mat dir) {
  int ***accumulator = malloc3dArray(thresholded.rows,thresholded.cols,100);
  for(int i = 0;i<thresholded.rows;i++) {
    for(int j = 0;j<thresholded.cols;j++) {
      for(int r = 0;r<100;r++){
      accumulator[i][j][r] = 0;
     }
    }
  }

  for(int y=0;y<thresholded.rows;y++) {
    for(int x =0;x<thresholded.cols;x++) {
      for(int r =20;r<=100;r++){
        if(thresholded.at<uchar>(y,x)==255){
            int yo = y + r*std::sin(dir.at<float>(y,x));
            int xo = x + r*std::cos(dir.at<float>(y,x));
            if(xo>=0 &&yo>=0 && yo<thresholded.rows && xo<thresholded.cols){
              accumulator[yo][xo][r] += 1;
            }
            int xon = x - r*std::cos(dir.at<float>(y,x));
            int yon = y - r*std::sin(dir.at<float>(y,x));
            if(xon>=0 &&yon>=0 && xon<thresholded.cols && yon<thresholded.rows){
              accumulator[yon][xon][r] += 1;
            }
          }
        }
      }
    }

    Mat convToMat(thresholded.rows,thresholded.cols, CV_32FC1, Scalar(0));
    for(int i=0;i<thresholded.rows;i++){
      for(int j=0;j<thresholded.cols;j++){
          convToMat.at<float>(i,j) = 0;
        for(int r=0;r<100;r++){
          convToMat.at<float>(i,j) += accumulator[i][j][r];
        }
      }
    }
    cv::normalize(convToMat,convToMat,0,255,NORM_MINMAX);
    imwrite("houghspace.jpg",convToMat);
}

int main(int argc, char** argv) {
  Mat input = imread("coins1.png", 0);
  Mat unnorm = sobel(input);

  Mat direction = imread("direction.jpg", 0);
  Mat magnitude = imread("magnitude.jpg", 0);
  threshold(magnitude, 65);

  Mat thresholded = imread("thresholded.jpg", 0);
  hough(thresholded, unnorm);

  return 0;
}
