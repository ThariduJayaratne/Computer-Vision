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

void hough(){
  Mat thresholded = threshold(65);
  Mat mag,dir = sobel();
  int ***accumulator = malloc3dArray(thresholded.rows,thresholded.cols,100);
  for(int y=0;y<thresholded.rows;y++){
    for(int x =0;x<thresholded.cols;x++){
      for(int r =20;r<=100;r++){
        if(thresholded.at<uchar>(i,j)==255){
            int xo = x + r*std::cos(dir.at<float>(y,x));
            int yo = y + r*std::sin(dir.at<float>(y,x));
          if(xo>=0 &&yo>=0 && xo<=thresholded.cols && yo<=thresholded.rows){
            accumulator[xo][yo][r] += 1;
          }
        }
      }

    }
  }
}



int main(int argc, char** argv) {
  // char* imageName = argv[1];
  // Mat image = imread(imageName,1);
  imwrite("thresholded.jpg",threshold(65));


  return 0;
}
