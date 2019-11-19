#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

Mat sobel (Mat input/*, Mat dx, Mat dy, Mat magnitude, Mat direction*/) {

  Mat dx(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat dy(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat magnitude(input.rows, input.cols, CV_32FC1, Scalar(0));
  Mat direction(input.rows, input.cols, CV_32FC1, Scalar(0));

  Mat kernelx = (Mat_<int>(3,3) << -1,0,1,-2,0,2,-1,0,1);
  Mat kernely = kernelx.t();

  for (int x = 1; x < input.rows - 1; x++) {
    for (int y = 1; y < input.cols - 1; y++) {

      float pixelx = 0;
      float pixely = 0;

      for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {

          pixelx += input.at<uchar>(x+i, y+j) * kernelx.at<int>(i+1, j+1);
          pixely += input.at<uchar>(x+i, y+j) * kernely.at<int>(i+1, j+1);

        }
      }

      dx.at<float>(x,y) = pixelx;
      dy.at<float>(x,y) = pixely;
      magnitude.at<float>(x, y) = sqrt((pixelx * pixelx) + (pixely * pixely));
      direction.at<float>(x, y) = atan2(pixely,pixelx);
    }
  }

  normalize(dx, dx, 0, 255, NORM_MINMAX);
  normalize(dy, dy, 0, 255, NORM_MINMAX);
  normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
  // normalize(direction, direction, 0 , 255, NORM_MINMAX);

  imwrite("dx.jpg", dx);
  imwrite("dy.jpg", dy);
  imwrite("magnitude.jpg", magnitude);

  return direction;
}

void threshold(Mat input, int val) {

  Mat output(input.rows, input.cols, CV_8UC1, Scalar(0));

  for (int i = 0; i < input.rows; i++) {
    for (int j = 0; j < input.cols; j++) {

      if (input.at<uchar>(i, j) > val) {
        output.at<uchar>(i, j) = 255;
      }
      else {
        output.at<uchar>(i, j) = 0;
      }
    }
  }
  imwrite("threshold.jpg", output);
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

void hough(Mat threshold, Mat direction, int radius) {

  Mat output(threshold.rows, threshold.cols, CV_32FC1, Scalar(0));
  // int accumulator[threshold.rows][threshold.cols][100];
  int ***accumulator = malloc3dArray(threshold.rows,threshold.cols,radius);
  for (int i = 0; i < threshold.rows; i++) {
    for (int j = 0; j < threshold.cols; j++) {
      for (int r = 0; r < radius; r++) {
        accumulator[i][j][r] = 0;
      }
    }
  }

  for (int x = 0; x < threshold.rows; x++) {
    for (int y = 0; y < threshold.cols; y++) {
      if (threshold.at<uchar>(x, y) == 255) {
        for (int r = 5; r <= radius; r++) {

          int xo = x + r * sin(direction.at<float>(x, y));
          int yo = y + r * cos(direction.at<float>(x, y));

          if (xo >= 0 && yo >= 0 && xo < threshold.rows && yo < threshold.cols) {
            output.at<float>(xo, yo) += 1;
            accumulator[xo][yo][r] += 1;
          }

          int xon = x - r * sin(direction.at<float>(x, y));
          int yon = y - r * cos(direction.at<float>(x, y));

          if (xon >= 0 && yon >= 0 && xon < threshold.rows && yon < threshold.cols) {
            output.at<float>(xon, yon) += 1;
            accumulator[xon][yon][r] += 1;
          }

        }
      }
    }
  }

  normalize(output, output, 0, 255, NORM_MINMAX);

  for (int i = 0; i < output.rows; i++) {
    for (int j = 0; j < output.cols; j++) {
      if (output.at<float>(i, j) == 255) {
        int maxRadius = 0;
        int temp = 0;
        for (int r = 0; r < radius; r++) {
          if (accumulator[i][j][r] > temp) {
            temp = accumulator[i][j][r];
            maxRadius = r;
          }
        }
        circle(output, Point(j, i), maxRadius, Scalar(0, 0, 255), 2);
      }
    }
  }

  imwrite("houghspace.jpg", output);

}


int main() {

  Mat input = imread("dart0.jpg", 0);
  Mat direction = sobel(input);

  Mat magnitude = imread("magnitude.jpg", 0);
  threshold(magnitude, 50);

  Mat threshold = imread("threshold.jpg", 0);
  hough(threshold, direction, 150);

  return 0;
}
