#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/include/opencv2/:/usr/include/opencv
export OPENCV_CFLAGS=-I/usr/include/opencv2/
export O_LIBS="-L/usr/lib64/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann"
