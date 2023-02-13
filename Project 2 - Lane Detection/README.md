# Project 2 - Lane Detection

## Project Description

This project is aimed at detecting lanes in a video stream of a driving scenario. The algorithm processes each frame of the video, identifies the lane markings and overlay the markings onto the original frame to highlight the detected lanes.

The lane detection algorithm makes use of computer vision techniques such as color thresholding, edge detection, and Hough line transforms to identify the lane markings in each frame. The algorithm first applies color thresholding to the image to extract pixels that correspond to the lane markings. Then, the edge detection algorithm is used to identify the edges in the thresholded image. Finally, the Hough line transform is used to identify the lines in the edge-detected image and fit these lines to the lane markings.

## Running Instructions

Navigate to project directory and put the required videos and images in it.

Run codes in the 'code' folder individually.

Provide input or output paths if required. Default paths within project directory are set as

1. For problem 1 - adaptive_hist_data/<images>.png
2. For problem 2 - input-'whiteline.mp4' ;output - 'Output/problem2_op.mp4'
3. For problem 3 - input-'challenge.mp4' ;output - 'Output/problem3_op.mp4'
  
## Output Videos
  
### 1. Histogram Equalization

A video created by combining the histogram equalized frames. Clockwise from top left we have:

1. The original frame
2. Traditional Histogram Equalization
3. Contrast Limited Adaptive Histogram Equalization
4. Adaptive Histogram Equalization

[![histogram](https://user-images.githubusercontent.com/35636842/218379612-6e042bc0-b92b-4b8e-9460-f5fd00923c7b.gif)](https://youtu.be/3OaGxncNj_I)

### 2. Lane stripe detection
[![Lane stripe detection](https://img.youtube.com/vi/t7pD86ErhBU/0.jpg)](https://youtu.be/t7pD86ErhBU)
  
### 3. Warped Lanes
[![Warped Lanes](https://img.youtube.com/vi/qWIDrhfOeQ4/0.jpg)](https://youtu.be/qWIDrhfOeQ4)

### 4. Sliding Window Output
[![Sliding Window Output](https://img.youtube.com/vi/SXZhIh9PfuU/0.jpg)](https://youtu.be/SXZhIh9PfuU)
  
### 5. Lane Detection
[![Lane Detection](https://img.youtube.com/vi/G_Rur_1il5k/0.jpg)](https://youtu.be/G_Rur_1il5k)
