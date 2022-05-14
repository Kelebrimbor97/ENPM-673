import cv2
import numpy as np
import glob
 
img_array = []
file_list = sorted(glob.glob('Output/*.png'))
for filename in file_list:
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('Output/histogram_combined.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

for i in range(11):
    for i in range(len(img_array)):
        out.write(img_array[i])
out.release()