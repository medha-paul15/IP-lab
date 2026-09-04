import cv2 as cv
import numpy as np

import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "photos/cat1.png")

img=cv.imread(image_path)
  
cv.imshow('normal',img)

# #1. convert to greyscale 
#gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

rows,cols,_=img.shape
gray=np.zeros((rows,cols),dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        b,g,r=img[i][j][0],img[i][j][1],img[i][j][2]
        grayscale = (0.299* r) + (0.587*g) + (0.114 * b)
        gray[i][j]=grayscale



cv.imshow('grayed cat', gray)

mean = np.mean(gray)
print("Mean intensity:", mean)

binary=np.zeros((rows,cols),dtype=np.uint8)


for i in range(rows):
    for j in range(cols):
        if gray[i][j]>=mean:
            binary[i][j]=255
        else:
            binary[i][j]=0


#_, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
cv.imshow("Binary cat", binary)

cv.waitKey(0)

