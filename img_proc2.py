import cv2 
import numpy as np
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "photos/cat1.png")

img=cv2.imread(image_path)
'''
#part 1
rows, cols, channels = img.shape

gray = np.zeros((rows, cols), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        B, G, R = img[i, j]

        mean = (int(R) + int(G) + int(B)) / 3

        gray[i, j] = int(mean)

cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray) 

cv2.waitKey(0) 
'''

#part 2
w_red=float(input("enter weight of red: "))
w_blue=float(input("enter weight of blue: "))
w_green=float(input("enter weight of green: "))

if (0 <= w_red <= 1 and
    0 <= w_blue<= 1 and
    0 <= w_green <= 1 and
    abs(w_red + w_green +w_blue) ==1):
    B,G,R=cv2.split(img)

    gray=(w_red*R + w_blue*B + w_green*G).astype(np.uint8)
    cv2.imshow("original image", img)
    cv2.imshow("greyscale image",gray)
    cv2.waitKey(0)

else:
    print("Invalid weights")








