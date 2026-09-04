#averaging filter: spatial filtering technique that replaces each pixel's value with the mathematical average of its neighboring pixels within a defined window
#weighted filter: given by h(x,y)= max(|x|.|y|) 
#gaussian filter

import os
import cv2
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "photos/cat1.png")

# Read image
img = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)

# Check image
if img is None:
    print("image was not found")
    exit()

#averaging

#input filter size
k = int(input("Enter filter size (odd number): "))
pad = k // 2 #to avoid shrinkage

# add zero padding around image
padded = np.pad(img, pad, mode='constant')
smooth = np.zeros_like(img) 

# Apply averaging filter manually
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        #kernel window size
        window = padded[i:i+k, j:j+k]
        #unweighted average
        smooth[i, j] = np.mean(window)

cv2.imshow("average smoothed image", smooth)


# Input padding method
print("1.zero Padding")
print("2.replication")
p = int(input("enter choice: "))

if p == 1:
    padded = np.pad(img, pad, mode='constant', constant_values=0)
else:
    padded = np.pad(img, pad, mode='edge')

# weighted kernel window
kernel = np.zeros((k, k), dtype=float)
#[(-1,-1) 0 1]
#
center = k // 2
for i in range(k): 
    for j in range(k):
        x = i - center
        y = j - center

        kernel[i, j] = max(abs(x), abs(y))

# Normalize kernel
kernel = kernel / np.sum(kernel)

print("Weighted Kernel:")
print(kernel)

# Output
gaussian = np.zeros_like(img, dtype=float)

# Apply filter
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        window = padded[i:i+k, j:j+k]
        gaussian[i, j] = np.sum(window * kernel)

gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)


# Display images
cv2.imshow("Original Image", img)
cv2.imshow("Average Smoothed Image", smooth)
cv2.imshow("Gaussian Smoothed image ", gaussian)


cv2.waitKey(0)
cv2.destroyAllWindows()
