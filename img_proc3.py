import cv2
import numpy as np
import os

# Read image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "photos/cat1.png")

img=cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to binary
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Take border width from user
width = int(input("Enter border width (pixels): "))

# Take border color from user
print("Enter border color (B G R values, 0-255):")
B = int(input("B: "))
G = int(input("G: "))
R = int(input("R: "))

border_color = (B, G, R)

# Add border to original image
color_border = cv2.copyMakeBorder(
    img,
    width, width, width, width,
    cv2.BORDER_CONSTANT,
    value=border_color
)

# For grayscale, border needs a single intensity value
gray_color = int((R + G + B) / 3)

gray_border = cv2.copyMakeBorder(
    gray,
    width, width, width, width,
    cv2.BORDER_CONSTANT,
    value=gray_color
)

# For binary image, border can only be black (0) or white (255)
binary_color = 255 if (R + G + B) / 3 >= 128 else 0

binary_border = cv2.copyMakeBorder(
    binary,
    width, width, width, width,
    cv2.BORDER_CONSTANT,
    value=binary_color
)

# Display
cv2.imshow("Original with Border", color_border)
cv2.imshow("Grayscale with Border", gray_border)
cv2.imshow("Binary with Border", binary_border)

cv2.waitKey(0)


