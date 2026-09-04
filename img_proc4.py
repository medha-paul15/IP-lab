import cv2 
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "photos/cat1.png")

# Read image
img = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)

# Check image
if img is None:
    print("image was not found")
    exit()

# Convert to binary
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Input border width
w = int(input("border width: "))

# Input border color
color = int(input("border color (0-255): "))

# ---------------- GRAYSCALE IMAGE ----------------

rows, cols = img.shape

# Create larger image filled with border color
gray_padded = np.full((rows + 2*w, cols + 2*w), color, dtype=np.uint8)

# Copy original image into center
for i in range(rows):
    for j in range(cols):
        gray_padded[i+w][j+w] = img[i][j]


# ---------------- BINARY IMAGE ----------------

rows, cols = binary.shape

# Create larger binary image filled with border color
binary_padded = np.full((rows + 2*w, cols + 2*w), color, dtype=np.uint8)

# Copy original binary image into center
for i in range(rows):
    for j in range(cols):
        binary_padded[i+w][j+w] = binary[i][j]


# Display
cv2.imshow("Original Grayscale", img)
cv2.imshow("Padded Grayscale", gray_padded)

cv2.imshow("Original Binary", binary)
cv2.imshow("Padded Binary", binary_padded)

cv2.waitKey(0)
cv2.destroyAllWindows()