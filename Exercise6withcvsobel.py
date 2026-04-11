import cv2
import numpy as np

def compute_sobel_gradients(image_path):
    # 1. Load the image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Compute Sobel gradients
    # ksize=3 uses the standard 3x3 Sobel kernel
    # cv2.CV_64F is used to capture negative gradients (Black-to-White)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Convert back to absolute 8-bit for visualization
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    return abs_sobelx, abs_sobely

# Execution
#gray_image = cv2.imread('im03small.png', cv2.IMREAD_GRAYSCALE)
Ix, Iy = compute_sobel_gradients('im03small.png')
print("Sobel Gradient in X direction (Ix):\n", Ix)
print("\nSobel Gradient in Y direction (Iy):\n", Iy)