import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the corrupted image (Fig. 4)
img = cv2.imread('image-corrupted-with-salt-and-pepper-noise.png')

# 2. Apply Gaussian Smoothing (Linear)
# ksize (5,5) and sigma 1.5
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1.5)

# 3. Apply Median Filtering (Non-Linear)
# ksize 5 (must be an odd integer)
median_blur = cv2.medianBlur(img, 5)

# --- Visualization ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original (Salt & Pepper Noise)")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gaussian Smoothing (Blurred Noise)")
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Median Filtering (Noise Removed)")
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()