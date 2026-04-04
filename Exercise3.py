import cv2
import numpy as np
import matplotlib.pyplot as plt

def custom_histogram_equalization(image):
    # Ensure the image is grayscale for the calculation
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Get image dimensions
    height, width = gray.shape
    total_pixels = height * width

    # 2. Calculate the histogram (frequency of each intensity 0-255)
    hist = np.zeros(256)
    for i in range(height):
        for j in range(width):
            hist[gray[i, j]] += 1

    # 3. Calculate Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()
    
    # 4. Normalize the CDF
    # Formula: h(v) = round((cdf(v) - cdf_min) / (total_pixels - cdf_min) * 255)
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = (cdf - cdf_min) * 255 / (total_pixels - cdf_min)
    cdf_normalized = np.ma.filled(cdf_normalized, 0).astype('uint8')

    # 5. Map the original pixels to the new equalized values
    equalized_img = cdf_normalized[gray]

    return gray, equalized_img

# --- Execution ---
# Load the runway image
img_path = './runway.png'
print(f"Processing image: {cv2.imread(img_path).shape}")
original_gray, equalized = custom_histogram_equalization(cv2.imread(img_path))


# --- Visualization ---
plt.figure(figsize=(14, 7))

# Original
plt.subplot(2, 2, 1)
plt.title("Original Low Contrast Image")
plt.imshow(original_gray, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Original Histogram (Clustered)")
plt.hist(original_gray.ravel(), 256, [0, 256], color='gray')

# Equalized
plt.subplot(2, 2, 2)
plt.title("After Histogram Equalization")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Equalized Histogram (Flattened/Stretched)")
plt.hist(equalized.ravel(), 256, [0, 256], color='blue')

plt.tight_layout()
plt.show()