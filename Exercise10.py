import numpy as np
import cv2

def manual_bilateral_filter(image, d, sigma_s, sigma_r):
    radius = d // 2
    padded_img = np.pad(image, radius, mode='edge').astype(np.float32)
    output = np.zeros_like(image, dtype=np.float32)
    x, y = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i+d, j:j+d]
            center_intensity = padded_img[i + radius, j + radius]
            intensity_diff = window - center_intensity
            range_weights = np.exp(-(intensity_diff**2) / (2 * sigma_r**2))
            combined_weights = spatial_weights * range_weights
            normalization_factor = np.sum(combined_weights)
            output[i, j] = np.sum(combined_weights * window) / normalization_factor
    return output.astype(np.uint8)

# --- Test the function ---
img = cv2.imread('./runway.png', cv2.IMREAD_GRAYSCALE)
# Use d=9, sigma_s=75, sigma_r=75 as common test parameters
result = manual_bilateral_filter(img, 9, 75, 75)



# 1. Load the image in grayscale (or BGR)
img = cv2.imread('./runway.png', cv2.IMREAD_GRAYSCALE)

# 2. Apply Gaussian Blur
# ksize: (5, 5) - The width and height of the kernel (must be odd)
# sigmaX: 2.0 - The standard deviation in the X direction
# sigmaY: 0 - If 0, it is automatically set to equal sigmaX
smoothed_opencv = cv2.GaussianBlur(img, (5, 5), sigmaX=2.0)

# 3. Save or display the result
cv2.imwrite('opencv_gaussian_result.jpg', smoothed_opencv)

# Optional: To see the difference
#cv2.imshow('Original', img)
#cv2.imshow('Gaussian Blur', smoothed_opencv)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 1. Load the image
img = cv2.imread('./media_18aa9b7278c703fc6f0baac75e8e7aa84c75f09d9.webp')

# 2. Apply Bilateral Filter
# d: Diameter of each pixel neighborhood (use 5 for real-time, 9 for offline)
# sigmaColor: Filter sigma in the color space (higher = more colors mixed)
# sigmaSpace: Filter sigma in the coordinate space (higher = further pixels affect each other)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# 3. Save or display
cv2.imwrite('bilateral_result.jpg', bilateral)

#4
# 1. Load the image and convert to grayscale
img = cv2.imread('runway.png', cv2.IMREAD_GRAYSCALE)

# 2. Parameters for the filter
# d=9: Neighborhood diameter
# sigma_s=75: Spatial standard deviation (Distance)
# sigma_r=75: Range standard deviation (Intensity difference)
d = 9
sigma_s = 75.0
sigma_r = 75.0

# 3. Call your manual function
# Note: Ensure the function 'manual_bilateral_filter' is defined in your script
filtered_image = manual_bilateral_filter(img, d, sigma_s, sigma_r)

# 4. Compare results
# You can compare this with OpenCV's built-in version to verify accuracy
cv_filtered = cv2.bilateralFilter(img, d, sigma_r, sigma_s)

# Calculate difference to check implementation accuracy
diff = cv2.absdiff(filtered_image, cv_filtered)

print(f"Mean absolute difference between Manual and OpenCV: {np.mean(diff)}")

cv2.imshow('Manual Bilateral', filtered_image)
cv2.imshow('OpenCV Bilateral', cv_filtered)
cv2.waitKey(0)