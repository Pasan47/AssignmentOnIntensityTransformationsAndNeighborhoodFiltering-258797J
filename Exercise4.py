import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
img = cv2.imread('photo-1653775545518-3679836a8933.avif', cv2.IMREAD_GRAYSCALE)

# Apply Otsu's thresholding
# ret is the computed threshold value, mask is the binary image
ret, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu's Threshold Value: {ret}")




def masked_histogram_equalization(image_path):
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Generate Otsu mask (Inverse because foreground is dark)
    # We want the foreground to be the 'True' part of the mask
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Calculate histogram only for the masked region
    hist, bins = np.histogram(img[mask > 0].flatten(), 256, [0, 256])
    
    # 4. Calculate CDF for the foreground
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # 5. Apply the equalization only to foreground pixels
    img_equalized = img.copy()
    img_equalized[mask > 0] = cdf_final[img[mask > 0]]
    
    return img, mask, img_equalized

# --- Execution ---
original, fg_mask, result = masked_histogram_equalization('photo-1653775545518-3679836a8933.avif')

# --- Visualization ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original (Silhouette)")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Foreground Mask (Otsu)")
plt.imshow(fg_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Foreground-Only Equalized")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()