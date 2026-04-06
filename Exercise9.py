import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image(image_path, sigma=1.0, amount=1.5):
    # 1. Load image (Grayscale for the runway)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "Error: Image not found."

    # 2. Create the blurred version (Low-pass)
    # We use GaussianBlur to create a smooth version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # 3. Calculate the "Mask" (Original - Blurred)
    # This isolates the edges and fine details
    mask = cv2.addWeighted(img, 1.0, blurred, -1.0, 0)

    # 4. Add the mask back to the original (Original + Amount * Mask)
    # amount > 1.0 increases sharpness; < 1.0 is a subtle correction
    sharpened = cv2.addWeighted(img, 1.0, mask, amount, 0)

    # Ensure pixel values are within [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return img, mask, sharpened

# --- Execution ---
original, edge_mask, sharp_result = sharpen_image('./shells.tif', sigma=1.5, amount=2.0)

# --- Visualization ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original (Soft)")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Edges (The Mask)")
plt.imshow(edge_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sharpened Result")
plt.imshow(sharp_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()