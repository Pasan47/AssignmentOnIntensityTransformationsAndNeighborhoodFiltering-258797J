import cv2
import numpy as np

def zoom_image(image, s, method='bilinear'):
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    new_h, new_w = int(h * s), int(w * s)
    
    # Create an empty output image
    # Note: Handles both grayscale and color (3 channels)
    if len(image.shape) == 3:
        output = np.zeros((new_h, new_w, image.shape[2]), dtype=np.uint8)
    else:
        output = np.zeros((new_h, new_w), dtype=np.uint8)

    # Pre-calculate inverse scale to map output -> input
    inv_s = 1.0 / s

    for i in range(new_h):
        for j in range(new_w):
            # Map back to source coordinates (floating point)
            src_x = j * inv_s
            src_y = i * inv_s

            if method == 'nearest':
                # (a) Nearest-Neighbor: Round to closest integer
                orig_x = min(int(round(src_x)), w - 1)
                orig_y = min(int(round(src_y)), h - 1)
                output[i, j] = image[orig_y, orig_x]

            elif method == 'bilinear':
                # (b) Bilinear: Weighted average of 4 neighbors
                # Find the 4 neighboring coordinates
                x1, y1 = int(np.floor(src_x)), int(np.floor(src_y))
                x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

                # Calculate distances (weights)
                dx, dy = src_x - x1, src_y - y1

                # Bilinear Formula: 
                # f(x,y) ≈ f(0,0)(1-dx)(1-dy) + f(1,0)dx(1-dy) + f(0,1)(1-dx)dy + f(1,1)dxdy
                val = (image[y1, x1] * (1 - dx) * (1 - dy) +
                       image[y1, x2] * dx * (1 - dy) +
                       image[y2, x1] * (1 - dx) * dy +
                       image[y2, x2] * dx * dy)
                
                output[i, j] = val.astype(np.uint8)

    return output

# --- Example Usage ---
img = cv2.imread('./im01small.png')
scale_factor = 2.5  # Zoom in by 2.5x

# Apply both methods
nn_zoom = zoom_image(img, scale_factor, method='nearest')
bl_zoom = zoom_image(img, scale_factor, method='bilinear')

cv2.imshow('Nearest Neighbor Zoom', nn_zoom)
cv2.imshow('Bilinear Zoom', bl_zoom)
cv2.waitKey(0)


def compute_normalized_ssd(original, zoomed):
    # Ensure images are the same size
    if original.shape != zoomed.shape:
        zoomed = cv2.resize(zoomed, (original.shape[1], original.shape[0]))
    
    # Convert to float for calculation to avoid overflow/underflow
    orig_f = original.astype(np.float32)
    zoom_f = zoomed.astype(np.float32)
    
    # Calculate Squared Difference
    squared_diff = (orig_f - zoom_f) ** 2
    
    # Sum and Normalize
    sum_sq_diff = np.sum(squared_diff)
    num_pixels = original.shape[0] * original.shape[1]
    if len(original.shape) == 3:
        num_pixels *= original.shape[2]
        
    normalized_ssd = sum_sq_diff / (num_pixels * (255**2))
    
    return normalized_ssd

# --- Execution ---
# 1. Start with original
original_img = cv2.imread('./runway.png')

# 2. Shrink it (Downsample)
small_img = zoom_image(original_img, 0.5, method='bilinear')

# 3. Zoom it back up (Upsample)
upsampled_nn = zoom_image(small_img, 2.0, method='nearest')
upsampled_bl = zoom_image(small_img, 2.0, method='bilinear')

# 4. Compare
ssd_nn = compute_normalized_ssd(original_img, upsampled_nn)
ssd_bl = compute_normalized_ssd(original_img, upsampled_bl)

print(f"Normalized SSD (Nearest Neighbor): {ssd_nn:.6f}")
print(f"Normalized SSD (Bilinear): {ssd_bl:.6f}")