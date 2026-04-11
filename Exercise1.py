import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# Load your image
image = cv2.imread('./runway.png')

# Apply different gamma values
# Gamma > 1.0 will brighten the image
# Gamma < 1.0 will darken the image
lightened = adjust_gamma(image, gamma=2.2)
darkened = adjust_gamma(image, gamma=0.5)

# Display results
#cv2.imshow("Original", image)
#cv2.imshow("Gamma 2.2 (Lightened)", lightened)
#cv2.imshow("Gamma 0.5 (Darkened)", darkened)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#contrast stretching function
def contrast_stretch(image, r1, r2):
    # Ensure image is float for precision
    img_float = image.astype(float) / 255.0
    
    # If r1 and r2 are already 0-1, do not divide by 255
    # Adding a tiny epsilon (1e-5) prevents division by zero
    diff = r2 - r1
    if diff <= 0:
        return image # Return original if range is invalid
        
    output = (img_float - r1) / diff
    
    # Clip to keep values in [0, 1] range
    output = np.clip(output, 0, 1)
    
    # Convert back to 0-255 uint8
    return (output * 255).astype(np.uint8)

# Assuming 'image' is already loaded via cv2.imread()
# Here we stretch everything between 20% and 80% brightness to the full 0-100% range
contrasted_image = contrast_stretch(image, 0.2, 0.8)

cv2.imshow("Contrast Stretched", contrasted_image)
cv2.waitKey(0) # Added to keep the window open
cv2.destroyAllWindows()