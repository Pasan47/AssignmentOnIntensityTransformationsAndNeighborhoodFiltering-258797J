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
cv2.imshow("Original", image)
cv2.imshow("Gamma 2.2 (Lightened)", lightened)
cv2.imshow("Gamma 0.5 (Darkened)", darkened)
cv2.waitKey(0)
cv2.destroyAllWindows()

#contrast stretching function
def contrast_stretch(image, r1, r2):
    # Normalize image to 0-1 for the math
    img_float = image.astype(float) / 255.0
    
    # Scale r1 and r2 to 0-1 if they are provided in 0-255 range
    r1_norm = r1 / 255.0
    r2_norm = r2 / 255.0

    # Apply the piecewise function
    # np.clip handles the r < r1 and r > r2 cases automatically after the linear shift
    output = (img_float - r1_norm) / (r2_norm - r1_norm)
    output = np.clip(output, 0, 1)
    
    # Convert back to 0-255 uint8
    return (output * 255).astype(np.uint8)