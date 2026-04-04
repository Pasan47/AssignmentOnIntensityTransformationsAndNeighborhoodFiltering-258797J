import cv2
import numpy as np

# Load the image
img = cv2.imread('media_18aa9b7278c703fc6f0baac75e8e7aa84c75f09d9.webp')

# Convert to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Gamma correction on L plane (gamma = 0.45)
# Formula: output = input^(gamma)
gamma = 0.45
l_norm = l / 255.0
l_corrected = np.uint8(np.power(l_norm, gamma) * 255)

# Merge and convert back
corrected_lab = cv2.merge((l_corrected, a, b))
result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

cv2.imwrite('gamma_corrected_result.jpg', result)