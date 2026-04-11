import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
# Function to plot histograms for BGR channels
def plot_histogram(image, title, subplot_pos):
    plt.subplot(2, 2, subplot_pos)
    plt.title(title)
    colors = ('b', 'g', 'r')
    for i, col in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

# 1. Check if file exists to avoid silent NoneType errors
file_path = 'media_18aa9b7278c703fc6f0baac75e8e7aa84c75f09d9.webp'

if not os.path.exists(file_path):
    print(f"Error: The file {file_path} was not found.")
else:
    img = cv2.imread(file_path)

    if img is None:
        print("Error: Could not decode the image. Check if the file is corrupted.")
    else:
        # 2. Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # 3. Gamma correction on L plane (gamma = 0.45)
        gamma = 0.45
        # Use a lookup table (LUT) for better performance and reliability
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        
        l_corrected = cv2.LUT(l, table)

        # 4. Merge and convert back
        corrected_lab = cv2.merge((l_corrected, a, b))
        result = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

        plt.figure(figsize=(12, 8))

        # Show Original Image
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        # Show Corrected Image
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Gamma Corrected (γ={gamma})")
        plt.axis('off')

        # Plot Histograms
        plot_histogram(img, "Original Histogram (BGR)", 3)
        plot_histogram(result, "Corrected Histogram (BGR)", 4)

        plt.tight_layout()
        plt.show()

        # 5. Save the result
        success = cv2.imwrite('gamma_corrected_result.jpg', result)
        
        if success:
            print("Image saved successfully as 'gamma_corrected_result.jpg'")
        else:
            print("Error: Could not write the image to disk.")