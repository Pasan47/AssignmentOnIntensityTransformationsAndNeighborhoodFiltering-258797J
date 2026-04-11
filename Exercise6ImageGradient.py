import numpy as np
from scipy.signal import convolve2d
import cv2

# Load the image directly in grayscale mode


def compute_image_gradients(image, sigma=1.0, kernel_size=5):
    """
    Applies Derivative-of-Gaussian kernels to a grayscale image.
    
    Args:
        image: 2D numpy array (grayscale image)
        sigma: Standard deviation for the Gaussian
        kernel_size: Size of the square kernel (should be odd)
        
    Returns:
        Ix: Gradient in the x-direction (horizontal)
        Iy: Gradient in the y-direction (vertical)
    """
    # 1. Create the coordinate grid for the kernel
    k = (kernel_size - 1) / 2
    ax = np.linspace(-k, k, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    
    # 2. Generate the kernels based on the partial derivative formulas
    # G = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
    g_base = (1 / (2 * np.pi * sigma**2)) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    kernel_x = -(xx / sigma**2) * g_base
    kernel_y = -(yy / sigma**2) * g_base
    
    # 3. Convolve the image with the kernels
    # 'boundary=symm' handles edge pixels by mirroring them
    # 'mode=same' ensures the output image is the same size as the input
    Ix = convolve2d(image, kernel_x, mode='same', boundary='symm')
    Iy = convolve2d(image, kernel_y, mode='same', boundary='symm')
    
    return Ix, Iy

# --- Execution Example ---
# Assuming 'gray_img' is a loaded grayscale image
gray_image = cv2.imread('im03small.png', cv2.IMREAD_GRAYSCALE)
Ix, Iy = compute_image_gradients(gray_image, sigma=1.5)
print("Gradient in X direction (Ix):\n", Ix)
print("\nGradient in Y direction (Iy):\n", Iy)