import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def generate_gaussian_kernel(size=5, sigma=2.0):
    # Create 1D coordinate vector centered at 0
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    # Calculate the squared distances
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    # Create 2D kernel using outer product
    kernel = np.outer(gauss, gauss)
    # Normalize the kernel so the sum is 1.0
    return kernel / kernel.sum()

kernel_5x5 = generate_gaussian_kernel(5, 2)
print(kernel_5x5)

# visualize 51*51 kernel as 3D surface plot



def generate_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / kernel.sum()

size = 51
sigma = 10
kernel = generate_gaussian_kernel(size, sigma)

# Create meshgrid for plotting
x = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
y = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, kernel, cmap='viridis', edgecolor='none')

ax.set_title(f'3D Surface Plot of a {size}x{size} Gaussian Kernel ($\sigma={sigma}$)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Weight')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

plt.savefig('gaussian_surface.png')

# Apply gaussian smoothing to an image
from scipy.signal import convolve2d
img = cv2.imread('./runway.png', cv2.IMREAD_GRAYSCALE)
smoothed_manual = convolve2d(img, kernel_5x5, mode='same', 
                             boundary='fill', fillvalue=0)
cv2.imwrite('scipy_gaussian_blur.jpg', smoothed_manual)

# do gaussian smoothing using OpenCV's built-in function for comparison
smoothed_opencv = cv2.GaussianBlur(img, (5, 5), sigmaX=2)
cv2.imwrite('opencv_gaussian_blur.jpg', smoothed_opencv)