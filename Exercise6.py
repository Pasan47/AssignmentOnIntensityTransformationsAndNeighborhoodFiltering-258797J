import numpy as np

def gaussian_derivative_kernels(size=5, sigma=2.0):
    # Create a coordinate grid centered at (0,0)
    k = (size - 1) / 2
    y, x = np.ogrid[-k:k+1, -k:k+1]
    
    # Compute the base Gaussian distribution
    # Normalization constant is omitted here as we normalize at the end
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Compute derivatives
    kernel_x = -(x / sigma**2) * gaussian
    kernel_y = -(y / sigma**2) * gaussian
    
    # Normalize: Usually, derivative kernels are normalized such that 
    # their response to a unit ramp is 1. 
    # A common approach is to divide by the sum of (x * kernel_x)
    kernel_x /= np.sum(x * kernel_x)
    kernel_y /= np.sum(y * kernel_y)
    
    return kernel_x, kernel_y

# Generate kernels
gx, gy = gaussian_derivative_kernels()

print("5x5 Gaussian Derivative Kernel (X-direction):\n", gx.round(4))
print("\n5x5 Gaussian Derivative Kernel (Y-direction):\n", gy.round(4))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dog_kernel(size=51, sigma=5.0):
    # Create coordinate grid
    k = (size - 1) / 2
    x_range = np.linspace(-k, k, size)
    y_range = np.linspace(-k, k, size)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Compute the Derivative-of-Gaussian in x-direction
    # G_x = - (x / sigma^2) * exp(-(x^2 + y^2) / (2 * sigma^2))
    gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = -(X / sigma**2) * gaussian
    
    # Setup 3D Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, kernel, cmap='coolwarm', edgecolor='none')
    
    ax.set_title(f'51x51 Derivative-of-Gaussian Kernel ($\sigma={sigma}$)')
    ax.set_xlabel('X direction')
    ax.set_ylabel('Y direction')
    ax.set_zlabel('Weight')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    plt.savefig('dog_kernel_3d.png')

plot_dog_kernel()