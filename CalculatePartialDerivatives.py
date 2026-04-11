import sympy as sp

def get_symbolic_gaussian_derivatives():
    # Define symbols
    x, y, sigma = sp.symbols('x y sigma')
    
    # Define the 2D Gaussian function G(x, y)
    # G(x, y) = (1 / (2 * pi * sigma**2)) * exp(-(x**2 + y**2) / (2 * sigma**2))
    g_xy = (1 / (2 * sp.pi * sigma**2)) * sp.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Compute partial derivatives
    dg_dx = sp.diff(g_xy, x)
    dg_dy = sp.diff(g_xy, y)
    
    return x, y, sigma, g_xy, dg_dx, dg_dy

# --- Execution ---
x, y, sigma, g, dx, dy = get_symbolic_gaussian_derivatives()

print("Original Gaussian G(x, y):")
sp.pprint(g)

print("\nPartial Derivative with respect to x (∂G/∂x):")
sp.pprint(sp.simplify(dx))

print("\nPartial Derivative with respect to y (∂G/∂y):")
sp.pprint(sp.simplify(dy))