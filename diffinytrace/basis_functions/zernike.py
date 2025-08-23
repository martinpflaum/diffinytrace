"""
Zernike polynomial basis functions for optical wavefront representation.

This module provides functions to compute Zernike polynomials, which are commonly used
in optics for describing wavefront aberrations over a circular aperture. The polynomials
are orthogonal over the unit disk and are indexed by radial order (n) and azimuthal 
frequency (m).

.. figure:: _static/zernike_plot1.png
   :alt: Zernike polynomials visualization
   :width: 60%
   :align: center

   Visualization of Zernike polynomials organized by radial order (rows) and azimuthal frequency (columns).


Example:
    Basic usage for computing and visualizing Zernike polynomials:

    >>> import torch
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import diffinytrace.basis_functions.zernike as zernike
    >>> 
    >>> # Create unit circle grid
    >>> grid_size = 256
    >>> x = torch.linspace(-1, 1, grid_size)
    >>> y = torch.linspace(-1, 1, grid_size)
    >>> X, Y = torch.meshgrid(x, y, indexing='ij')
    >>> 
    >>> # Create mask for unit circle
    >>> mask = (X**2 + Y**2) <= 1.0
    >>> x_points = X[mask]
    >>> y_points = Y[mask]
    >>> points = torch.stack([x_points, y_points], dim=1)
    >>> 
    >>> # Evaluate Zernike polynomials
    >>> max_n = 6  # Maximum radial degree
    >>> basis_values = zernike.basis_function(max_n, points)
    >>> 
    >>> # Group basis functions by radial degree
    >>> basis_by_degree = {}
    >>> for basis_idx in range(basis_values.shape[1]):
    ...     radial_degree = zernike.get_radial_order(basis_idx)
    ...     if radial_degree not in basis_by_degree:
    ...         basis_by_degree[radial_degree] = []
    ...     basis_by_degree[radial_degree].append(basis_idx)
    >>> 
    >>> # Visualize the polynomials
    >>> max_cols = max(len(indices) for indices in basis_by_degree.values())
    >>> num_rows = len(basis_by_degree)
    >>> fig, axes = plt.subplots(num_rows, max_cols, figsize=(3*max_cols, 3*num_rows))
    >>> 
    >>> for row_idx, (radial_degree, basis_indices) in enumerate(sorted(basis_by_degree.items())):
    ...     for col_idx, basis_idx in enumerate(basis_indices):
    ...         # Create 2D array with NaN outside unit circle
    ...         tmp = torch.full((grid_size, grid_size), float('nan'))
    ...         tmp[mask] = basis_values[:, basis_idx]
    ...         
    ...         # Plot
    ...         ax = axes[row_idx, col_idx]
    ...         im = ax.imshow(tmp.numpy(), extent=[-1, 1, -1, 1], 
    ...                       origin='lower', cmap='jet', vmin=-1, vmax=1)
    ...         azimuthal = zernike.get_azimuthal_frequency(basis_idx)
    ...         ax.set_title(f"$Z^{{{azimuthal}}}_{{{radial_degree}}}$", fontsize=25)
    ...         ax.set_xticks([])
    ...         ax.set_yticks([])
    ...         ax.set_aspect('equal')
    >>> 
    >>> plt.tight_layout()
    >>> plt.show()

Notes:
    - Zernike polynomials are only defined for points within the unit circle (r ≤ 1)
    - Radial order n determines the number of radial variations
    - Azimuthal frequency m determines the angular variations and symmetry
"""

# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "basis_function",
    "get_num_basis",
    "get_radial_order",
    "get_azimuthal_frequency"
]

import torch
import math
import numpy as np

def __zernike_calc(n, m, r_powers):
    radial_sum = torch.zeros_like(r_powers[0])
    m = abs(m)
    
    for k in range((n - m) // 2 + 1):
        coef = math.factorial(n - k) / (
            math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
        if k%2==1:
            coef = -coef
        power_idx = n - 2 * k - m
        
        if power_idx < 0:
            raise RuntimeError("Potential zero division!")
            #tmp = r_powers[abs(power_idx)]
            #radial_sum += coef / tmp
        if power_idx % 2 == 1:
            raise RuntimeError("tried to acces odd power idx!")
            
        radial_sum += coef*r_powers[abs(power_idx)]
    
    return radial_sum

def basis_function(max_radial_order, points):
    """
    Compute Zernike polynomials for a given set of points.
    
    Args:
        max_radial_order (int): Maximum radial order.
        points (torch.Tensor): Tensor of shape (N, 2) containing the x and y coordinates of the points.
    
    Returns:
        torch.Tensor: Tensor of shape (N, num_coeffs) containing the Zernike polynomial values.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    #r = torch.sqrt(x**2 + y**2)
    r2 = x**2 + y**2
    # Precompute powers of r from r^0 to r^max_radial_order
    r_powers = [] #[r ** i for i in range(max_radial_order+1)]
    for i in range(max_radial_order+1):
        if i%2 == 0:
            r_powers += [r2**(i/2.0)]
        else:
            r_powers += [None]
        
    # List to store Zernike polynomial results
    zernike_polynomials = []
    
    # Loop over radial and azimuthal degrees
    for n in range(max_radial_order + 1):
        for m in range(-n, n + 1, 2):  # m must have the same parity as n
            #r_m = r_powers[abs(m)]  # Precompute r^m for both cos and sin components
            
            if m >= 0:
                #TODO Remove weird complex number stuff!
                multiplier = torch.real((y + 1j * x)**abs(m))#TODO multiply after zerinke_calc!
                #this is to slow!!
                zernike_polynomials.append(multiplier*__zernike_calc(n, m, r_powers))
            else:
                multiplier = torch.imag((y + 1j * x)**abs(m))
                zernike_polynomials.append(multiplier*__zernike_calc(n, m, r_powers))
    
    # Stack all Zernike polynomials into a single tensor
    zernike_basis = torch.stack(zernike_polynomials, dim=1)
    
    return zernike_basis

def get_num_basis(max_radial_order):
    """
    Calculate the number of basis functions for Zernike polynomials up to a given radial order.
    The number of basis functions is given by the formula (n + 1) * (n + 2) / 2.

    Args:
        max_radial_order (int): Maximum radial order.

    Returns:
        int: Number of coefficients.
    """

    n = max_radial_order+1
    return int(n*(n+1) / 2)


def get_radial_order(basis_idx):
    """
    Calculate the radial degree from the basis function index.
    
    Args:
        basis_idx (int): Index of the basis function.
    
    Returns:
        int: Radial degree.
    """
    basis_idx_runner = basis_idx

    num_azimuthal_frequencies = 1
    row_idx = 0
    while True:
        if basis_idx_runner < num_azimuthal_frequencies:
            return row_idx

        basis_idx_runner = basis_idx_runner-num_azimuthal_frequencies
        
        num_azimuthal_frequencies += 1
        row_idx += 1

def get_azimuthal_frequency(basis_idx):
    """
    Calculate the azimuthal frequency from the basis function index.

    Args:
        basis_idx (int): Index of the basis function.
    
    Returns:
        int: Azimuthal frequency (m value).
    """
    # First get the radial degree n
    row_idx = get_radial_order(basis_idx)
    
    
    num_azimuthal_frequencies = 1
    basis_idx_runner = basis_idx
    for k in range(row_idx):
        basis_idx_runner = basis_idx_runner-num_azimuthal_frequencies
        num_azimuthal_frequencies += 1
    
    x_idx_start = None
    if num_azimuthal_frequencies % 2 == 0:
        half = num_azimuthal_frequencies // 2
        tmp = (half-1)*2+1
        x_idx_start = - tmp
    else:
        half = num_azimuthal_frequencies // 2
        tmp = (half)*2
        x_idx_start = - tmp
        #x_idx_start = - (num_azimuthal_frequencies // 2)
    # For radial degree n, we have m values: -n, -n+2, ..., -2, 0, 2, ..., n-2, n
    # The azimuthal frequency m follows the pattern:
    # pos_in_row = 0 -> m = -n
    # pos_in_row = 1 -> m = -n + 2
    # ...
    # pos_in_row = n -> m = n
    return x_idx_start + basis_idx_runner*2