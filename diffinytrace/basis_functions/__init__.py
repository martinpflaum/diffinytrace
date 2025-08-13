# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

"""
The basis functions module provides a collection of functions for working with B-splines, Legendre polynomials, Zernike polynomials, and Chebyshev polynomials.

Examples:

    Example 2: Using 1D B-spline basis functions
    ```python
    import torch
    import matplotlib.pyplot as plt

    U = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
    n = 3
    p = 3  # This is order 3
    xis = torch.linspace(0, 1, 100)
    xN = bspline_basis_funs1D(U, p, n, xis)

    num_points = xN.shape[0]
    tmp = xN.reshape(num_points, -1, 1) * xN.reshape(num_points, 1, -1)

    for yin in xN.T:
        plt.plot(xis, yin)

    plt.gca().set_aspect('equal')
    ```

    Example 3: Inserting control points in a B-spline
    ```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    n = 4
    control_points = torch.randn((n, 2))  # Random control points
    p = 4  # Quadratic B-spline
    U = torch.tensor([0.0] * (p - 1) + list(np.linspace(0, 1.0, n + p - 2 * (p - 1))) + [1.0] * (p - 1))
    U = U.float()

    print(U.shape[0] - p == n, n >= p)
    for m in range(100):
        U_new, new_control_points = bspline_insert1D(U, p, torch.rand((1)), control_points)
        print("new_control_points", new_control_points)
        print("control_points", control_points)

    xis = torch.linspace(0, 1, 1000)
    xN1 = bspline_basis_funs1D(U, p, 3, xis)
    out1 = xN1 @ control_points
    xN2 = bspline_basis_funs1D(U_new, p, 4, xis)
    out2 = xN2 @ new_control_points
    plt.plot(out1[:, 0], out1[:, 1], linewidth=5.0)
    plt.plot(out2[:, 0], out2[:, 1], "--")
    torch.mean((out1 - out2) ** 2)
    ```

    Example 4: Using Legendre polynomials (Not finished)
    ```python
    degree = 21
    get_num_coeff(degree)

    # Example usage
    x = torch.tensor([0.5,0.1])
    y = torch.tensor([0.5,0.1])
    basis = legendre_2d_basis(degree, x, y)
    # Example usage
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)

    # Compute the 2D Legendre polynomial of degree (2, 3)
    L_2_3 = legendre_2d(x, y, 3, 1)
    L_2_3.shape
    x = torch.linspace(-1, 1, 100)
    P = precompute_legendre_polynomials(21, x)
    import matplotlib.pyplot as plt
    for L_x in P:
        plt.plot(x,L_x)
    ```
"""

__all__ = [
    "bspline",
    "legendre",
    "zernike",
    "chebyshev"
]

from . import bspline
from . import legendre
from . import zernike
from . import chebyshev