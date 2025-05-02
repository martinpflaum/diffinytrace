# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import torch

def precompute_legendre_polynomials(degree, x):
    """
    Precomputes all Legendre polynomials up to a given degree.

    Args:
    degree (int): Maximum degree of the Legendre polynomials.
    x (torch.Tensor): Input tensor for x-coordinates.

    Returns:
    list of torch.Tensor: List of precomputed Legendre polynomials [P_0(x), P_1(x), ..., P_degree(x)].
    """
    P = [torch.ones_like(x)]  # P_0(x) = 1
    if degree >= 1:
        P.append(x)  # P_1(x) = x
    
    for n in range(2, degree + 1):
        Pn = ((2 * n - 1) * x * P[-1] - (n - 1) * P[-2]) / n
        P.append(Pn)
    
    return P

def legendre_2d_basis(degree, x, y):
    """
    Generates 2D Legendre polynomial basis functions up to a given degree using precomputed 1D polynomials.

    Args:
    degree (int): Maximum degree of the Legendre polynomials.
    x (torch.Tensor): x-coordinates as a torch tensor.
    y (torch.Tensor): y-coordinates as a torch tensor.

    Returns:
    torch.Tensor: Tensor of shape (num_basis_functions, *x.shape) with all 2D basis functions.
    """
    # Precompute 1D Legendre polynomials for x and y
    Px = precompute_legendre_polynomials(degree, x)
    Py = precompute_legendre_polynomials(degree, y)

    basis_functions = {}

    for i in range(degree + 1):
        for j in range(degree + 1 - i):  # Ensure that i + j <= degree
            if not i+j in basis_functions.keys():
                basis_functions[i+j] = []
            basis_functions[i+j] += [Px[i] * Py[j]]
    out = []
    for key in basis_functions.keys():
        out += basis_functions[key]

    # Stack basis functions along a new dimension
    return torch.stack(out, dim=1)

def get_num_coeff(degree):
    """
    Returns the number of coefficients for a given degree of Legendre polynomials.
    The number of coefficients is given by the formula (degree + 1) * (degree + 2) / 2.
    
    Args:
        degree (int): Degree of the Legendre polynomial.
        
    Returns:
        int: Number of coefficients.
    """
    return (degree + 1) * (degree + 2) // 2
