"""
Copyright (C) 2024 Martin Pflaum

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import torch

def precompute_legendre_polynomials(degree, x):
    """
    Precomputes all Legendre polynomials up to a given degree.

    Parameters:
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

    Parameters:
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
     return (degree + 1) * (degree + 2) // 2

"""
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

"""