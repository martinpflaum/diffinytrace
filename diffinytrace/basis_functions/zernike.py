# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


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

def basis_function(max_n, points):
    x = points[:, 0]
    y = points[:, 1]
    
    #r = torch.sqrt(x**2 + y**2)
    r2 = x**2 + y**2
    # Precompute powers of r from r^0 to r^max_n
    r_powers = [] #[r ** i for i in range(max_n+1)]
    for i in range(max_n+1):
        if i%2 == 0:
            r_powers += [r2**(i/2.0)]
        else:
            r_powers += [None]
        
    # List to store Zernike polynomial results
    zernike_polynomials = []
    
    # Loop over radial and azimuthal degrees
    for n in range(max_n + 1):
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

def get_num_coeffs(max_radial_degree):
    n = max_radial_degree+1
    return int(n*(n+1) / 2)


def get_radial_degree(num_coeff):
    out = math.sqrt(num_coeff*2)
    return int(np.floor(out))-1