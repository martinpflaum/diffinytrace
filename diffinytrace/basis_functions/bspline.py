# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


import matplotlib.pyplot as plt
import torch
def cox_de_boor_recursion(U,k,n,xis,k_curr):
    r"""Cox-de Boor recursion for B-spline basis functions.
    
    Args:
        U (torch.Tensor): Knot vector.
        k (int): Order of the B-spline.
        n (int): Number of control points.
        xis (torch.Tensor): Evaluation points.
        k_curr (int): Current recursion level.
    
    Returns:
        torch.Tensor: B-spline basis function values at the evaluation points.
    """
    #TODO REMOVE n 
    U = U.to(xis.device)
    if k_curr<0:
        raise RuntimeError("cox_de_boor_recursion wrong input. k_curr < 0")
    if k_curr == 0:
        out = (xis[None].T < U[1:]).to(xis.device,dtype=xis.dtype)*(xis[None].T >= U[:-1]).to(xis.device,dtype=xis.dtype)
        mask = xis == U[U.shape[0]-1]
        #out[mask] = torch.zeros_like(out[mask])
        out[mask,-1] = 1.0
        return out

    Ni = cox_de_boor_recursion(U,k,n,xis,k_curr-1)
        
    Niplus = Ni[:,1:]
    Ni = Ni[:,:-1]

         
    def save_divisor(up,divisor):
        #something is wrong here fix it
        zeros = divisor==0
        divisor[:,zeros.reshape(-1)] = 1.
        up[:,zeros.reshape(-1)] = 1.
        return up/divisor

    tmp1 = save_divisor((xis[None].T - U[:-k_curr-1]),(U[k_curr:-1]-U[:-k_curr-1]).reshape(1,-1))
    tmp2 = save_divisor(-(xis[None].T-U[k_curr+1:]),(U[k_curr+1:]-U[1:-k_curr]).reshape(1,-1))
    
    out =  tmp1*Ni+tmp2*Niplus
    
    return out

def basis_1d(points,U,k,n,val_range):
    #TODO REMOVE n 
    if U[0] != 0.0 or U[-1]!=1.0:
        raise RuntimeError("Knots should always between 0.0 and 1.0 and also contain these values!")
    points = (points-val_range[0])/(val_range[1]-val_range[0])#points are now between 0. and 1.0
    k_curr = k
    return cox_de_boor_recursion(U,k,n,points,k_curr-1)

def basis_2d(points,Us,orders,ns,x_range,y_range):
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise RuntimeError("The points must be in local coordinates and of shape [#points,2]")
    device = points.device
    if Us[0].device != device:
        Us[0] = Us[0].to(device)
    if Us[1].device != device:
        Us[1] = Us[1].to(device)
    
    #Move evaluation to cor.py make abstraction
    Ns1 = basis_1d(points[:,0],Us[0],orders[0],ns[0],x_range)    
    Ns2 = basis_1d(points[:,1],Us[1],orders[1],ns[1],y_range)
    num_points = Ns1.shape[0]
    N2D = Ns1.reshape(num_points,-1,1)*Ns2.reshape(num_points,1,-1)
    
    return N2D


def surface_2d(points, Us, orders, ns, x_range, y_range, control_points):
    #TODO maybe test more...
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise RuntimeError("The points must be in local coordinates and of shape [#points,2]")
    device = points.device
    if Us[0].device != device:
        Us[0] = Us[0].to(device)
    if Us[1].device != device:
        Us[1] = Us[1].to(device)
    
    num_points = points.shape[0]
    # Compute basis functions in x and y directions
    Ns1 = basis_1d(points[:, 0], Us[0], orders[0], ns[0], x_range)
    Ns2 = basis_1d(points[:, 1], Us[1], orders[1], ns[1], y_range)
    
    # Unique knots and step size in x-direction
    U1_unique = torch.unique(Us[0])
    du1 = U1_unique[1] - U1_unique[0]

    # Unique knots and step size in y-direction
    U2_unique = torch.unique(Us[1])
    du2 = U2_unique[1] - U2_unique[0]

    # Compute start and end indices for basis functions in x and y directions
    _points_x = (points[:,0]-x_range[0])/(x_range[1]-x_range[0])#points are now between 0. and 1.0
    _points_y = (points[:,1]-y_range[0])/(y_range[1]-y_range[0])#points are now between 0. and 1.0
    
    start_idx1 = (_points_x / du1).floor().to(torch.int32)
    start_idx2 = (_points_y / du2).floor().to(torch.int32)
    
    # Wrap the indices using modulus to ensure they fit within the control points' range
    n1 = Ns1.shape[1]
    n2 = Ns2.shape[1]
    start_idx1 = start_idx1 % n1
    start_idx2 = start_idx2 % n2
    
    # Reshape control points to be compatible with the grid
    single_valued = False
    if len(control_points.shape) == 1:
        single_valued = control_points.shape[0]==ns[0]*ns[1]
    else:
        if len(control_points.shape) == 2:
            single_valued = control_points.shape[0] == ns[0] and control_points.shape[1] == ns[1]
            
            
    control_points = control_points.reshape(ns[0],ns[1], -1)

    # Create tensor for extracting control points (using broadcasting)
    idx1 = torch.arange(orders[0] + 1).unsqueeze(0).to(start_idx1.device) + start_idx1.unsqueeze(1)
    idx2 = torch.arange(orders[1] + 1).unsqueeze(0).to(start_idx2.device) + start_idx2.unsqueeze(1)

    # Wrap the indices around using modulus for valid ranges
    idx1 = idx1 % n1
    idx2 = idx2 % n2

    # Extract basis function values and control points using broadcasting
    basis_values_1d_1 = Ns1.gather(1, idx1)  # Extract relevant basis values for x
    basis_values_1d_2 = Ns2.gather(1, idx2)  # Extract relevant basis values for y

    # Compute outer product of basis functions (NxM grid for each point)
    N2D = basis_values_1d_1[:, :, None] * basis_values_1d_2[:, None, :]
    N2D = N2D.reshape(num_points,-1)
    # Use advanced indexing to gather control points
    
    idx_flat = idx1.reshape(num_points,-1,1)*control_points.shape[1]+idx2.reshape(num_points,1,-1)
    idx_flat = idx_flat.reshape(num_points,-1)
    control_points = control_points.reshape(ns[0]*ns[1], -1)
    control_point_subset = control_points[idx_flat]
    surface_points = torch.einsum('ik,ikl->il', N2D, control_point_subset)
    if single_valued:
        surface_points = surface_points[:,0]
    return surface_points



def insert_knot1D_single(U,korder,new_knot,control_points,dim=0):
    device = control_points.device
    dtype = control_points.dtype
    if dim != 0:
        control_points = control_points.transpose(0,dim)
    if not torch.is_tensor(U):
        U = torch.tensor(U)
    k = (U<new_knot).int().sum()
    U_new = torch.zeros((U.shape[0]+1),device=device,dtype=dtype)
    U_new[:k] = U[:k]
    U_new[k] = new_knot
    U_new[k+1:] = U[k:]
    new_shape = list(control_points.shape)
    new_shape[0] += 1
    control_points_new = torch.zeros((new_shape),device=device,dtype=dtype)
    i_s = torch.arange(k-korder,k)
    if i_s[0] > 0:
        control_points_new[:i_s[0]] = control_points[:i_s[0]]
    alpha_i = (new_knot-U_new[i_s])/(U_new[i_s+korder]-U_new[i_s])
    alpha_i = alpha_i.reshape(-1,*[1]*(control_points.dim()-1))
    control_points_new[i_s] = alpha_i*control_points[i_s]+(1.0-alpha_i)*control_points[i_s-1]
    control_points_new[i_s[-1]+1:] = control_points[i_s[-1]:]
    if dim != 0:
        control_points_new = control_points_new.transpose(0,dim)
    return U_new,control_points_new


def insert_knots1D(U,korder,new_knot_list,control_points,dim=0):
    for new_knot in new_knot_list:
        U,control_points = insert_knot1D_single(U,korder,new_knot,control_points,dim=dim)
    return U,control_points


def refine2D(Us,orders,coeff=None):
    U1,U2 = Us
    U1_unique = torch.unique(U1)
    dU1 = U1_unique[1]-U1_unique[0]
    new_knots_U1 =torch.linspace(torch.min(U1_unique)+dU1*0.5,torch.max(U1_unique)-dU1*0.5,U1_unique.shape[0]-1)
    U2_unique = torch.unique(U2)
    dU2 = U2_unique[1]-U2_unique[0]
    new_knots_U2 =torch.linspace(torch.min(U2_unique)+dU2*0.5,torch.max(U2_unique)-dU2*0.5,U2_unique.shape[0]-1)
         
    if not coeff is None:
        coeff = coeff.detach()
        U1,coeff = insert_knots1D(U1,orders[0],new_knots_U1,coeff,dim=0)
        U2,coeff = insert_knots1D(U2,orders[1],new_knots_U2,coeff,dim=1)
        return [U1,U2],coeff
    else:
        return U1,U2