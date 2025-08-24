r"""
B-spline surfaces for freeform geometry.

B-splines are popular for describing freeform surfaces because they allow
local changes to the geometry :cite:`THBray`. Their smoothness is controlled
by the spline degrees, which determine continuity and differentiability
:cite:`IGA`. A tensor-product B-spline surface is defined by two *knot vectors*,
a grid of univariate B-spline *basis functions*, and a bi-directional net of
*control points* :cite:`nurbs`. Below, we summarize the main components.

Notes:
    **Knot vectors.**
    A surface uses two (typically clamped) nondecreasing knot vectors
    :math:`U` and :math:`V`:

    .. math::

       U = \{\underbrace{0,\dots,0}_{p+1},\, u_{p+1}, \dots, u_{n},\,
            \underbrace{1,\dots,1}_{p+1}\}, \qquad
       V = \{\underbrace{0,\dots,0}_{q+1},\, v_{q+1}, \dots, v_{m},\,
            \underbrace{1,\dots,1}_{q+1}\}.

    Here :math:`p` and :math:`q` are the degrees in the :math:`u`- and
    :math:`v`-directions. A knot vector :math:`U=\{u_0,\dots,u_M\}` is a
    nondecreasing sequence, i.e., :math:`u_i \le u_{i+1}`; each element is a
    *knot*.

    **Univariate B-spline basis (Cox–de Boor).**
    In the :math:`u`-direction (analogously for :math:`v`), the basis
    :math:`\{N_{i,p}\}` is defined recursively :cite:`nurbs`:

    .. math::

       N_{i,0}(u) =
       \begin{cases}
         1, & u_i \le u < u_{i+1},\\
         0, & \text{otherwise},
       \end{cases}

    .. math::

       N_{i,p}(u) =
       \frac{u - u_i}{u_{i+p} - u_i}\, N_{i,p-1}(u) \;+\;
       \frac{u_{i+p+1} - u}{u_{i+p+1} - u_{i+1}}\, N_{i+1,p-1}(u).

    In the :math:`v`-direction, the basis :math:`\{M_{j,q}\}` is

    .. math::

       M_{j,0}(v) =
       \begin{cases}
         1, & v_j \le v < v_{j+1},\\
         0, & \text{otherwise},
       \end{cases}

    .. math::

       M_{j,q}(v) =
       \frac{v - v_j}{v_{j+q} - v_j}\, M_{j,q-1}(v) \;+\;
       \frac{v_{j+q+1} - v}{v_{j+q+1} - v_{j+1}}\, M_{j+1,q-1}(v).

    **Control points.**
    Control points :math:`\mathbf{P}_{i,j}` link the basis to geometry.
    They can be scalars, 2D, or 3D vectors.

    **Surface definition.**
    The tensor-product B-spline surface is

    .. math::
       :label: eq-bspline-Z

       Z(u,v) = \sum_{i=0}^{n} \sum_{j=0}^{m}
                N_{i,p}(u)\, M_{j,q}(v)\, \mathbf{P}_{i,j}.

    **Implementation details (this library).**
    We use scalar control points :math:`\mathbf{P}_{i,j}` (height field),
    uniformly increasing clamped knot vectors, and :math:`u,v \in [0,1]`.
    To couple an explicit surface to the ray tracer, we map physical
    coordinates :math:`\hat{x}_1,\hat{x}_2` to the parametric domain via a
    scale :math:`h`:

    .. math::

       S(\hat{x}_1,\hat{x}_2) =
       Z\!\left(\frac{\hat{x}_1}{h},\, \frac{\hat{x}_2}{h}\right).

.. figure:: _static/bspline_plot1.png
   :alt: Freeform lens with B-spline surface
   :width: 60%
   :align: center

   Visualization of a Freeform lens with a B-spline surface.

Examples:
    Define a lens with a B-spline surface and plot it:

    .. code-block:: python

       import torch
       import diffinytrace as dit

       aperture_half = 30.0
       aperture_radius = aperture_half
       lens_thickness = 8.0
       material = dit.materials["NBK7"]
       transform = dit.transforms.Identity()

       # degree [p, q] and control net size [n_u, n_v] (example values)
       bspline = dit.Bspline(aperture_half, [3, 3], [8, 8])
       plane = dit.Plane()

       with torch.no_grad():
           bspline.coeff.data = torch.randn_like(bspline.coeff.data) * 3.0

       lens = dit.Lens(transform, lens_thickness, bspline, plane,
                       material, aperture_radius)

       dit.plotting.system3D.plot(lens, zticks=[0, 5])

"""

# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "cox_de_boor_recursion",
    "basis_1D",
    "basis_2D",
    "surface_2D",
    "insert_knot_1D_single",
    "insert_knots1D",
    "refine2D"
]

import torch
from typing import Tuple,List,Callable,Optional,Union


def cox_de_boor_recursion(U: torch.Tensor, k: int, n: int, xis: torch.Tensor, k_curr: int) -> torch.Tensor:
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

def basis_1D(points:torch.Tensor,
             U:torch.Tensor,
             k:int,
             n:int,
             val_range:tuple[float,float])->torch.Tensor:
    """
    Compute 1D B-spline basis functions at given points.

    Args:
        points (torch.Tensor): Points where the basis functions are evaluated.
        U (torch.Tensor): Knot vector.
        k (int): Order of the B-spline.
        n (int): Number of control points.
        val_range (tuple[float, float]): Range of the target interval (e.g., (0.0, 1.0)).

    Returns:
        torch.Tensor: B-spline basis function values at the evaluation points.

    Raises:
        RuntimeError: If the knot vector does not start at 0.0 or end at 1.0.

    Example:
        >>> import torch
        >>> import matplotlib.pyplot as plt
        >>> from diffinytrace.basis_functions import bspline
        >>> U = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
        >>> n = 3
        >>> k = 3  # This is order 3
        >>> print(U[0], U[-1])
        >>> xis = torch.linspace(0, 1, 100)
        >>> xN = bspline.basis_1D(xis, U, k, n, [0., 1.])
        >>> num_points = xN.shape[0]
        >>> tmp = xN.reshape(num_points, -1, 1) * xN.reshape(num_points, 1, -1)
        >>> for yin in xN.T:
        ...     plt.plot(xis, yin)
        >>> plt.gca().set_aspect('equal')
    """
    
    if U[0] != 0.0 or U[-1]!=1.0:
        raise RuntimeError("Knots should always between 0.0 and 1.0 and also contain these values!")
    points = (points-val_range[0])/(val_range[1]-val_range[0])#points are now between 0. and 1.0
    k_curr = k
    return cox_de_boor_recursion(U,k,n,points,k_curr-1)

def basis_2D(points:torch.Tensor,
             Us:List[torch.Tensor],
             orders:List[int],
             ns:List[int],
             x_range:tuple,
             y_range:tuple) -> torch.Tensor:
    """Compute the 2D B-spline basis functions for given points.
    
    Args:
        points (torch.Tensor): Points where the basis functions are evaluated.
        Us (list[torch.Tensor]): Knot vectors for x and y directions.
        orders (list[int]): Orders of the B-spline in x and y directions.
        ns (list[int]): Number of control points in x and y directions.
        x_range (tuple): Range of the target plane in the x direction.
        y_range (tuple): Range of the target plane in the y direction.
    
    Returns:
        torch.Tensor: 2D B-spline basis function values at the evaluation points.
    
    Example:
        >>> import diffinytrace as dit
        >>> from diffinytrace.basis_functions.bspline import basis_2D
        >>> import torch
        >>> 
        >>> U1 = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
        >>> Us = [U1, U1]
        >>> ps = [3, 3]
        >>> ns = [3, 3]
        >>> 
        >>> side_points = 100
        >>> _x = torch.linspace(0, 1, side_points)
        >>> _y = torch.linspace(0, 1, side_points)
        >>> grid_y, grid_x = torch.meshgrid(_y, _x, indexing='ij')
        >>> points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=-1)
        >>> 
        >>> N2D = basis_2D(points, Us, ps, ns, torch.tensor([0, 1]), torch.tensor([0, 1]))
        >>> 
        >>> xi = 0
        >>> yi = 2
        >>> dit.plotting.quantity2D.plot(
        >>>     N2D[:, yi, xi].reshape(side_points, side_points),
        >>>     "basis fun",
        >>>     [0, 1],
        >>>     [0, 1],
        >>>     xlabel="x",
        >>>     ylabel="y"
        >>> )

    Raises:
        RuntimeError: If the input points are not in local coordinates or have an incorrect shape.

    """
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise RuntimeError("The points must be in local coordinates and of shape [#points,2]")
    device = points.device
    if Us[0].device != device:
        Us[0] = Us[0].to(device)
    if Us[1].device != device:
        Us[1] = Us[1].to(device)
    
    #Move evaluation to cor.py make abstraction
    Ns1 = basis_1D(points[:,0],Us[0],orders[0],ns[0],x_range)    
    Ns2 = basis_1D(points[:,1],Us[1],orders[1],ns[1],y_range)
    num_points = Ns1.shape[0]
    N2D = Ns1.reshape(num_points,-1,1)*Ns2.reshape(num_points,1,-1)
    
    return N2D


def surface_2D(points: torch.Tensor, Us: List[torch.Tensor], orders: List[int], ns: List[int], x_range: tuple, y_range: tuple, control_points: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a 2D B-spline surface at given points using provided knot vectors, orders, and control points.

    Args:
        points (torch.Tensor): Points where the surface is evaluated, shape [num_points, 2].
        Us (List[torch.Tensor]): Knot vectors for x and y directions [U_x, U_y].
        orders (List[int]): Orders of the B-spline in x and y directions [order_x, order_y].
        ns (List[int]): Number of control points in x and y directions [n_x, n_y].
        x_range (tuple): Range of the target plane in the x direction (min, max).
        y_range (tuple): Range of the target plane in the y direction (min, max).
        control_points (torch.Tensor): Control points, shape [n_x, n_y, ...] or [n_x*n_y, ...].

    Returns:
        torch.Tensor: Evaluated surface points at the input locations.

    Raises:
        RuntimeError: If the input points are not in local coordinates or have an incorrect shape.

    Example:
        >>> import torch
        >>> from diffinytrace.basis_functions import bspline
        >>> n_x, n_y = 4, 4
        >>> control_points = torch.randn((n_x, n_y, 2))
        >>> k_x, k_y = 3, 3
        >>> U_x = torch.linspace(0, 1, n_x + k_x)
        >>> U_y = torch.linspace(0, 1, n_y + k_y)
        >>> points = torch.rand((100, 2))
        >>> surface = bspline.surface_2D(points, [U_x, U_y], [k_x, k_y], [n_x, n_y], (0.0, 1.0), (0.0, 1.0), control_points)
    """
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise RuntimeError("The points must be in local coordinates and of shape [#points,2]")
    device = points.device
    if Us[0].device != device:
        Us[0] = Us[0].to(device)
    if Us[1].device != device:
        Us[1] = Us[1].to(device)
    
    num_points = points.shape[0]
    # Compute basis functions in x and y directions
    Ns1 = basis_1D(points[:, 0], Us[0], orders[0], ns[0], x_range)
    Ns2 = basis_1D(points[:, 1], Us[1], orders[1], ns[1], y_range)
    
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
    basis_values_1D_1 = Ns1.gather(1, idx1)  # Extract relevant basis values for x
    basis_values_1D_2 = Ns2.gather(1, idx2)  # Extract relevant basis values for y

    # Compute outer product of basis functions (NxM grid for each point)
    N2D = basis_values_1D_1[:, :, None] * basis_values_1D_2[:, None, :]
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



def insert_knot_1D_single(U: torch.Tensor, 
                         korder: int, 
                         new_knot: torch.Tensor, 
                         control_points: torch.Tensor, 
                         dim: int=0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Insert a single knot into a 1D B-spline knot vector and update control points.

    Args:
        U (torch.Tensor): Original knot vector.
        korder (int): Order of the B-spline.
        new_knot (torch.Tensor or float): Knot value to insert.
        control_points (torch.Tensor): Control points (shape: [n, ...]).
        dim (int, optional): Dimension along which to insert the knot (default: 0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Updated knot vector, updated control points).

    Example:
        >>> import torch
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> n = 4
        >>> control_points = torch.randn((n, 2))  # Random control points
        >>> k = 4  # Quadratic B-spline
        >>> U = torch.tensor([0.0] * (k - 1) + list(np.linspace(0, 1.0, n + k - 2 * (k - 1))) + [1.0] * (k - 1))
        >>> U = U.float()
        >>> print(U.shape[0] - k == n, n >= k)
        >>> for m in range(100):
        ...     U_new, new_control_points = bspline.insert_knot_1D_single(U, k, torch.rand((1)), control_points)
        ...     print("new_control_points", new_control_points)
        ...     print("control_points", control_points)
        ...     xis = torch.linspace(0, 1, 1000)
        ...     xN1 = bspline.basis_1D(xis, U, k, 3, [0, 1.])
        ...     out1 = xN1 @ control_points
        ...     xN2 = bspline.basis_1D(xis, U_new, k, 4, [0, 1.])
        ...     out2 = xN2 @ new_control_points
        ...     plt.plot(out1[:, 0], out1[:, 1], linewidth=5.0)
        ...     plt.plot(out2[:, 0], out2[:, 1], "--")
        ...     torch.mean((out1 - out2) ** 2)
    """
    
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


def insert_knots_1D(U:torch.Tensor, 
                    korder:int, 
                    new_knot_list:List[float], 
                    control_points:torch.Tensor, 
                    dim:int=0)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Insert multiple knots into a 1D B-spline knot vector and update control points.

    Args:
        U (torch.Tensor): Original knot vector.
        korder (int): Order of the B-spline.
        new_knot_list (Iterable): List or tensor of knot values to insert.
        control_points (torch.Tensor): Control points.
        dim (int, optional): Dimension along which to insert the knots (default: 0).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Updated knot vector, updated control points).
    """
    for new_knot in new_knot_list:
        U,control_points = insert_knot_1D_single(U,korder,new_knot,control_points,dim=dim)
    return U,control_points


def refine_2D(Us:List[torch.Tensor],
              orders:List[int], 
              coeff:Optional[torch.Tensor]=None) -> Union[Tuple[List[torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Refine 2D B-spline knot vectors by inserting midpoints between existing knots.
    Optionally updates coefficients (control points) accordingly.

    Args:
        Us (list[torch.Tensor]): List of knot vectors [U1, U2] for x and y directions.
        orders (list[int]): List of orders [order_x, order_y] for x and y directions.
        coeff (torch.Tensor, optional): Coefficient tensor (control points) to update.

    Returns:
        Tuple[list[torch.Tensor], torch.Tensor] or Tuple[torch.Tensor, torch.Tensor]:
            If coeff is provided, returns ([U1_new, U2_new], coeff_new).
            Otherwise, returns (U1_new, U2_new).
    """
    U1,U2 = Us
    U1_unique = torch.unique(U1)
    dU1 = U1_unique[1]-U1_unique[0]
    new_knots_U1 =torch.linspace(torch.min(U1_unique)+dU1*0.5,torch.max(U1_unique)-dU1*0.5,U1_unique.shape[0]-1)
    U2_unique = torch.unique(U2)
    dU2 = U2_unique[1]-U2_unique[0]
    new_knots_U2 =torch.linspace(torch.min(U2_unique)+dU2*0.5,torch.max(U2_unique)-dU2*0.5,U2_unique.shape[0]-1)
         
    if not coeff is None:
        coeff = coeff.detach()
        U1,coeff = insert_knots_1D(U1,orders[0],new_knots_U1,coeff,dim=0)
        U2,coeff = insert_knots_1D(U2,orders[1],new_knots_U2,coeff,dim=1)
        return [U1,U2],coeff
    else:
        return U1,U2