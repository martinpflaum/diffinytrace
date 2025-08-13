# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "Surface",
    "Plane",
    "Aspheric",
    "Bspline",
    "Legendre",
    # "Zernike"  # Uncomment if you implement Zernike
]

import torch
import torch.nn as nn
import numpy as np
#from diffinytrace.basis_functions import *
from .transforms import SemiFunctionalModule
from .optimize import make_parameter_from_input
from . import basis_functions

class Surface(SemiFunctionalModule):
    r"""
    While we all have an intuitive idea of what curves and surfaces are, we need a mathematically accurate definition from which we can proceed to illustrate how different types of algorithms are implemented. In the following, we will introduce three common ways of describing curves and surfaces.

    1. **Parametric Equations**  
    Parametric curves are functions that map a single variable :math:`\theta` (the parameter) to a vector in :math:`\mathbb{R}^2`. Thus, such curves are referred to as parametrized or parametrically defined curves (see :cite:`implicit_surfaces`). The variable :math:`\theta` is an element of the *parametric domain* of the parametric curve (see :cite:`IGA`). For example, a circle can be described with the *parametric domain* :math:`[0, 2\pi]` and the function :math:`f: [0, 2\pi] \mapsto \mathbb{R}^2`,

    .. math::

        f(\theta) = 
        \begin{bmatrix}
        \cos \theta \\
        \sin \theta
        \end{bmatrix}.

    Similarly, parametric surfaces can be described by a function that maps from a two-dimensional *parametric domain* to :math:`\mathbb{R}^3` (see :cite:`implicit_surfaces`).

    2. **Explicit Equations**  
    Curves and surfaces can also be expressed using explicit equations. When describing a curve with explicit equations, an explicit function :math:`f: \mathbb{R} \to \mathbb{R}` of the form :math:`y = f(x)` assigns a unique value of :math:`y` to each :math:`x \in \mathbb{R}`. The values of :math:`y` can then be seen as a description of the curve. Unfortunately, it is not possible to describe all curves and surfaces with this method. For example, considering the unit circle, only one semicircle can be represented at a time using explicit equations such as:

    .. math::

        y = \sqrt{1 - x^2} \quad \text{or} \quad y = -\sqrt{1 - x^2}.

    Similarly, three-dimensional surfaces can be described explicitly using functions of the form :math:`y = f(x_1, x_2)`, which assign a unique :math:`y`-value to each pair of :math:`(x_1, x_2)`-coordinates (see :cite:`implicit_surfaces`).

    3. **Implicit Equations**  
    A planar curve is defined implicitly, or in Cartesian coordinates, when it is described as the set of solutions to an equation involving two variables, typically expressed as :math:`f(y_1, y_2) = 0`. For example, the equation

    .. math::

        y_1^2 + y_2^2 - 1 = 0

    represents an implicit unit circle in :math:`\mathbb{R}^2`. Similarly, an implicit surface can be expressed with an equation in the form of (see :cite:`implicit_surfaces`):

    .. math::

        f(y_1, y_2, y_3) = 0.

    **Optical Surfaces**  
    In our ray tracer, we use a less general description of the surfaces. We will call surfaces relevant for ray tracing *optical surfaces*. Every *optical surface* is composed of an *explicit surface* :math:`\hat{S}: \mathbb{R}^2 \mapsto \mathbb{R}` and a transformation matrix :math:`M \in \mathbb{R}^{4 \times 4}`. In the following, we will state the *implicit surface* description for the ray tracer itself and the *parametric surface* description for plots and constraint optimization.

    1. **Implicit Surface Description**  
    Here, surfaces are described implicitly by the equation :math:`s(\hat{y}) = 0`. The function :math:`s` is composed of the explicit description :math:`\hat{S}(\hat{x}_1, \hat{x}_2)` and an affine transformation matrix :math:`M` as follows:

    .. math::

        \begin{bmatrix} \hat{x} \\ 1 \end{bmatrix} = M^{-1} \begin{bmatrix} \hat{y} \\ 1 \end{bmatrix}^T

    .. math::

        s(\hat{y}) = \hat{S}(\hat{x}_1(\hat{y}), \hat{x}_2(\hat{y})) - \hat{x}_3(\hat{y})

    This description allows us to calculate ray-surface intersections efficiently. Typically, we do not state :math:`M^{-1}` explicitly in the implementation but simply apply the transformation itself directly.

    2. **Parametric Surface Description**  
    In this approach, surfaces are defined by parameterizing coordinates. For optical surfaces, the surface is described again as a composition of the explicit description :math:`\hat{S}` and a transformation matrix :math:`M` as follows:

    .. math::

        \begin{bmatrix} S(\hat{x}_1, \hat{x}_2) \\ 1 \end{bmatrix} = M \begin{bmatrix} \hat{x}_1 \\ \hat{x}_2 \\ \hat{S}(\hat{x}_1, \hat{x}_2) \\ 1 \end{bmatrix}

    In our library, the *parametric domains* are defined by the lenses or target surfaces (detectors). For example, in the case of a round lens, the *parametric domain* would be the disc determined by the aperture radius. This surface description is typically used for plotting but is also useful in the context of constraint optimization.
    
    Examples:
        >>> import diffinytrace as dit
        >>> aperture_radius = 30.
        >>> lens_thickness = 8.
        >>> material = dit.materials["NBK7"]
        >>> transform = dit.transforms.Identity()
        >>> asphere = dit.Aspheric(1./40., 0.0, [-0.00001])
        >>> plane = dit.Plane()
        >>> lens = dit.Lens(transform, lens_thickness,
        >>>          asphere, plane,
        >>>          material, aperture_radius)
        >>> dit.plotting.system2D.plot(lens)
    """
    @staticmethod
    def functional(O,*params_list):
        raise NotImplementedError("functional not implemented")

    def get_functional_param_args(self):
        raise NotImplementedError("params_list not implemented")

    def explicit(self,local_pos):
        raise NotImplementedError("explicit function not implemented")

class Plane(Surface):
    """
    A class to represent a plane surface in 3D space.
    The plane is defined by the equation z = 0, and the functional method
    returns the z-coordinate of the input points.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def functional(O):
        return O[:,-1]
    
    def get_functional_param_args(self):
        return []
    def explicit(self,local_pos):
        if local_pos.shape[-1] != 2:
            raise RuntimeError("local_pos needs to be of shape [:,2]")
        


        x = local_pos[:,0]
        y = local_pos[:,1]

        O_new = torch.zeros((local_pos.shape[0],3),device=local_pos.device,dtype=local_pos.dtype)
        O_new[:,0] = x
        O_new[:,1] = y

        return self.functional(O_new,*self.get_functional_param_args())
    
class Aspheric(Surface):
    r"""
    This is the aspheric surface class, implementation follows: 
    https://en.wikipedia.org/wiki/Aspheric_lens.

    The surface is parameterized as an implicit function :math:`f(x, y, z) = 0`.
    For simplicity, we assume the surface function :math:`f(x, y, z)` can be decomposed as:

    .. math::

        f(x, y, z) = g(x, y) + h(z),

    where :math:`g(x, y)` and :math:`h(z)` are explicit functions:

    .. math::

        r^2 = x^2 + y^2

    .. math::

        g(x, y) = \frac{c \cdot r^2}{1 + \sqrt{1 - (1 + k) \cdot \frac{r^2}{R^2}}} 
                  + a_0 \cdot r^4 + a_1 \cdot r^6 + \cdots

    .. math::

        h(z) = -z

    Args:
        c (float): Surface curvature, or one over the radius of curvature.
        k (float): Conic coefficient.
        ai (list or None): Aspheric parameters, could be a vector. When None, the surface is spherical.
    """
    def __init__(self, curvature, conic_coeff=None, aspheric_param=None):
        super().__init__()
        if not torch.is_tensor(curvature):
            curvature = torch.tensor(curvature)
        curvature = curvature.to(torch.get_default_dtype())
        
        conic_coeff_requires_grad = True
        if conic_coeff is None:
            conic_coeff_requires_grad = False
            conic_coeff = torch.tensor(0.0)
        
        self.curvature = make_parameter_from_input(curvature)
        self.conic_coeff = make_parameter_from_input(conic_coeff)
        self.conic_coeff.requires_grad = conic_coeff_requires_grad

        self.aspheric_param = None
        if aspheric_param is not None:
            self.aspheric_param = make_parameter_from_input(aspheric_param)

        
    @staticmethod
    def g(x, y,curvature,conic_coeff,aspheric_param):
        return Aspheric._g(x**2 + y**2,curvature,conic_coeff,aspheric_param)

    @staticmethod
    def h(z,curvature,conic_coeff,aspheric_param):
        return -z
    
    @staticmethod
    def _g(r2,curvature,conic_coeff,aspheric_param):
        #r2 is r**2.
        tmp = r2*curvature
        total_surface = tmp / (1 + torch.sqrt(1 - (1+conic_coeff) * tmp*curvature))
        higher_surface = 0.
        if aspheric_param is not None:
            for i in range(len(aspheric_param)):
                higher_surface += aspheric_param[i]*(r2**(i+2))
        return total_surface + higher_surface
    
    @staticmethod
    def functional(O,curvature,conic_coeff,aspheric_param):
        x = O[:,0]
        y = O[:,1]
        z = O[:,2]

        return Aspheric.g(x, y,curvature,conic_coeff,aspheric_param) + Aspheric.h(z,curvature,conic_coeff,aspheric_param)
    
    def get_functional_param_args(self):
        return [self.curvature,self.conic_coeff,self.aspheric_param]
    

    def explicit(self,local_pos):
        if local_pos.shape[-1] != 2:
            raise RuntimeError("local_pos needs to be of shape [:,2]")
        x = local_pos[:,0]
        y = local_pos[:,1]

        O_new = torch.zeros((local_pos.shape[0],3),device=local_pos.device,dtype=local_pos.dtype)
        O_new[:,0] = x
        O_new[:,1] = y

        return self.functional(O_new,*self.get_functional_param_args())

"""
class Zernike(Surface):
    #TODO reimplement!
    def __init__(self,aperture_radius,max_radial_degree):
        super().__init__()
    
        self.max_radial_degree = max_radial_degree
        self.coeff = make_parameter_from_input(torch.zeros((basis_functions.zernike.get_num_coeffs(max_radial_degree))))
        self.aperture_radius = torch.tensor(aperture_radius)

    def refine(self):
        #TODO move to a parent class
        with torch.no_grad():
            coeff = make_parameter_from_input(torch.zeros((basis_functions.zernike.get_num_coeffs(self.max_radial_degree+1))))
            coeff.data[:self.coeff.shape[0]] = self.coeff.data.detach()
            self.coeff = coeff
            self.max_radial_degree = self.max_radial_degree+1
    

    @staticmethod
    def functional(O,coeffs,aperture_radius):
        points = O[:,[0,1]]/aperture_radius
        max_n = basis_functions.zernike.get_radial_degree(coeffs.shape[0])
        zernike_surface = basis_functions.zernike.basis_function(max_n, points)@coeffs
        
        return zernike_surface-O[:,-1]
        
    def get_functional_param_args(self):
        return [self.coeff,self.aperture_radius]
    
    def explicit(self,local_pos):
        if local_pos.shape[-1] != 2:
            raise RuntimeError("local_pos needs to be of shape [:,2]")
        x = local_pos[:,0]
        y = local_pos[:,1]

        O_new = torch.zeros((local_pos.shape[0],3),device=local_pos.device,dtype=local_pos.dtype)
        O_new[:,0] = x
        O_new[:,1] = y

        return self.functional(O_new,*self.get_functional_param_args())
    
"""
def bspline_n_after_refinement(n,k):
    return ((2*n+1)-k)

class Bspline(Surface):
    """
    A class to represent a B-spline surface in 3D space.
    The surface is defined by the B-spline basis functions and control points.
    The functional method returns the z-coordinate of the input points.
    """
    
    def __init__(self,aperture_radius,orders,ns):
        super().__init__()
        #orders is order!!!
        #order = degree + 1

        U1 = [0.]*(orders[0]-1)+list(np.linspace(0.,1.,ns[0]-orders[0]+2))+[1.0]*(orders[0]-1)
        U2 = [0.]*(orders[1]-1)+list(np.linspace(0.,1.,ns[1]-orders[1]+2))+[1.0]*(orders[1]-1)
        U1 = torch.tensor(U1,dtype=torch.get_default_dtype())
        U2 = torch.tensor(U2,dtype=torch.get_default_dtype())
        
        self.Us = [U1,U2]
        self.ns = ns
        self.orders = orders
        print("orders",self.orders)
        print("ns",self.ns)
        self.coeff = make_parameter_from_input(torch.zeros((self.ns)))
        self.aperture_radius = torch.tensor(aperture_radius)    

    def get_CAD_coeff(self,affine_transform):
        """
        Get the CAD coefficients from the affine transform.

        Args:
            affine_transform (torch.Tensor): Affine transformation matrix.

        Returns:
            numpy.ndarray: Control points of the B-spline surface.
        """
        affine_transform = affine_transform.detach().cpu()
        dtype = affine_transform.dtype
        coeff = self.coeff.detach()

        v = torch.zeros((coeff.shape[0],coeff.shape[1],4)).cpu()
        ys = torch.linspace(-self.aperture_radius,self.aperture_radius,coeff.shape[0]).cpu()
        xs = torch.linspace(-self.aperture_radius,self.aperture_radius,coeff.shape[1]).cpu()
        xs,ys = torch.meshgrid(xs,ys)
        v[:,:,0] = xs
        v[:,:,1] = ys
        v[:,:,2] = coeff
        v[:,:,3] = torch.ones_like(v[:,:,3])  
        v = v.reshape(-1,4).cpu().to(dtype=dtype)
        Mv = None
        M = affine_transform
        Mv = v@M.T
        out = Mv[:,[0,1,2]]
        out = out.reshape(coeff.shape[0],coeff.shape[1],3)
        return out.detach().cpu().numpy()
    
    def get_CAD_face(self,affine_transform):
        """
        Get the CAD face from the affine transform.
        
        Args:
            affine_transform (torch.Tensor): Affine transformation matrix.
        
        Returns:
            CAD face object.
        """

        from . export.cad import makeBsplineFace
        control_points = self.get_CAD_coeff(affine_transform)
        U1,U2 = self.Us
        u_order,v_order = self.orders
        u_degree = u_order-1
        v_degree = v_order-1
        return makeBsplineFace(control_points,U1,U2,u_degree,v_degree)
        
    def refine(self):
        """
        Refine the B-spline surface by increasing the number of control points.
        The number of control points is increased by 1 in each direction."""
        Us,coeff = basis_functions.bspline.refine2D(self.Us,self.orders,self.coeff)
        self.Us = Us
        with torch.no_grad():
            xtmp = make_parameter_from_input(coeff.data.detach())
            self.coeff = xtmp

            self.ns[0] = self.coeff.shape[0]
            self.ns[1] = self.coeff.shape[1]
            
    def functional(self,O,coeff,aperture_radius):
        points = O[:,[0,1]]#/(2.0*aperture_radius))+0.5
        x_range = [-aperture_radius,aperture_radius]
        y_range = [-aperture_radius,aperture_radius]

        
        _zsurface = basis_functions.bspline.surface_2d(points, self.Us, self.orders, self.ns, x_range, y_range, coeff)
        return _zsurface-O[:,-1]

        
    def get_functional_param_args(self):
        return [self.coeff,self.aperture_radius]
    
    def explicit(self,local_pos):
        """
        Convert local position to global position using the B-spline surface functional.
        
        Args:
            local_pos (torch.Tensor): Local position in 2D space.
        Returns:
            torch.Tensor: Global position in 3D space.
        """
        if local_pos.shape[-1] != 2:
            raise RuntimeError("local_pos needs to be of shape [:,2]")
        device = local_pos.device
        dtype = local_pos.dtype
        
        x = local_pos[:,0]
        y = local_pos[:,1]

        O_new = torch.zeros((local_pos.shape[0],3),device=device,dtype=dtype)
        O_new[:,0] = x
        O_new[:,1] = y

        return self.functional(O_new,*self.get_functional_param_args())

"""
    def to(self, *args, **kwargs):
        print("bspline to called")
        U1,U2 = self.Us
        self.Us = [U1.to(*args, **kwargs),U2.to(*args, **kwargs)]
        super(Bspline, self).to(*args, **kwargs)
        return self
"""
class Legendre(Surface):

    """
    A class to represent a Legendre surface in 3D space.
    Its kinda work in progress.
    """
    def __init__(self,aperture_radius,degree):
        super().__init__()
    
        self.degree = degree
        self.coeff = make_parameter_from_input(torch.zeros((basis_functions.legendre.get_num_coeff(degree))))
        self.aperture_radius = torch.tensor(aperture_radius)

    def refine(self):
        #TODO move to a parent class
        with torch.no_grad():
            coeff = make_parameter_from_input(torch.zeros((basis_functions.legendre.get_num_coeff(self.degree+1))))
            coeff.data[:self.coeff.shape[0]] = self.coeff.data.detach()
            self.coeff = coeff
            self.degree = self.degree+1
    

    def functional(self,O,coeffs,aperture_radius):
        points = O[:,[0,1]]/aperture_radius
        z = basis_functions.legendre.legendre_2d_basis(self.degree,points[:,0],points[:,1])@coeffs
        
        return z-O[:,-1]
        
    def get_functional_param_args(self):
        return [self.coeff,self.aperture_radius]
    
    def explicit(self,local_pos):
        if local_pos.shape[-1] != 2:
            raise RuntimeError("local_pos needs to be of shape [:,2]")
        device = local_pos.device
        dtype = local_pos.dtype
        
        x = local_pos[:,0]
        y = local_pos[:,1]

        O_new = torch.zeros((local_pos.shape[0],3),device=device,dtype=dtype)
        O_new[:,0] = x
        O_new[:,1] = y

        return self.functional(O_new,*self.get_functional_param_args())




