# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.



#%%
import torch
import torch.nn as nn
import numpy as np
#from diffinytrace.basis_functions import *
from .transforms import SemiFunctionalModule
from .optimize import make_parameter_from_input
from . import basis_functions

class Surface(SemiFunctionalModule):
    """
    The functional should be 0 at O=(0,0,0)!
    maybe default mode and another mode?
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
    """
    THis From diffoptics
    This is the aspheric surface class, implementation follows: https://en.wikipedia.org/wiki/Aspheric_lens.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    r**2=x**2+y**2??
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2/R**2 )) + ai[0] * r**4 + ai[1] * r**6 + \cdots.
    h(z) = -z.
    
    Args (new attributes):
        c: Surface curvature, or one over radius of curvature.
        k: Conic coefficient. 
        ai: Aspheric parameters, could be a vector. When None, the surface is spherical.
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
        from . export.cad import makeBsplineFace
        control_points = self.get_CAD_coeff(affine_transform)
        U1,U2 = self.Us
        u_order,v_order = self.orders
        u_degree = u_order-1
        v_degree = v_order-1
        return makeBsplineFace(control_points,U1,U2,u_degree,v_degree)
        
    def refine(self):
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




