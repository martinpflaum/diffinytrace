"""
MIT License

Copyright (c) 2025 Martin Pflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


#%%
import torch
import torch.nn as nn
import numpy as np
import math
from .plotting import Plotable
from .integrators import Disc,Cube
from .spectrum import VisibleSunlight_am15g
from .physical_object import PhysicalObject


class RaySource(PhysicalObject):
    """
    The RaySource is a the base class off all Objects emitting rays. It has a sample function which
    samples points. These points are used to initialize the rays with the forward function.
    """
    def __init__(self,transform):
        PhysicalObject.__init__(self)
        self.transform = transform

    def sample(self,num_points,method):
        raise NotImplementedError("sample() not implemented")
        
    def forward(self,x,n_func_enviroment):
        raise NotImplementedError("forward() not implemented")
        #return O1,D1,wl,metadata
    
    def get_plot_points2D(self,resolution):
        print("get_plot_points2D not implemented")
        return []
    
    def get_plot_points3D(self,resolution):
        print("get_plot_points3D not implemented")
        return []
    
    def get_volume(self):
        raise NotImplementedError("area not implemented")

    def get_transform(self):
        return self.transform



class LightSource(RaySource,Plotable):
    """
    """
    def __init__(self,transform,integrator,flux_func=None,total_power=1.0,num_points_normalize=700000,method_normalize="sobol"):
        Plotable.__init__(self,"#f8cecc","#b85450")
        RaySource.__init__(self,transform)
        self.total_power = total_power
        self.integrator = integrator
        self.norm_val = None
        if flux_func is None:
            def t_flux_func(x):
                device = x.device
                dtype = x.dtype
                return torch.ones(x.shape[0],device=device,dtype=dtype)/self.get_volume()
            flux_func = t_flux_func
            self.norm_val = 1.0

        self._flux_func = flux_func

        if self.norm_val is None: 
            points,weights = self.integrator.sample(num_points_normalize,method_normalize)
            self.norm_val = torch.sum(self._flux_func(points)*weights).cpu().item()

    def sample(self,num_points,method):
        return self.integrator.sample(num_points,method)

    def get_flux(self,x):
        return self._flux_func(x)*(self.total_power/self.norm_val)
    
    def get_volume(self):
        return self.integrator.get_volume()

class PlaneSource(LightSource):
    def __init__(self,transform,aperture_radius,integrator,is_square=False,flux_func=None,total_power=1.0,num_points_normalize=700000,method_normalize="sobol"):
        super().__init__(transform,integrator,flux_func,total_power,num_points_normalize,method_normalize)
        self.aperture_radius = aperture_radius
        self.is_square = is_square

    def get_plot_points2D(self,resolution):
        aperture_radius = self.aperture_radius
        x = None
        y = None

        y = torch.linspace(-aperture_radius,aperture_radius,resolution)
        x = torch.zeros_like(y)
        
        tmpx,_ = self.sample(1,"sobol")
        x_integrator = torch.zeros((x.shape[0],tmpx.shape[1]))
        x_integrator[:,0] = x
        x_integrator[:,1] = y
        z = None
        with torch.no_grad():
            O2,D2,wl,_,_ = self(x_integrator,None)

            x = O2[:,0].detach().reshape(-1)
            y = O2[:,1].detach().reshape(-1)
            z = O2[:,2].detach().reshape(-1)
            
        if not self.is_square:
            mul = (torch.sqrt(x*x+y*y)>aperture_radius).float()/torch.sqrt(x*x+y*y)*aperture_radius
            mul += (torch.sqrt(x*x+y*y)<aperture_radius).float()
            x = x*mul
            y = y*mul

        transform=self.transform
        v = torch.zeros((x.shape[0],4))
        v[:,0] = x
        v[:,1] = y
        v[:,2] = z
        v[:,3] = torch.ones_like(v[:,3])   
        
        Mv = None
        with torch.no_grad():
            M = transform.get_transformation_matrix().detach()
            Mv = v@M.T
        x = Mv[:,0].reshape(-1)
        y = Mv[:,1].reshape(-1)
        z = Mv[:,2].reshape(-1)
        return [(z,y)]            

    def get_plot_points3D(self,resolution):
        aperture_radius = self.aperture_radius
        x = None
        y = None

        _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
        _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
            
        tmpx,_ = self.sample(1,"sobol")
        x_integrator = torch.zeros((x.shape[0],tmpx.shape[1]))
        x_integrator[:,0] = x
        x_integrator[:,1] = y
        z = None
        with torch.no_grad():
            O2,D2,wl,_,_ = self(x_integrator,None)

            x = O2[:,0].detach().reshape(-1)
            y = O2[:,1].detach().reshape(-1)
            z = O2[:,2].detach().reshape(-1)
            
        if not self.is_square:
            mul = (torch.sqrt(x*x+y*y)>aperture_radius).float()/torch.sqrt(x*x+y*y)*aperture_radius
            mul += (torch.sqrt(x*x+y*y)<aperture_radius).float()
            x = x*mul
            y = y*mul

        transform=self.transform
        v = torch.zeros((x.shape[0],4))
        v[:,0] = x
        v[:,1] = y
        v[:,2] = z
        v[:,3] = torch.ones_like(v[:,3])   
        
        Mv = None
        with torch.no_grad():
            M = transform.get_transformation_matrix().detach()
            Mv = v@M.T
        x = Mv[:,0].reshape(resolution,resolution)
        y = Mv[:,1].reshape(resolution,resolution)
        z = Mv[:,2].reshape(resolution,resolution)
        return [(x,y,z)]

    
class PlaneSource1D(LightSource):
    def __init__(self,transform,aperture_radius,integrator,flux_func,total_power=1.0,num_points_normalize=700000,method_normalize="sobol"):
        super().__init__(transform,integrator,flux_func,total_power,num_points_normalize,method_normalize)
        self.aperture_radius = aperture_radius
    
    def get_plot_points2D(self,resolution):
        aperture_radius = self.aperture_radius
        
        transform=self.transform
        v = torch.zeros((resolution,4))
        v[:,1] = torch.linspace(-aperture_radius,aperture_radius,resolution)
        v[:,3] = torch.ones_like(v[:,3])   
        
        Mv = None
        with torch.no_grad():
            M = transform.get_transformation_matrix().detach()
            Mv = v@M.T
        
        x = Mv[:,0].reshape(-1)
        y = Mv[:,1].reshape(-1)
        z = Mv[:,2].reshape(-1)
        return [(z,y)] 
    
    def get_plot_points3D(self,resolution):
        raise RuntimeError("get_plot_points3D is not implemented for PlaneSource1D")

def make_cone_directions(num_rays, unif1, unif2, theta_max_rad):
    """
    Sample directions uniformly within a cone of angular radius `theta_max_rad` centered on the z-axis.

    Parameters:
    - num_rays (int): Number of direction vectors to sample within the cone.
    - unif1 (torch.Tensor): Tensor of uniform samples for cos(theta) sampling.
    - unif2 (torch.Tensor): Tensor of uniform samples for the azimuthal angle sampling.
    - theta_max_rad (float): Angular radius of the cone in radians.

    Returns:
    - directions (torch.Tensor): Tensor of shape (num_rays, 3), with each row a direction vector.
    """
    # Ensure uniform samples are tensors and match the number of rays
    if unif1.shape[0] != num_rays or unif2.shape[0] != num_rays:
        raise ValueError("Length of unif1 and unif2 must match num_rays.")
    theta_max_rad = torch.tensor(theta_max_rad) 
    # Uniform sampling of cos(theta) within the cone angle
    thetas = -theta_max_rad + unif1 * (theta_max_rad*2.0)
    
    # Uniform sampling of azimuthal angle phi
    phi = (2.0 * torch.pi) * unif2

    # Spherical to Cartesian conversion (cone aligned with z-axis)
    x = torch.sin(thetas) * torch.cos(phi)
    y = torch.sin(thetas) * torch.sin(phi)
    z = torch.cos(thetas)

    # Combine into a directions tensor
    directions = torch.stack((x, y, z), dim=1)
    
    return directions


"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num_rays = 20
directions = sample_cone_directions(num_rays)

# Plotting the sampled directions as lines in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot lines from the origin to each sampled direction
for direction in directions:
    ax.plot([0, direction[0]], [0, direction[1]], [0, direction[2]], color='blue', alpha=0.6)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Sampled Directions in a 4.65 mrad Cone Centered Around the Z-axis")
ax.view_init(elev=20, azim=30)  # Adjust viewing angle for better visualization
plt.show()


"""
class VisibleSunlightSimpleMonochromatic(PlaneSource):
    def __init__(self,transform,aperture_radius,wl=0.5,is_square=True,total_power=1.0):
        """
        cube1 = Cube([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius]])
        cube2 = Cube([[0.,1.],[0.,1.]])

        integrator = Compose([cube1,cube2])
        
        """
        
        integrator = Cube([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],[0.,1.],[0.,1.]])
        self.theta_max_rad = 4.65/1000.
        self.wl = wl
        PlaneSource.__init__(self,transform,aperture_radius,integrator,is_square,None,total_power)
        
    def sample(self,num_points,method="monte_carlo"):
        if not ((method == "sobol_pow2")or (method == "sobol") or (method == "monte_carlo")):
            raise RuntimeError("Only sobol_pow2,sobol or monte_carlo sampling supported for VisibleSunlightSimpleMonochromatic")
        return self.integrator.sample(num_points,method)
        
    def forward(self,x,n_func_enviroment):
        N = x.shape[0]
        device = x.device
        dtype = x.dtype
        O1 = torch.zeros(N,3,device=device,dtype=dtype)
        O1[:,[0,1]] = x[:,[0,1]]
        O1[:,-1] = 0.0
        D1 = make_cone_directions(N, x[:,2], x[:,3], self.theta_max_rad)
        #print(D1)
        wl = torch.ones((N),device=device,dtype=dtype)*self.wl 
        #self.spectrum(x[4])
        
        #TODO inconsitent with plotting!!S
        O2 = self.transform.to_global_pos(O1)
        D2 = self.transform.to_global_dir(D1)
        
        ray_paths = [O2.detach()]

        valid = self.integrator.in_bounds(x)
        PL,OPL = 0.0,0.0
        meta_data = {}
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data


        
class VisibleSunlightSimple(PlaneSource):
    def __init__(self,transform,aperture_radius,is_square=True,total_power=1.0):
        self.spectrum = VisibleSunlight_am15g()
        self.aperture_radius = aperture_radius
        integrator = Cube([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],[0.,1.],[0.,1.],self.spectrum.bounds])
        PlaneSource.__init__(self,transform,aperture_radius,integrator,is_square,None,total_power)
        self.theta_max_mrad = 4.65
        

    def sample(self,num_points,method="monte_carlo"):
        if not ((method == "sobol_pow2")or (method == "sobol") or (method == "monte_carlo")):
            raise RuntimeError("Only sobol_pow2,sobol or monte_carlo sampling supported for VisibleSunlightSimple")
        return self.integrator.sample(num_points,method)


    def forward(self,x,n_func_enviroment):    
        N = x.shape[0]
        device = x.device
        dtype = x.dtype
        O1 = torch.zeros(N,3,device=device,dtype=dtype)
        O1[:,[0,1]] = x[:,[0,1]]
        O1[:,-1] = 0.0

        D1 = make_cone_directions(N, x[:,2], x[:,3], self.theta_max_mrad)
        wl = self.spectrum.bounds[0]+(self.spectrum.bounds[1]-self.spectrum.bounds[0])*x[:,4] 
        #self.spectrum(x[4])
        
        #TODO inconsitent with plotting!!S
        O2 = self.transform.to_global_pos(O1)
        D2 = self.transform.to_global_dir(D1)
        
        ray_paths = [O2.detach()]

        valid = self.integrator.in_bounds(x)
        PL,OPL = 0.0,0.0
        meta_data = {}
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

     
class CollimatedMonochromatic(PlaneSource):
    def __init__(self,transform,aperture_radius,wl,is_square=False,flux_func=None,total_power=1.0):
        integrator = None
        aperture_radius = abs(aperture_radius)
        if is_square:
            integrator = Cube([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius]])
        else:
            integrator = Disc(aperture_radius)
        super().__init__(transform,aperture_radius,integrator,is_square,flux_func,total_power)
        self.wl = wl
        

    def sample(self,num_points,method="monte_carlo"):
        return self.integrator.sample(num_points,method)

    def forward(self,x,n_func_enviroment):    
        N = x.shape[0]
        device = x.device
        dtype = x.dtype
        O1 = torch.zeros(N,3,device=device,dtype=dtype)
        O1[:,[0,1]] = x
        O1[:,-1] = 0.0

        D1 = torch.zeros_like(O1)
        D1[:,-1] = 1.0
        PL,OPL = 0.0,0.0
        #TODO inconsitent with plotting!!S
        O2 = self.transform.to_global_pos(O1)
        D2 = self.transform.to_global_dir(D1)
        
        ray_paths = [O2.detach()]

        valid = self.integrator.in_bounds(x)

        wl = torch.ones(N,device=device,dtype=dtype)*self.wl
        meta_data = {}
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data


class CollimatedGaussianBeam(CollimatedMonochromatic):
    def __init__(self,transform,aperture_radius,wl,gaussian_constant,total_power=1.0):
        def flux_func(x):
            rho = torch.norm(x,dim=1)
            return torch.exp(-gaussian_constant*(rho**2.0))
        super().__init__(transform,aperture_radius,wl,is_square=False,flux_func=flux_func,total_power=total_power)

class CollimatedMonochromatic1D(PlaneSource1D):
    def __init__(self,transform,aperture_radius,wl,flux_func=None,total_power=1.0):
        self.wl = wl
        integrator = Cube([[-aperture_radius,aperture_radius]])
        super().__init__(transform,aperture_radius,integrator,flux_func,total_power)
     
    def sample(self,num_points,method="monte_carlo"):
        return self.integrator.sample(num_points,method)

    def forward(self,x,n_func_enviroment):    
        N = x.shape[0]
        device = x.device
        dtype = x.dtype
        O1 = torch.zeros(N,3,device=device,dtype=dtype)
        O1[:,1] = x[:,0]
        O1[:,-1] = 0.0

        D1 = torch.zeros_like(O1)
        D1[:,-1] = 1.0
        PL,OPL = 0.0,0.0

        O2 = self.transform.to_global_pos(O1)
        D2 = self.transform.to_global_dir(D1)
        
        ray_paths = [O2.detach()]
        valid = self.integrator.in_bounds(x)
        wl = torch.ones(N,device=device,dtype=dtype)*self.wl

        meta_data = {}
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

    
    
        