import torch
import math
import numpy as np
from .core import Smoother
from ...integrators import Cube
import torch
import colour

from .gaussian_smoother import GaussianSmoother,gaussian_func2D

class GaussianXYZSmoother(Smoother):
    def __init__(self,colour_channel,num_integration_points,v_range,h_range,v_num_conv_points:int,h_num_conv_points:int,sigma:float,desired_irradiance_func=None):
        self.colour_idx = colour_channel #could be a list ["x","y"],"x"
        self.smoother = GaussianSmoother(num_integration_points,v_range,h_range,v_num_conv_points,h_num_conv_points,sigma,desired_irradiance_func)
        
    def get_smooth_colour_XYZ(self,points,ray_multi,wl):
        device = points.device
        dtype = points.dtype
        xyz = colour.wavelength_to_XYZ(wl.detach().cpu().numpy())
        xyz = xyz.T
        val = xyz[self.colour_idx]
        val = torch.tensor(val,device=device,dtype=dtype)
        
        out = self.smoother.get_smooth_irradiance(points,ray_multi*val)
        return out
    
    def get_smooth_irradiance(self,points,ray_multi):
        return gaussian_func2D(points,self.v_range,self.h_range,self.v_num_conv_points,self.h_num_conv_points,self.sigma,ray_multi)

    def get_merit_function_value(self,points,ray_multi,wl):
        device = points.device
        if self.smooth_desired_irradiance is None:
                self.calc_smooth_desired_irradiance()
        if device != self.smooth_desired_irradiance.device:
            self.smooth_desired_irradiance = self.smooth_desired_irradiance.to(device)
        if device != self.weights.device:
            self.weights = self.weights.to(device)
        smoothed_irradiance = self.get_smooth_colour_XYZ(points,ray_multi,wl)
        
        #TODO change to integral
        smoothed_irradiance = smoothed_irradiance/smoothed_irradiance.sum()
        
        residual = self.smooth_desired_irradiance-smoothed_irradiance
        residual = residual.reshape(-1)

        integral = torch.sum(self.weights*(residual**2.0))

        
        return torch.sqrt(integral)#torch.linalg.norm(residual)
    

class GaussianXYZSmootherSquare(GaussianXYZSmoother):
    def __init__(self,num_integration_points,aperture_radius,num_conv_points:int,sigma:float,desired_irradiance_func=None):
        super().__init__(num_integration_points,[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],num_conv_points,num_conv_points,sigma,desired_irradiance_func)
        
    