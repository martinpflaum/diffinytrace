import torch
import numpy as np
from ..source import LightSource
from ..element import trace_to_detector,SequentialOpticalSystem,Detector
def smoothed_irradiance(optical_system:SequentialOpticalSystem,sequence,source:LightSource,detector:Detector,smoother,num_rays=100000,device=torch.get_default_device(),method_ray_tracing="sobol")->torch.Tensor:
    x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
    Qval = source.get_flux(x.detach())
    smoothed_irradiance = smoother.get_smooth_irradiance(y.detach(),Qval*weights)
    return smoothed_irradiance    

def binned_irradiance(optical_system:SequentialOpticalSystem,sequence,source:LightSource,detector:Detector,grid,num_rays=100000,device=torch.get_default_device(),method_ray_tracing="sobol")->torch.Tensor:
    x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
    Qval = source.get_flux(x.detach())
    irradiance = grid.sum(y,Qval*weights)/grid.get_pixel_area()
    return irradiance


"""
def smoothed_RGB(optical_system,sequence,source:LightSource,detector,num_rays,rgb_renderer,device,method_ray_tracing="sobol"):
    x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
    Qval = source.get_flux(x.detach())
    out = rgb_renderer.get_smooth_colour_RGB(y.detach(),Qval*weights,wl)
    return out    
"""
