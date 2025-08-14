# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "smoothed_irradiance",
    "binned_irradiance"
]

import torch
import numpy as np
from typing import List
from ..source import LightSource
from ..element import trace_to_detector,SequentialOpticalSystem,Detector

def smoothed_irradiance(optical_system:SequentialOpticalSystem,
                        sequence:List,
                        source:LightSource,
                        detector:Detector,
                        smoother,
                        num_rays:int,
                        device=torch.get_default_device(),
                        method_ray_tracing:str="sobol_pow2")->torch.Tensor:
    """
    Calculate the smoothed irradiance on the detector using ray tracing.
    
    Args:
        optical_system (SequentialOpticalSystem): The optical system to trace rays through.
        sequence: The sequence of elements in the optical system.
        source (LightSource): The light source used for ray tracing.
        detector (Detector): The detector where the rays are traced to.
        smoother: The smoother object used for smoothing the irradiance.
        num_rays (int, optional): The number of rays to trace. Default is 100000.
        device (torch.device, optional): The device to perform computations on. Default is the default device.
        method_ray_tracing (str, optional): The method used for ray tracing. Default is "sobol".
    
    Returns:
        torch.Tensor: The smoothed irradiance on the detector.
    """
    x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
    Qval = source.get_flux(x.detach())
    smoothed_irradiance = smoother.get_smooth_irradiance(y.detach(),Qval*weights)
    return smoothed_irradiance    

def binned_irradiance(optical_system:SequentialOpticalSystem,
                      sequence:List,
                      source:LightSource,
                      detector:Detector,
                      grid,
                      num_rays:int,
                      device=torch.get_default_device(),
                      method_ray_tracing:str="sobol_pow2")->torch.Tensor:
    """
    Calculate the binned irradiance on the detector using ray tracing.
    
    Args:
        optical_system (SequentialOpticalSystem): The optical system to trace rays through.
        sequence: The sequence of elements in the optical system.
        source (LightSource): The light source used for ray tracing.
        detector (Detector): The detector where the rays are traced to.
        grid: The grid used for binning the irradiance.
        num_rays (int, optional): The number of rays to trace. Default is 100000.
        device (torch.device, optional): The device to perform computations on. Default is the default device.
        method_ray_tracing (str, optional): The method used for ray tracing. Default is "sobol".
    
    Returns:
        torch.Tensor: The binned irradiance on the detector.
    """
    irradiance = None
    with torch.no_grad():
        x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
        Qval = source.get_flux(x.detach())
        irradiance = grid.sum(y,Qval*weights)/grid.get_pixel_area()
    #irradiance = irradiance.reshape(grid.x_)
    return irradiance


"""
def smoothed_RGB(optical_system,sequence,source:LightSource,detector,num_rays,rgb_renderer,device,method_ray_tracing="sobol"):
    x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
    Qval = source.get_flux(x.detach())
    out = rgb_renderer.get_smooth_colour_RGB(y.detach(),Qval*weights,wl)
    return out    
"""
