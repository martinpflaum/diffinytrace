# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


__all__ = [
    "gaussian_func1D",
    "gaussian_func2D",
    "calc_smooth_desired_irradiance",
    "GaussianSmoother",
    "make_evaluation_function",
    "make_merit_function",
    "GaussianSmootherSquare"
]

import torch
import math
import numpy as np
from .integrators import Cube
import torch
from typing import Callable,List,Tuple,Optional
import gc
import torch
import numpy as np
from .element import trace_to_detector,SequentialOpticalSystem
import math
from .source import LightSource
from .target_grid import Grid
import gc
import warnings
from .render import binned_irradiance

def gaussian_func1D(eval_points:torch.Tensor,
                    x_range,
                    num_gauss_points:int,
                    sigma:float,
                    include_boundary=False)->torch.Tensor:
    """
    Gaussian function for 1D convolution.
    
    Args:
        eval_points (torch.Tensor): Points where the Gaussian function is evaluated.
        x_range (tuple): Range of the target plane.
        num_gauss_points (int): Number of Gaussian points.
        sigma (float): Standard deviation of the Gaussian function.
        include_boundary (bool): Whether to include the boundary points.
        
    Returns:
        torch.Tensor: Evaluated Gaussian function.
    """
    device = eval_points.device
    dtype = eval_points.dtype
    
    eval_points = eval_points.reshape(-1)
    xgrid = None
    if include_boundary:
        xgrid = torch.linspace(x_range[0],x_range[1],num_gauss_points,dtype=dtype,device=device)
    else:
        xgrid = torch.linspace(x_range[0],x_range[1],num_gauss_points+1,dtype=dtype,device=device)

        dxgrid = xgrid[1]-xgrid[0]
        xgrid = xgrid[:-1]
        xgrid = xgrid + dxgrid*0.5
    dist = (xgrid.reshape(-1,1)-eval_points.reshape(1,-1))
    
    const = 1.0/math.sqrt((2.0*math.pi)*sigma*sigma)
    multiplier = const
    out = multiplier*torch.exp(-(dist**2.0/(2.0*(sigma**2.0))))
    return out

def gaussian_func2D(eval_points:torch.Tensor,
                                  x_range,
                                  y_range,
                                  x_grid_size:int,
                                  y_grid_size:int,
                                  sigma:float|torch.Tensor,
                                  val_multi:torch.Tensor|None=None,
                                  summed:bool=True,
                                  include_boundary=False)->torch.Tensor:
    """
    Gaussian function for 2D convolution.
    
    Args:
        eval_points (torch.Tensor): Points where the Gaussian function is evaluated.
        y_range (tuple): Range of the target plane in the vertical direction.
        x_range (tuple): Range of the target plane in the horizontal direction.
        y_grid_size (int): Number of Gaussian points in the vertical direction.
        x_grid_size (int): Number of Gaussian points in the horizontal direction.
        sigma (float): Standard deviation of the Gaussian function.
        val_multi (torch.Tensor|None): Optional multiplier for the Gaussian function.
        summed (bool): Whether to sum the Gaussian function.
        include_boundary (bool): Whether to include the boundary points.

    Returns:
        torch.Tensor: Evaluated Gaussian function.
    """
    
    if eval_points.shape[-1] != 2:
        raise RuntimeError("points need to be in local coordinates and shape [numraysx2]")
    eval_points1 = eval_points[:,1]
    eval_points2 = eval_points[:,0]
    
    out1 = gaussian_func1D(eval_points1,y_range,y_grid_size,sigma,include_boundary)
    out2 = gaussian_func1D(eval_points2,x_range,x_grid_size,sigma,include_boundary)
    
    if not val_multi is None:
        out1 = out1*val_multi.reshape(1,-1)
    if summed is False:

        print("summed=False should only be used when debugging!! Its very very slow.")
        out = out1.reshape(y_grid_size,1,-1)*out2.reshape(1,x_grid_size,-1)
        return out
    return out1@out2.T

def calc_smooth_desired_irradiance(desired_irradiance_fun:Callable,
                                   x_range:List[float],
                                   y_range:List[float],
                                   x_grid_size:int,
                                   y_grid_size:int,
                                   sigma:float,
                                   num_integration_points:int,
                                   num_splits=5,
                                   dtype=torch.get_default_dtype(),
                                   device=torch.get_default_device())->torch.Tensor:
    """
    Calculates the smoothed desired irradiance using Gaussian convolution.

    Args:
        desired_irradiance_fun (Callable): Function that computes the desired irradiance at given points.
        x_range (List[float]): Range of the target plane in the x direction [min, max].
        y_range (List[float]): Range of the target plane in the y direction [min, max].
        x_grid_size (int): Number of pixels in the x direction.
        y_grid_size (int): Number of pixels in the y direction.
        sigma (float): Standard deviation of the Gaussian kernel.
        num_integration_points (int): Number of integration points for numerical integration.
        num_splits (int, optional): Number of splits for integration to reduce memory usage. Defaults to 5.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.get_default_dtype().
        device (torch.device, optional): Device for computation. Defaults to torch.get_default_device().

    Returns:
        torch.Tensor: Smoothed desired irradiance map.
    """
    gc.collect()
    integrator = Cube([x_range,y_range])
    points,weights = integrator.sample(num_integration_points,"sobol_pow2")
        


    splitted_points = torch.split(points, num_integration_points // num_splits)
    splitted_weights = torch.split(weights, num_integration_points // num_splits)

    with torch.no_grad():
        out = []
        for k in range(num_splits):
            split_points = splitted_points[k].to(device=device,dtype=dtype)
            split_weights = splitted_weights[k].to(device=device,dtype=dtype)
            tmp = gaussian_func2D(split_points,x_range,y_range,x_grid_size,y_grid_size,sigma=sigma,val_multi=desired_irradiance_fun(split_points)*split_weights)
            #print("tmp.shape",tmp.shape)
            out.append(tmp)
            del split_points, split_weights, tmp

    out = torch.mean(torch.stack(out), dim=0)
    del points, weights, splitted_points, splitted_weights
    gc.collect()
    return out
    


class GaussianSmoother():
    r"""
    GaussianSmoother applies Gaussian convolution to smooth irradiance maps.

    This class provides methods for smoothing irradiance data using a Gaussian kernel and integrating values over a grid.

    Args:
        x_range (list): Range of the target plane in the x direction [min, max].
        y_range (list): Range of the target plane in the y direction [min, max].
        x_grid_size (int): Number of pixels in the x direction.
        y_grid_size (int): Number of pixels in the y direction.
        sigma (float): Standard deviation of the Gaussian kernel.
        desired_irradiance_fun (Callable): Function that computes the desired irradiance at given points.
        smoothed_num_integration_points (int): Number of integration points for smoothing.
        smoothed_num_splits (int): Number of splits for integration to reduce memory usage.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.get_default_dtype().
        device (torch.device, optional): Device for computation. Defaults to torch.get_default_device().

    Attributes:
        x_grid_size (int): Number of pixels in the x direction.
        y_grid_size (int): Number of pixels in the y direction.
        sigma (float): Standard deviation of the Gaussian kernel.
        include_boundary (bool): Whether to include boundary points in the grid.
        x_range (list): Range of the target plane in the x direction.
        y_range (list): Range of the target plane in the y direction.
        grid (Grid): Grid object for pixel centers.
        discrete_desired_irradiance (torch.Tensor): Desired irradiance at pixel centers.
        smoothed_desired_irradiance (torch.Tensor): Smoothed desired irradiance map.
    """

    def __init__(self,
                 x_range:list,
                 y_range:list,
                 x_grid_size:int,
                 y_grid_size:int,
                 sigma:float,
                 desired_irradiance_fun:Callable,
                 smoothed_num_integration_points:int,
                 smoothed_num_splits:int,
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device()):
        self.x_grid_size,self.y_grid_size = x_grid_size,y_grid_size
        self.sigma = sigma
        self.include_boundary = False
        self.x_range,self.y_range = x_range,y_range

        self.grid = Grid(x_range,y_range,x_grid_size,y_grid_size)
        centers = self.grid.get_pixel_centers().reshape(-1,2)
        self.discrete_desired_irradiance:torch.Tensor = desired_irradiance_fun(centers)
        
        
    
        self.smoothed_desired_irradiance:torch.Tensor = calc_smooth_desired_irradiance(desired_irradiance_fun,
                                   x_range,y_range,
                                   x_grid_size,
                                   y_grid_size,
                                   sigma=sigma,
                                   num_integration_points=smoothed_num_integration_points,
                                   num_splits=smoothed_num_splits,
                                   dtype=dtype,
                                   device=device)
    
    def smooth_irradiance(self,points:torch.Tensor,ray_multi:torch.Tensor)->torch.Tensor:
        """
        Computes the smoothed irradiance at given points using a Gaussian kernel.

        Args:
            points (torch.Tensor): Array of points where the irradiance is evaluated, shape [N, 2].
            ray_multi (torch.Tensor): Multiplicative weights for each point, e.g., ray flux.

        Returns:
            torch.Tensor: Smoothed irradiance values at the specified points.
        """
        return gaussian_func2D(points,self.x_range,self.y_range,self.x_grid_size,self.y_grid_size,self.sigma,ray_multi,include_boundary=self.include_boundary)

    def none_smooth_irradiance(self,points:torch.Tensor,ray_multi:torch.Tensor)->torch.Tensor:
        """
        Computes the non-smoothed irradiance at given points by summing ray contributions in each grid cell.

        Args:
            points (torch.Tensor): Array of points where the irradiance is evaluated, shape [N, 2].
            ray_multi (torch.Tensor): Multiplicative weights for each point, e.g., ray flux.

        Returns:
            torch.Tensor: Non-smoothed irradiance values at the specified grid cells.
        """
        irradiance = self.grid.sum(points,ray_multi)/self.grid.get_pixel_area()
        return irradiance

    def integrate_values(self, vals:torch.Tensor)->torch.Tensor:
        """
        Integrates the provided values over the grid using midpoint rule.

        Args:
            vals (torch.Tensor): Values to integrate, typically irradiance or residuals, shape matching the grid.

        Returns:
            torch.Tensor: The integrated sum over the grid.
        """
        integrator = Cube([self.x_range, self.y_range])
        _, weights = integrator.sample([self.x_grid_size, self.y_grid_size], "midpoint")
        weights = weights.to(device=vals.device, dtype=vals.dtype)
        return (vals * weights).sum()


class GaussianSmootherSquare(GaussianSmoother):
    r"""
    GaussianSmootherSquare applies Gaussian smoothing to square grids.

    This class is a specialized version of GaussianSmoother for cases where the x and y ranges are identical,
    and the grid is square (same number of pixels in both directions).

    Args:
        x_range (list): Range of the target plane in both x and y directions [min, max].
        x_grid_size (int): Number of pixels in both x and y directions.
        sigma (float): Standard deviation of the Gaussian kernel.
        desired_irradiance_fun (Callable): Function that computes the desired irradiance at given points.
        smoothed_num_integration_points (int): Number of integration points for smoothing.
        smoothed_num_splits (int): Number of splits for integration to reduce memory usage.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.get_default_dtype().
        device (torch.device, optional): Device for computation. Defaults to torch.get_default_device().
    """
    def __init__(self,
                 aperture_radius:list,
                 grid_size:int,
                 sigma:float,
                 desired_irradiance_fun:Callable,
                 smoothed_num_integration_points:int,
                 smoothed_num_splits:int,
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device()):
        
        super().__init__(x_range=[-aperture_radius,aperture_radius],y_range=[-aperture_radius,aperture_radius],
                         x_grid_size=grid_size,
                         y_grid_size=grid_size,
                         sigma=sigma,
                         desired_irradiance_fun=desired_irradiance_fun,
                         smoothed_num_integration_points=smoothed_num_integration_points,
                         smoothed_num_splits=smoothed_num_splits,
                         dtype=dtype,
                         device=device)

def make_evaluation_function(optical_system:SequentialOpticalSystem,
                        sequence:List,
                        source:LightSource,
                        detector,
                        smoother:GaussianSmoother,
                        num_splits:int=10,
                        num_rays_per_split:int=1000000,
                        method_ray_tracing="monte_carlo",
                        device=torch.get_default_device())->Callable:
    """
    Creates an evaluation function for comparing simulated and desired irradiance.

    Args:
        optical_system (SequentialOpticalSystem): The optical system to be used for ray tracing.
        sequence: The sequence of optical elements.
        source (LightSource): The light source for the simulation.
        detector: The detector object.
        smoother (GaussianSmoother): Smoother object for irradiance comparison.
        num_splits (int, optional): Number of splits for ray tracing to reduce memory usage. Defaults to 10.
        num_rays_per_split (int, optional): Number of rays per split. Defaults to 1,000,000.
        method_ray_tracing (str, optional): Ray tracing method ('monte_carlo', etc.). Defaults to "monte_carlo".
        device (torch.device, optional): Device for computation. Defaults to torch.get_default_device().

    Returns:
        Callable: A function that computes the L2 error between simulated and desired irradiance.
    """
    def evaluate():
        raycounting_list = []
        for k in (range(num_splits)):
            tmp = binned_irradiance(optical_system=optical_system,sequence=sequence,source=source,detector=detector,grid=smoother.grid,num_rays=num_rays_per_split,method_ray_tracing=method_ray_tracing,device=device)
            tmp = tmp.detach().cpu()
            raycounting_list.append(tmp)
        raycounting = torch.mean(torch.stack(raycounting_list),dim=0).detach().cpu()
        
        smoother.last_raycounting = raycounting
        residual = raycounting.reshape(-1)-smoother.discrete_desired_irradiance.reshape(-1)
        
        L2_error = torch.sqrt(smoother.integrate_values(residual**2))
        #RMSE = torch.sum((residual**2))
        return L2_error
    return evaluate

def make_merit_function(optical_system:SequentialOpticalSystem,
                        sequence:List,
                        source:LightSource,
                        detector,
                        smoother:GaussianSmoother,
                        num_rays:int,
                        method_ray_tracing="sobol_pow2",
                        use_desired_irradiance_smoothing=True,
                        device=torch.get_default_device())->Callable:
    """
    Creates a merit function for the given optical system, source, and detector.
    
    Args:
        optical_system (SequentialOpticalSystem): The optical system to be used.
        sequence: The sequence of elements in the optical system.
        source (LightSource): The light source to be used.
        detector: The detector to be used.
        num_rays (int): Number of rays to be traced.
        smoother (Smoother): The smoother object for merit function calculation.
        device: The device to be used for calculations.
        method_ray_tracing (str): Method for ray tracing ('sobol' or 'midpoint').
        use_desired_irradiance_smoothing (bool): Whether to use desired irradiance smoothing.
        use_power_correction (bool): Whether to use power correction.
        save_last_eval (bool): Whether to save the last evaluation.
    
    Returns:
        Callable: A function that computes the merit value.
    """

    def merit_function()->torch.Tensor:
        """
        """
        x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
        Qval = source.get_flux(x)
        
        #print("total energy rays:",(Qval*weights).sum())
        
        if smoother.smoothed_desired_irradiance is None and use_desired_irradiance_smoothing == True:
            raise RuntimeError("Using desired irradiance smoothing but smoothed_desired_irradiance was not provided!--calc_smooth_desired_irradiance")
        
        smooth_irradiance = smoother.smooth_irradiance(y,Qval*weights)
        residual = None

        if use_desired_irradiance_smoothing:
            residual = smooth_irradiance.reshape(-1)-smoother.smoothed_desired_irradiance.reshape(-1).to(device=device)
        else:
            residual = smooth_irradiance.reshape(-1)-smoother.discrete_desired_irradiance.reshape(-1).to(device=device)

        return torch.sqrt(smoother.integrate_values(residual**2))


    return merit_function