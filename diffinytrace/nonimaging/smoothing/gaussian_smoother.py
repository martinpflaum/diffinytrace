# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import torch
import math
import numpy as np
from .core import Smoother
from ...integrators import Cube
import torch


def gaussian_func1D(eval_points:torch.Tensor,p_range,num_gauss_points:int,sigma:float,include_boundary=True)->torch.Tensor:
    """
    Gaussian function for 1D convolution.
    
    Args:
        eval_points (torch.Tensor): Points where the Gaussian function is evaluated.
        p_range (tuple): Range of the target plane.
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
        xgrid = torch.linspace(p_range[0],p_range[1],num_gauss_points,dtype=dtype,device=device)
    else:
        xgrid = torch.linspace(p_range[0],p_range[1],num_gauss_points+1,dtype=dtype,device=device)

        dxgrid = xgrid[1]-xgrid[0]
        xgrid = xgrid[:-1]
        xgrid = xgrid + dxgrid*0.5
    dist = (xgrid.reshape(-1,1)-eval_points.reshape(1,-1))
    
    const = 1.0/math.sqrt((2.0*math.pi)*sigma*sigma)
    multiplier = const
    out = multiplier*torch.exp(-(dist**2.0/(2.0*(sigma**2.0))))
    return out


def gaussian_func2D(eval_points:torch.Tensor,
                                  v_range,
                                  h_range,
                                  v_num_conv_points:int,
                                  h_num_conv_points:int,
                                  sigma:float|torch.Tensor,
                                  val_multi:torch.Tensor|None=None,
                                  summed:bool=True,
                                  include_boundary=True)->torch.Tensor:
    """
    Gaussian function for 2D convolution.
    
    Args:
        eval_points (torch.Tensor): Points where the Gaussian function is evaluated.
        v_range (tuple): Range of the target plane in the vertical direction.
        h_range (tuple): Range of the target plane in the horizontal direction.
        v_num_conv_points (int): Number of Gaussian points in the vertical direction.
        h_num_conv_points (int): Number of Gaussian points in the horizontal direction.
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
    
    out1 = gaussian_func1D(eval_points1,v_range,v_num_conv_points,sigma,include_boundary)
    out2 = gaussian_func1D(eval_points2,h_range,h_num_conv_points,sigma,include_boundary)
    
    if not val_multi is None:
        out1 = out1*val_multi.reshape(1,-1)
    if summed is False:

        print("summed=False should only be used when debugging!! Its very very slow.")
        out = out1.reshape(v_num_conv_points,1,-1)*out2.reshape(1,h_num_conv_points,-1)
        return out
    return out1@out2.T
"""
def get_smooth_desired_irradiance(irradiance_func,
                              num_integration_points:int,
                                  v_range,
                                  h_range,
                                  v_num_conv_points:int,
                                  h_num_conv_points:int,
                                  sigma:float|torch.Tensor)->torch.Tensor:
    
    
    cube = Cube(torch.tensor([v_range,h_range]))

    num_integration_points = int(math.sqrt(num_integration_points))
    if num_integration_points % 2 == 0:
        num_integration_points += 1
    eval_points,weights = cube.sample([num_integration_points,num_integration_points],"sobol")
    
    

    out1 = gaussian_func1D(eval_points[:,0],v_range,v_num_conv_points,sigma)
    out2 = gaussian_func1D(eval_points[:,1],h_range,h_num_conv_points,sigma)

    
    val_multi = irradiance_func(eval_points)*weights
    
    out1 = out1*val_multi.reshape(1,-1)    
    return out1@out2.T
"""


class GaussianSmoother(Smoother):
    def __init__(self,\
                v_range,\
                h_range,\
                v_num_conv_points:int,\
                h_num_conv_points:int,\
                sigma:float,\
                device:torch.device=torch.get_default_device(),\
                dtype:torch.dtype=torch.get_default_dtype(),\
                num_integration_points_desired=2**20,\
                desired_irradiance_func=None,\
                residual_integration_method="midpoint",\
                total_power_desired=1.0,
                v_num_eval_points:int=64,
                h_num_eval_points:int=64,
                use_eval_avg=False):
        """
        Initialize the GaussianSmoother object.

        Args:
            v_range (list): Range of the target plane in the vertical direction.
            h_range (list): Range of the target plane in the horizontal direction.
            v_num_conv_points (int): Number of Gaussian points in the vertical direction.
            h_num_conv_points (int): Number of Gaussian points in the horizontal direction.
            sigma (float): Standard deviation of the Gaussian function.
            device (torch.device, optional): Device to perform computations on. Default is the default device.
            dtype (torch.dtype, optional): Data type for computations. Default is the default data type.
            num_integration_points_desired (list, optional): Number of integration points desired. Default is [701, 701].
            desired_irradiance_func (callable, optional): Function to compute desired irradiance. Default is None.
            residual_integration_method (str, optional): Method for residual integration. Default is "midpoint".
            total_power_desired (float, optional): Desired total power. Default is 1.0.
            v_num_eval_points (int, optional): Number of evaluation points in the vertical direction. Default is 64.
            h_num_eval_points (int, optional): Number of evaluation points in the horizontal direction. Default is 64.
            use_eval_avg (bool, optional): Whether to use average evaluation. Default is True.
        """

        self._sigma = sigma
        super().__init__(v_range,h_range,
                         v_num_conv_points,h_num_conv_points,
                         device,dtype,
                         num_integration_points_desired,desired_irradiance_func,
                         residual_integration_method,total_power_desired,
                         v_num_eval_points = v_num_eval_points,
                         h_num_eval_points = h_num_eval_points,
                         use_eval_avg=use_eval_avg)

    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        if self._sigma == value:
            return
        self._sigma = value
        self.desired_smooth_irradiance = None
        
    def get_smooth_irradiance(self,points,ray_multi):
        return gaussian_func2D(points,self.v_range,self.h_range,self.v_num_conv_points,self.h_num_conv_points,self.sigma,ray_multi,include_boundary=self.include_boundary)
   
class GaussianSmootherSquare(GaussianSmoother):
    def __init__(self,\
                aperture_radius,\
                num_conv_points:int,\
                sigma:float,\
                device=torch.get_default_device(),\
                dtype=torch.get_default_dtype(),\
                num_integration_points_desired=2**20,\
                desired_irradiance_func=None,\
                residual_integration_method="midpoint",\
                total_power_desired=1.0,
                num_eval_points = 64,
                use_eval_avg=False):
        """
        Initialize the GaussianSmootherSquare object.
        
        Args:
            aperture_radius (float): Radius of the square aperture.
            num_conv_points (int): Number of Gaussian points.
            sigma (float): Standard deviation of the Gaussian function.
            device (torch.device, optional): Device to perform computations on. Default is the default device.
            dtype (torch.dtype, optional): Data type for computations. Default is the default data type.
            num_integration_points_desired (list, optional): Number of integration points desired. Default is [701, 701].
            desired_irradiance_func (callable, optional): Function to compute desired irradiance. Default is None.
            residual_integration_method (str, optional): Method for residual integration. Default is "midpoint".
            total_power_desired (float, optional): Desired total power. Default is 1.0.
            num_eval_points (int, optional): Number of evaluation points. Default is 64.
            use_eval_avg (bool, optional): Whether to use average evaluation. Default is True.
        """
        

        super().__init__([-aperture_radius,aperture_radius],
                         [-aperture_radius,aperture_radius],
                         num_conv_points,num_conv_points,
                         sigma,device,dtype,
                         num_integration_points_desired,desired_irradiance_func,
                         residual_integration_method,total_power_desired,
                         num_eval_points,num_eval_points,use_eval_avg)
        
    