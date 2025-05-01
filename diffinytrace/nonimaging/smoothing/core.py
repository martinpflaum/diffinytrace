# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import torch
import numpy as np
from ...integrators import Cube
from ...element import trace_to_detector,SequentialOpticalSystem
import math
from ...source import LightSource
from ...target_grid import Grid
import gc

class Smoother:
    """
    The Smoother class is the base class for most algorithmic differentiable ray tracing measurement functions.
    
    """

    def __init__(self,v_range,
                h_range,
                v_num_conv_points:int,
                h_num_conv_points:int,
                device=torch.get_default_device(),
                dtype=torch.get_default_dtype(),
                num_integration_points_desired=[701,701],
                desired_irradiance_func=None,
                residual_integration_method="midpoint",
                total_power_desired=1.0,
                v_num_eval_points:int=64,
                h_num_eval_points:int=64,
                use_eval_avg=True):
        r"""
        Constructor for the Smoother class.
        
        Args:
            v_range (tuple): Vertical range of the grid.
            h_range (tuple): Horizontal range of the grid.
            v_num_conv_points (int): Number of vertical convolution points.
            h_num_conv_points (int): Number of horizontal convolution points.
            device (torch.device): Device to use for computations.
            dtype (torch.dtype): Data type for computations.
            num_integration_points_desired (list[int]): Number of integration points for desired irradiance.
            desired_irradiance_func (callable): Function to describing the desired irradiance.
            residual_integration_method (str): Method for residual integration ('midpoint' or 'simpson').
            total_power_desired (float): Total power of the source and desired irradiance.
            v_num_eval_points (int): Number of vertical gaussian measurement functions.
            h_num_eval_points (int): Number of horizontal gaussian measurement functions.
            use_eval_avg (bool): Whether to use averaging durring the evaluation.
        """
 
        self.device = device
        self.dtype = dtype
        
        self.__desired_irradiance_func = desired_irradiance_func
        self.num_integration_points_desired = num_integration_points_desired
        self.v_num_conv_points = v_num_conv_points
        self.h_num_conv_points = h_num_conv_points
        self.v_range = v_range
        self.h_range = h_range
        
        self._desired_smooth_irradiance = None 
        self._desired_none_smooth_irradiance_opti = None 
        self._desired_none_smooth_irradiance_eval = None 

        self.h_num_eval_points = h_num_eval_points
        self.v_num_eval_points = v_num_eval_points
        
        self.grid_eval = Grid(v_range,h_range,v_num_eval_points,h_num_eval_points)
        self.grid_opti = Grid(v_range,h_range,self.v_num_conv_points,self.h_num_conv_points)
        self.residual_integration_method = residual_integration_method

        self.total_power_desired = total_power_desired
        
        self.integrator = Cube([self.v_range,self.h_range])
        self.conv_points,self.weights = self.integrator.sample([self.v_num_conv_points,self.h_num_conv_points],residual_integration_method)
        self.include_boundary = True
        self.use_eval_avg = use_eval_avg
        
        if residual_integration_method=="simpson":
            if self.v_num_conv_points % 2 != 1 and self.h_num_conv_points % 2 != 1:
                raise RuntimeError("v_num_conv_points and h_num_conv_points must be odd numbers")
            
        if residual_integration_method=="simpson":
            self.include_boundary = True
        elif residual_integration_method=="midpoint":
            self.include_boundary = False
        else:
            raise RuntimeError("residual_integration_method not supported!")

        if desired_irradiance_func is not None:
            self.calc_smooth_desired_irradiance()
        self.last_eval_merit_val = None

        self.counter = 0

    @property
    def desired_smooth_irradiance(self):
        if self._desired_smooth_irradiance is None:
            self.calc_smooth_desired_irradiance()
        return self._desired_smooth_irradiance
    
    @desired_smooth_irradiance.setter
    def desired_smooth_irradiance(self, value):
        self._desired_smooth_irradiance = value

    @property
    def desired_none_smooth_irradiance_opti(self):
        if self._desired_none_smooth_irradiance_opti is None:
            self.calc_none_smooth_desired_irradiance_opti()
        return self._desired_none_smooth_irradiance_opti
    
    @desired_none_smooth_irradiance_opti.setter
    def desired_none_smooth_irradiance_opti(self, value):
        self._desired_none_smooth_irradiance_opti = value

    @property
    def desired_none_smooth_irradiance_eval(self):
        if self._desired_none_smooth_irradiance_eval is None:
            self.calc_none_smooth_desired_irradiance_eval()
        return self._desired_none_smooth_irradiance_eval
    
    @desired_none_smooth_irradiance_eval.setter
    def desired_none_smooth_irradiance_eval(self, value):
        self._desired_none_smooth_irradiance_eval = value

    

    def get_desired_irradiance_none_smoothed(self,y)->torch.Tensor:
        r"""
        Returns the desired irradiance without smoothing.
        
        Args:
            y (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The desired irradiance.
        """
        return self.__desired_irradiance_func(y)*self.__desir_irradiance_func_multi
    

    def get_none_smooth_irradiance(self,y:torch.Tensor,val_multi:torch.Tensor)->torch.Tensor:
        """
        Returns the non-smoothed irradiance.
        
        Args:
            y (torch.Tensor): The input tensor.
            val_multi (torch.Tensor): The value multiplier.
        
        Returns:
            torch.Tensor: The non-smoothed irradiance.
        """
        
        y = y.detach().cpu()
        val_multi = val_multi.detach().cpu()
        return self.grid_eval.sum(y,val_multi)/self.grid_eval.get_pixel_area()

    def calc_none_smooth_desired_irradiance_eval(self,device=None,dtype=None):
        """
        Calculates the non-smoothed desired irradiance for evaluation.
        
        Args:
            device (torch.device, optional): Device to use for calculations.
            dtype (torch.dtype, optional): Data type to use for calculations.
        
        Returns:
            None
        """
        
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if self.use_eval_avg:
            gc.collect()
            y,weights = self.integrator.sample(self.num_integration_points_desired,"sobol") 
            y = y.to(device=device,dtype=dtype)
            weights = weights.to(device=device,dtype=dtype)
            
            tmp = self.get_desired_irradiance_none_smoothed(y)
            self._desired_none_smooth_irradiance_eval = self.grid_eval.sum(y,weights*tmp)/self.grid_eval.get_pixel_area()
            gc.collect()
        else:
            gc.collect()
            y = self.grid_eval.get_pixel_centers()
            y = y.to(device=device,dtype=dtype)
            
            shape = y.shape
            y = y.reshape(-1,2)
            tmp = self.get_desired_irradiance_none_smoothed(y)
            self._desired_none_smooth_irradiance_eval = tmp.reshape(shape[0],shape[1])
            gc.collect()
                
    def calc_none_smooth_desired_irradiance_opti(self,device=None,dtype=None):        
        """
        Calculates the non-smoothed desired irradiance for optimization.
        
        Args:
            device (torch.device, optional): Device to use for calculations.
            dtype (torch.dtype, optional): Data type to use for calculations.
        
        Returns:
            None
        """

        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        gc.collect()
        y = self.grid_opti.get_pixel_centers()
        y = y.to(device=device,dtype=dtype)
        
        shape = y.shape
        y = y.reshape(-1,2)
        tmp = self.get_desired_irradiance_none_smoothed(y)
        self._desired_none_smooth_irradiance_opti = tmp.reshape(shape[0],shape[1])
        gc.collect()
        


    def calc_smooth_desired_irradiance(self,device=None,dtype=None):
        """
        Calculates the smoothed desired irradiance.
        
        Args:
            device (torch.device, optional): Device to use for calculations.
            dtype (torch.dtype, optional): Data type to use for calculations.
        Returns:
            None
        """
        
        #sqrt_integration_points = int(math.sqrt(float(self.num_integration_points)))
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        
        gc.collect()
        if self.__desired_irradiance_func is None:
            raise RuntimeError("Smoother Error: desired_irradiance_func is None. Specify it in the constructor or set it manually!")
            
        y,weights = self.integrator.sample(self.num_integration_points_desired,"sobol") 
        y = y.to(device=device,dtype=dtype)
        
        weights = weights.to(device=device,dtype=dtype)
        
        val_multi = self.__desired_irradiance_func(y)*weights
        self.__desir_irradiance_func_multi = (self.total_power_desired/torch.sum(val_multi))
        val_multi = val_multi*self.__desir_irradiance_func_multi

        self._desired_smooth_irradiance = self.get_smooth_irradiance(y,val_multi) 
        gc.collect()
        

    def get_integral_over_distribution(self,val:torch.Tensor)->torch.Tensor:
        """
        Returns the integral over the distribution.
        
        Args:
            val (torch.Tensor): The input tensor.
        
        Returns:
            torch.Tensor: The integral over the distribution.
        """
        
        device = val.device
        dtype = val.dtype
        
        if (device != self.weights.device) or (dtype != self.weights.dtype):
            self.weights = self.weights.to(device=device,dtype=dtype).detach()

        return torch.sum(self.weights.reshape(*val.shape)*val)
    

    def get_eval_merit_function_value(self,points:torch.Tensor,ray_multi:torch.Tensor,wl:torch.Tensor):
        """
        Returns the merit function value for evaluation.
        
        Args:
            points (torch.Tensor): The input points.
            ray_multi (torch.Tensor): The ray multipliers.
            wl (torch.Tensor): The wavelengths.
            
        Returns:
            torch.Tensor: The merit function value.
        """
        
        device = "cpu"
        dtype = points.dtype
        points = points.cpu()
        ray_multi = ray_multi.cpu()
        wl = wl.cpu()
        self.counter+=1
        with torch.no_grad():
            if (device != self.desired_none_smooth_irradiance_eval.device) or (dtype != self.desired_none_smooth_irradiance_eval.dtype) :
                self.desired_none_smooth_irradiance_eval = self.desired_none_smooth_irradiance_eval.to(device=device,dtype=dtype).detach()
                
            desired_irr = self.desired_none_smooth_irradiance_eval
            desired_irr = desired_irr.cpu()

            none_smooth_irr = self.get_none_smooth_irradiance(points,ray_multi).cpu()
            residual = desired_irr-none_smooth_irr

            residual = residual.reshape(-1)
            residual = residual.detach()
            integral = torch.sum((residual**2.0)*torch.tensor(self.grid_eval.get_pixel_area(),device=device,dtype=dtype))
            return torch.sqrt(integral).cpu().item()


    def get_merit_function_value(self,
            points:torch.Tensor,
            ray_multi:torch.Tensor,
            wl:torch.Tensor,
            use_desired_irradiance_smoothing=True,
            use_power_correction=False,
            save_last_eval = False)->torch.Tensor:
        """
        Returns the merit function value.
        
        Args:
            points (torch.Tensor): The input points.
            ray_multi (torch.Tensor): The ray multipliers.
            wl (torch.Tensor): The wavelengths.
            use_desired_irradiance_smoothing (bool): Whether to use desired irradiance smoothing.
            use_power_correction (bool): Whether to use power correction.
            save_last_eval (bool): Whether to save the last evaluation.
        
        Returns:
            torch.Tensor: The merit function value.
        """
        device = points.device
        dtype = points.dtype
        desired_irr = None

        if (use_power_correction) and (not use_desired_irradiance_smoothing):
            raise ValueError("use_power_correction can only be used with desired_irradiance_smoothing")
        
        if (self.residual_integration_method!="midpoint") and (not use_desired_irradiance_smoothing):
            raise RuntimeError("can only use midpoint residual_integration_method if use_desired_irradiance_smoothing=False") 
        
        if save_last_eval:
            self.last_eval_merit_val = self.get_eval_merit_function_value(points,ray_multi,wl)
        

        if use_desired_irradiance_smoothing:
            
            if (device != self.desired_smooth_irradiance.device) or (dtype != self.desired_smooth_irradiance.dtype) :
                self.desired_smooth_irradiance = self.desired_smooth_irradiance.to(device=device,dtype=dtype).detach()

            desired_irr = self.desired_smooth_irradiance
        else:
            
            if (device != self.desired_none_smooth_irradiance_opti.device) or (dtype != self.desired_none_smooth_irradiance_opti.dtype) :
                self.desired_none_smooth_irradiance_opti = self.desired_none_smooth_irradiance_opti.to(device=device,dtype=dtype).detach()

            desired_irr = self.desired_none_smooth_irradiance_opti

                

        smoothed_irradiance = self.get_smooth_irradiance(points,ray_multi)


        if save_last_eval:
            with torch.no_grad():
                desired_irr_power = self.get_integral_over_distribution(desired_irr.detach())
                smoothed_irradiance_power = self.get_integral_over_distribution(smoothed_irradiance.detach())
                self.last_desired_irr_power = desired_irr_power.cpu().detach().numpy()
                self.last_smoothed_irradiance_power = smoothed_irradiance_power.cpu().detach().numpy()
        
        #smoothed_irradiance = smoothed_irradiance/smoothed_irradiance.sum()
        #print("sum compare:",self.get_integral_over_distribution(smoothed_irradiance),self.get_integral_over_distribution(desired_irr))
        
        residual = desired_irr-smoothed_irradiance
        residual = residual.reshape(-1)

        residual_integral = self.get_integral_over_distribution(residual**2.0)


        out = torch.sqrt(residual_integral)

        #Power correction not 100% correct...sqrt is at the wrong place...
        if use_power_correction:
            raise RuntimeError("power_correction is not implemented.")
            """    desired_irr_power = self.get_integral_over_distribution(desired_irr)
            outside_domain_desired_irr_power = self.total_power_desired-desired_irr_power
            smoothed_irradiance_power = self.get_integral_over_distribution(smoothed_irradiance)
            outside_domain_smoothed_irradiance = self.total_power_desired-smoothed_irradiance_power

            return out + torch.abs(torch.sqrt(outside_domain_desired_irr_power)-torch.sqrt(outside_domain_smoothed_irradiance))
            """
        else:
            return out
        
    def get_smooth_irradiance(self,points,val_multi)->torch.Tensor:
        """
        Returns the smoothed irradiance.
        
        Args:
            points (torch.Tensor): The input points.
            val_multi (torch.Tensor): The value multipliers.
        
        Returns:
            torch.Tensor: The smoothed irradiance.
        """
        raise NotImplementedError("get_smooth_irradiance is not implemented")


def create_merit_function(optical_system:SequentialOpticalSystem,
                        sequence,
                        source:LightSource,
                        detector,
                        num_rays,
                        smoother:Smoother,
                        device,
                        method_ray_tracing="sobol",
                        use_desired_irradiance_smoothing=True,
                        use_power_correction=False,
                        save_last_eval=False):
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
        x,weights,y,wl = trace_to_detector(optical_system,sequence,source,detector,num_rays,device,method_ray_tracing=method_ray_tracing)
        Qval = source.get_flux(x)
        #print("total energy rays:",(Qval*weights).sum())
        merit_value = smoother.get_merit_function_value(y,Qval*weights,wl,use_desired_irradiance_smoothing,use_power_correction,save_last_eval)        
        return merit_value
    return merit_function