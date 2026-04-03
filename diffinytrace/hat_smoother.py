

from typing import Callable
import torch
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



class HatSmoother:
    def __init__(self,
                 x_range:list,
                 y_range:list,
                 x_grid_size:int,
                 y_grid_size:int,
                 desired_irradiance_fun:Callable,
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device()):
        self.x_range = x_range
        self.y_range = y_range
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.desired_irradiance_fun = desired_irradiance_fun
        self.dtype = dtype
        self.device = device

        self.grid = Grid(x_range,y_range,x_grid_size,y_grid_size)
        centers = self.grid.get_pixel_centers().reshape(-1,2)
        self.discrete_desired_irradiance:torch.Tensor = desired_irradiance_fun(centers).reshape(y_grid_size,x_grid_size)
        