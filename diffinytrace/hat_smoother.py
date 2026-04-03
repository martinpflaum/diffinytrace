

from typing import Callable
import torch

class HatSmoother:
    def __init__(self,
                 x_range:list,
                 y_range:list,
                 x_grid_size:int,
                 y_grid_size:int,
                 desired_irradiance_fun:Callable,
                 dtype=torch.get_default_dtype(),
                 device=torch.get_default_device()):
    pass