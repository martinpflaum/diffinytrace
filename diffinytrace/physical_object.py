# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "PhysicalObject",
    "PhysicalSurface"
]

import torch
import torch.nn as nn


class PhysicalObject(nn.Module):
    """
    Abstract base class for physical objects in the optical system.
    This class can be used to define surface distance constraints and is
    also used for plotting.
    """
    def __init__(self):
        super().__init__()

    def get_transformation_matrix(self):
        return self.get_transform().get_transformation_matrix()

    def to_global_dir(self,direction):
        return self.get_transform().to_global_dir(direction)

    def to_local_dir(self,direction):
        return self.get_transform().to_local_dir(direction)

    def to_global_pos(self,position):
        return self.get_transform().to_global_pos(position)

    def to_local_pos(self,position):
        return self.get_transform().to_local_pos(position)

    def get_transform(self):
        raise NotImplementedError("PhysicalObject: get_transform not implemented")

class PhysicalSurface(PhysicalObject):
    """
    Abstract base class for physical surfaces in the optical system.
    This class can be used to define surface distance constraints and is
    also used for plotting.
    """
    def __init__(self):
        super().__init__()
    
    def get_constraint_funs_leq_zero(self):
        raise NotImplementedError("PhysicalSurface: get_constraint_funs_geq_zero is not implemented")
    
    """
    def get_corners_in_parameter_space(self):
        raise NotImplementedError("get_corners: is not implemented")
    
    def get_edge_funcs_in_parameter_space(self):
        raise NotImplementedError("get_edge_funcs_in_parameter_space: is not implemented")
    """
    def parametric_sample(self,num_points,method="sobol")-> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("PhysicalSurface: sample() not implemented")

    def parametric_surface(self,parametric_pos)->torch.Tensor:
        raise NotImplementedError("PhysicalSurface: parametric_surface is not implemented")
    
