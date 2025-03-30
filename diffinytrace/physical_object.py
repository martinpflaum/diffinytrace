"""
Copyright (C) 2024 Martin Pflaum

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import torch
import torch.nn as nn


class PhysicalObject(nn.Module):
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
    
