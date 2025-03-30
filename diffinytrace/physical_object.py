"""
MIT License

Copyright (c) 2025 Martin Pflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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
    
