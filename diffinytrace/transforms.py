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
from .intersection import SemiFunctionalModule,cat_semi_functionals
import numpy as np
from . optimize import make_parameter_from_input  
class Transform(SemiFunctionalModule):
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        raise NotImplementedError("params_list not implemented")

    @staticmethod
    def functional(O,*params):
        raise NotImplementedError("functional not implemented")
    
    def get_transformation_matrix(self,device=None,dtype=None):
        raise NotImplementedError("get_transformation_matrix not implemented")
    
    def get_transform(self):
        return self
    
    def to_global_dir(self,direction):
        M = self.get_transformation_matrix(direction.device,direction.dtype)
        R = M[np.ix_([0,1,2],[0,1,2])]
        out = direction@R.T
        return out

    def to_local_dir(self,direction):
        M = self.get_transformation_matrix(direction.device,direction.dtype)
        R = M[np.ix_([0,1,2],[0,1,2])]
        R_inv = torch.inverse(R)
        out = direction@R_inv.T
        return out

    def to_global_pos(self,position):
        M = self.get_transformation_matrix(position.device,position.dtype)
        v = torch.zeros((position.shape[0],4),device=position.device,dtype=position.dtype)
        v[:,[0,1,2]] = position
        v[:,3] = torch.ones_like(v[:,3])   
        _out = v@M.T
        out = _out[:,[0,1,2]] 
        return out
    
    def to_local_pos(self,position):
        return self.functional(position,*self.get_functional_param_args())

        
class Identity(Transform):
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        return []

    @staticmethod
    def functional(O):
        return O

    def get_transformation_matrix(self,device=None,dtype=None):
        out = torch.eye(4,device=device,dtype=dtype)
        return out 

class Compose(Transform):
    def __init__(self,transform_list):
        super().__init__()
        self.transform_list = nn.ModuleList(transform_list)
        self.functional = cat_semi_functionals(self.transform_list)
    
    def get_functional_param_args(self):
        out = []
        for elem in self.transform_list:
            out += elem.get_functional_param_args()
        return out
    
    def get_transformation_matrix(self,device=None,dtype=None):
        out = torch.eye(4,device=device,dtype=dtype)
        for elem in self.transform_list:
            tmp = elem.get_transformation_matrix(device,dtype)
            if not device is None and tmp.device != device:
                tmp = tmp.to(device)
            if not dtype is None and tmp.dtype != dtype:
                tmp = tmp.to(dtype)
            if out.device != tmp.device:
                tmp = tmp.to(device)
            if out.dtype != tmp.dtype:
                tmp = tmp.to(dtype)
            out = out @ tmp        
        return out


class Offset(Transform):
    def __init__(self,pos,parent_transform=Identity()):
        super().__init__()
        self.pos = make_parameter_from_input(pos)
        self.parent_transform = parent_transform.get_transform()

    def get_functional_param_args(self):
        return [self.pos]+self.parent_transform.get_functional_param_args()

    def functional(self,O,pos,*parent_param_args):
        O = self.parent_transform.functional(O,*parent_param_args)
        return O-pos

    def get_transformation_matrix(self,device=None,dtype=None):
        if device is None:
            device = self.pos.device
        if dtype is None:
            dtype = self.pos.dtype
        parent_transform_matrix = self.parent_transform.get_transformation_matrix(device=device,dtype=dtype)
        this_matrix = torch.eye(4,device=device,dtype=dtype)
        
        this_matrix[[0,1,2],-1] = self.pos.to(device=device,dtype=dtype)    
        
        out = parent_transform_matrix@this_matrix
        return out 


class Distance(Transform):
    def __init__(self,distance,axis = 2,parent_transform=Identity()):
        super().__init__()
        self.distance = make_parameter_from_input(distance)
        self.unit_vec = torch.tensor([0.,0.,0.])
        #self.register_buffer('unit_vec', torch.tensor([0.,0.,0.]))  # Buffer attribute

        self.unit_vec[axis] = 1.0 #is constant!
        self.parent_transform = parent_transform.get_transform()

    def get_functional_param_args(self):
        unit_vec = self.unit_vec
        if unit_vec.device != self.distance.device:
            unit_vec = unit_vec.to(self.distance.device)
        return [self.distance,unit_vec]+self.parent_transform.get_functional_param_args()

    def functional(self,O,distance,unit_vec,*parent_param_args):
        O = self.parent_transform.functional(O,*parent_param_args)
        O = O-distance*unit_vec
        return O
    
    def get_transformation_matrix(self,device=None,dtype=None):
        if device is None:
            device = self.distance.device
        if dtype is None:
            dtype = self.distance.dtype

        unit_vec = self.unit_vec.to(device=device,dtype=dtype)
        parent_transform_matrix = self.parent_transform.get_transformation_matrix(device,dtype)
        this_matrix = torch.eye(4,device=device,dtype=dtype)
        this_matrix[[0,1,2],-1] = self.distance.to(device=device,dtype=dtype)*unit_vec
        out = parent_transform_matrix@this_matrix
        return out 
"""
    def to(self, *args, **kwargs):
        super(Distance, self).to(*args, **kwargs)
        self.unit_vec = self.unit_vec.to(*args, **kwargs)
        return self
"""


"""
def rotation_matrix_x(angle):
    angle = angle*(2.0*torch.pi/360.0)
    
    device = angle.device
    dtype = angle.dtype
    

    return torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle), -torch.sin(angle)],
        [0, torch.sin(angle), torch.cos(angle)]
    ],device=device,dtype=dtype)

def rotation_matrix_y(angle):
    angle = angle*(2.0*torch.pi/360.0)
    device = angle.device
    dtype = angle.dtype


    return torch.tensor([
        [torch.cos(angle), 0, torch.sin(angle)],
        [0, 1, 0],
        [-torch.sin(angle), 0, torch.cos(angle)]
    ],device=device,dtype=dtype)

def rotation_matrix_z(angle):
    angle = angle*(2.0*torch.pi/360.0)

    device = angle.device
    dtype = angle.dtype
    return torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0],
        [0, 0, 1]
    ],device=device,dtype=dtype)

import torch
"""
def rotation_matrix_x(angle):
    """
    Returns the 4x4 rotation matrix for a rotation around the X-axis by `angle` degrees.
    """
    # Convert angle from degrees to radians
    angle = angle * (2.0 * torch.pi / 360.0)
    device = angle.device
    dtype = angle.dtype

    # Initialize a 4x4 identity matrix
    rot_x = torch.eye(3, dtype=dtype, device=device)

    # Set the rotation entries
    rot_x[1, 1] = torch.cos(angle)
    rot_x[1, 2] = -torch.sin(angle)
    rot_x[2, 1] = torch.sin(angle)
    rot_x[2, 2] = torch.cos(angle)

    return rot_x

def rotation_matrix_y(angle):
    """
    Returns the 4x4 rotation matrix for a rotation around the Y-axis by `angle` degrees.
    """
    # Convert angle from degrees to radians
    angle = angle * (2.0 * torch.pi / 360.0)
    device = angle.device
    dtype = angle.dtype

    # Initialize a 4x4 identity matrix
    rot_y = torch.eye(3, dtype=dtype, device=device)

    # Set the rotation entries
    rot_y[0, 0] = torch.cos(angle)
    rot_y[0, 2] = torch.sin(angle)
    rot_y[2, 0] = -torch.sin(angle)
    rot_y[2, 2] = torch.cos(angle)

    return rot_y

def rotation_matrix_z(angle):
    """
    Returns the 4x4 rotation matrix for a rotation around the Z-axis by `angle` degrees.
    """
    # Convert angle from degrees to radians
    angle = angle * (2.0 * torch.pi / 360.0)
    device = angle.device
    dtype = angle.dtype

    # Initialize a 4x4 identity matrix
    rot_z = torch.eye(3, dtype=dtype, device=device)

    # Set the rotation entries
    rot_z[0, 0] = torch.cos(angle)
    rot_z[0, 1] = -torch.sin(angle)
    rot_z[1, 0] = torch.sin(angle)
    rot_z[1, 1] = torch.cos(angle)

    return rot_z


class Rotation(Transform):
    def __init__(self,angle,axis,parent_transform=Identity()):
        #TODO test rotation for combi angle_x, angle_y, angle_z Reihenfolge egal?
        super().__init__()
        self.angle = make_parameter_from_input(angle)
        self.axis = axis
        self.parent_transform = parent_transform.get_transform()

    def get_functional_param_args(self):
        return [self.angle]+self.parent_transform.get_functional_param_args()

    
    def functional(self,O,angle,*parent_param_args):
        #R = rotate_3d(angle_x, angle_y, angle_z)
        O = self.parent_transform.functional(O,*parent_param_args)
        R = None
        if self.axis == 0: 
            R = rotation_matrix_x(360.0-angle)
        if self.axis == 1: 
            R = rotation_matrix_y(360.0-angle)
        if self.axis == 2: 
            R = rotation_matrix_z(360.0-angle)
    
        return O@R.T

    def get_transformation_matrix(self,device=None,dtype=None):
        if device is None:
            device = self.angle.device
        if dtype is None:
            dtype = self.angle.dtype
        
        parent_transform_matrix = self.parent_transform.get_transformation_matrix(device,dtype)
        R = None
        if self.axis == 0: 
            R = rotation_matrix_x(self.angle)
        if self.axis == 1: 
            R = rotation_matrix_y(self.angle)
        if self.axis == 2: 
            R = rotation_matrix_z(self.angle)
        
        if R.device != device:
            R = R.to(device)
        if R.dtype != dtype:
            R = R.to(dtype)
        this_matrix = torch.eye(4,device=device,dtype=dtype)
        this_matrix[:3,:3] = R
        out = parent_transform_matrix@this_matrix
        return out 


