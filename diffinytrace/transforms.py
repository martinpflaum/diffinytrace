# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "Transform",
    "Identity",
    "Compose",
    "Offset",
    "Distance",
    "Rotation",
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z"
]

import torch
import torch.nn as nn
from .intersection import SemiFunctionalModule,cat_semi_functionals
import numpy as np
from . optimize import make_parameter_from_input  

class Transform(SemiFunctionalModule):
    """
    Base class for coordinate transformations.

    This class provides interfaces to transform directions and positions between
    local and global coordinate systems using homogeneous coordinates.

    Methods:
        get_functional_param_args(): Return parameters required for the transformation.
        functional(O, *params): Apply transformation in functional style.
        get_transformation_matrix(): Return the 4x4 transformation matrix.
        to_global_dir(direction): Transform direction to global space.
        to_local_dir(direction): Transform direction to local space.
        to_global_pos(position): Transform position to global space.
        to_local_pos(position): Transform position to local space.
    """
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        """
        Return parameters required for the transformation which constructs the surfaces through the functional.
        
        Returns:
            list: List of parameters required for the functional which constructs the surfaces.
        """
        raise NotImplementedError("params_list not implemented")

    @staticmethod
    def functional(O,*params)->torch.Tensor:
        """
        Apply transformation in functional style. This is global to local.
        
        Args:
            O (torch.Tensor): Input tensor to be transformed.
            *params: Parameters for the transformation.
        
        """
        raise NotImplementedError("functional not implemented")
    
    def get_transformation_matrix(self,device=None,dtype=None)->torch.Tensor:
        """
        Return the 4x4 transformation matrix.
        
        Args:
            device (torch.device, optional): Device for the matrix.
            dtype (torch.dtype, optional): Data type for the matrix.    
            
        Returns:
            torch.Tensor: 4x4 transformation matrix.
        """
        raise NotImplementedError("get_transformation_matrix not implemented")
    
    def get_transform(self):
        """
        Returns itself.
        """
        return self
    
    def to_global_dir(self,direction:torch.Tensor) -> torch.Tensor:
        """
        Transform direction to global space.
        Args:
            direction (torch.Tensor): Direction vector in local space.
        Returns:
            torch.Tensor: Direction vector in global space.
        """
        M = self.get_transformation_matrix(direction.device,direction.dtype)
        R = M[np.ix_([0,1,2],[0,1,2])]
        out = direction@R.T
        return out

    def to_local_dir(self,direction:torch.Tensor) -> torch.Tensor:
        """
        Transform direction to local space.
        Args:
            direction (torch.Tensor): Direction vector in global space.
        Returns:
            torch.Tensor: Direction vector in local space.
        """
        M = self.get_transformation_matrix(direction.device,direction.dtype)
        R = M[np.ix_([0,1,2],[0,1,2])]
        R_inv = torch.inverse(R)
        out = direction@R_inv.T
        return out

    def to_global_pos(self,position:torch.Tensor) -> torch.Tensor:
        """
        Transform position to global space.
        Args:
            position (torch.Tensor): Position vector in local space.
        Returns:
            torch.Tensor: Position vector in global space.
        """
        M = self.get_transformation_matrix(position.device,position.dtype)
        v = torch.zeros((position.shape[0],4),device=position.device,dtype=position.dtype)
        v[:,[0,1,2]] = position
        v[:,3] = torch.ones_like(v[:,3])   
        _out = v@M.T
        out = _out[:,[0,1,2]] 
        return out

    def to_local_pos(self,position:torch.Tensor) -> torch.Tensor:
        """
        Transform position to local space.
        Args:
            position (torch.Tensor): Position vector in global space.
        Returns:
            torch.Tensor: Position vector in local space.
        """
        return self.functional(position,*self.get_functional_param_args())

        
class Identity(Transform):
    """
    Identity transformation that returns input positions unchanged.

    Example:
        >>> import diffinytrace as dit
        >>> transf1 = dit.transforms.Identity()
    """
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        return []

    @staticmethod
    def functional(O:torch.Tensor) -> torch.Tensor:
        return O

    def get_transformation_matrix(self,device=None,dtype=None) -> torch.Tensor:
        out = torch.eye(4,device=device,dtype=dtype)
        return out 

class Compose(Transform):
    """
    Compose multiple transforms in sequence.

    Args:
        transform_list (list[Transform]): List of transformations to apply in order.
    """
    def __init__(self,transform_list):
        super().__init__()
        self.transform_list = nn.ModuleList(transform_list)
        self.functional = cat_semi_functionals(self.transform_list)
    
    def get_functional_param_args(self):
        out = []
        for elem in self.transform_list:
            out += elem.get_functional_param_args()
        return out

    def get_transformation_matrix(self,device=None,dtype=None) -> torch.Tensor:
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
    r"""
    Translation transform using an offset vector.

    The offset transformation shifts a position by a specified vector 
    \( \vec{w} = (w_x, w_y, w_z) \). The transformation matrix \( M \) 
    for an offset transformation is:

    .. math::

        M^{offset}(w_x, w_y, w_z) = 
        \begin{bmatrix}
        1 & 0 & 0 & w_x \\
        0 & 1 & 0 & w_y \\
        0 & 0 & 1 & w_z \\
        0 & 0 & 0 & 1
        \end{bmatrix}
    
    Example:
        >>> import diffinytrace as dit
        >>> transf1 = dit.transforms.Identity()
        >>> transf2 = dit.transforms.Offset([1.0, 2.0, 3.0], parent_transform=transf1)

    Args:
        pos (Tensor or list or float): The offset position as a 3D vector.
        parent_transform (Transform, optional): Optional parent transformation.
    """
    def __init__(self,pos,parent_transform=Identity()):
        super().__init__()
        self.pos = make_parameter_from_input(pos)
        self.parent_transform = parent_transform.get_transform()

    def get_functional_param_args(self):
        return [self.pos]+self.parent_transform.get_functional_param_args()

    def functional(self,O:torch.Tensor,pos:torch.Tensor,*parent_param_args)->torch.Tensor:
        O = self.parent_transform.functional(O,*parent_param_args)
        return O-pos

    def get_transformation_matrix(self,device=None,dtype=None) -> torch.Tensor:
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
    r"""
    Applies a translation along a specific axis by a given distance.

    The distance transformation applies a translation by a specific distance along a given axis 
    (e.g., \( x \)-, \( y \)-, or \( z \)-axis). The transformation matrix \( M \) for a distance 
    transformation along the \( z \)-axis is given by:

    .. math::

        M^{dist}_z(d) = 
        \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & d \\
        0 & 0 & 0 & 1
        \end{bmatrix},

    where \( d \) represents the distance of translation along the \( z \)-axis.

    Args:
        distance (float or Tensor): Distance to translate.
        axis (int): Axis along which translation is applied (0=X, 1=Y, 2=Z).
        parent_transform (Transform): Optional parent transformation.

    Example:
        >>> import diffinytrace as dit
        >>> transf1 = dit.transforms.Identity()
        >>> transf2 = dit.transforms.Distance(10.0,axis=2,parent_transform=transf1)

    Notes:
        For the local to global transformation it applies the following transformation:

        .. math::

            \mathbf{x}_\text{local} = \mathbf{x}_\text{parent} + d \cdot \mathbf{e}_i
    """
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

    def functional(self,O:torch.Tensor,distance:torch.Tensor,unit_vec:torch.Tensor,*parent_param_args)->torch.Tensor:
        O = self.parent_transform.functional(O,*parent_param_args)
        O = O-distance*unit_vec
        return O

    def get_transformation_matrix(self,device=None,dtype=None)->torch.Tensor:
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

def rotation_matrix_x(angle:torch.Tensor) -> torch.Tensor:
    """
    Construct a 3x3 rotation matrix around the X-axis.

    Args:
        angle (Tensor): Angle in degrees.

    Returns:
        Tensor: 3x3 rotation matrix.
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

def rotation_matrix_y(angle:torch.Tensor) -> torch.Tensor:
    """
    Construct a 3x3 rotation matrix around the Y-axis.

    Args:
        angle (Tensor): Angle in degrees.

    Returns:
        Tensor: 3x3 rotation matrix.
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

def rotation_matrix_z(angle:torch.Tensor) -> torch.Tensor:
    """
    Construct a 3x3 rotation matrix around the Z-axis.

    Args:
        angle (Tensor): Angle in degrees.

    Returns:
        Tensor: 3x3 rotation matrix.
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
    r"""
    Applies a 3D rotation around a principal axis.

    The rotational transformation rotates a point or direction around a specific axis 
    (e.g., \( x \)-, \( y \)-, and \( z \)-axis). For example, the rotation matrix 
    around the \( z \)-axis is:

    .. math::

        M^{rot}_z(\theta_z) = 
        \begin{bmatrix}
        \cos \theta_z & -\sin \theta_z & 0 & 0 \\
        \sin \theta_z & \cos \theta_z & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix}

    Args:
        angle (float or Tensor): Rotation angle in degrees.
        axis (int): Axis index (0=X, 1=Y, 2=Z).
        parent_transform (Transform, optional): Optional parent transformation.

    Example:
        >>> import diffinytrace as dit
        >>> transf1 = dit.transforms.Identity()
        >>> transf2 = dit.transforms.Distance(10.0,axis=2,parent_transform=transf1)
        >>> transf3 = dit.transforms.Rotation(45.,axis=0,parent_transform=transf2)

        
    """
    def __init__(self,angle,axis,parent_transform=Identity()):
        #TODO test rotation for combi angle_x, angle_y, angle_z Reihenfolge egal?
        super().__init__()
        self.angle = make_parameter_from_input(angle)
        self.axis = axis
        self.parent_transform = parent_transform.get_transform()

    def get_functional_param_args(self):
        return [self.angle]+self.parent_transform.get_functional_param_args()


    def functional(self,O:torch.Tensor,angle:torch.Tensor,*parent_param_args)->torch.Tensor:
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

    def get_transformation_matrix(self,device=None,dtype=None) -> torch.Tensor:
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


