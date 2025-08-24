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
        """
        Returns the transformation matrix of the object.

        Returns:
            torch.Tensor: The transformation matrix.
        """
        return self.get_transform().get_transformation_matrix()

    def to_global_dir(self,direction):
        """
        Converts a direction vector from local to global coordinates.

        Args:
            direction (torch.Tensor): Direction vector in local coordinates.

        Returns:
            torch.Tensor: Direction vector in global coordinates.
        """
        return self.get_transform().to_global_dir(direction)

    def to_local_dir(self,direction):
        """
        Converts a direction vector from global to local coordinates.

        Args:
            direction (torch.Tensor): Direction vector in global coordinates.

        Returns:
            torch.Tensor: Direction vector in local coordinates.
        """
        return self.get_transform().to_local_dir(direction)

    def to_global_pos(self,position):
        """
        Converts a position from local to global coordinates.

        Args:
            position (torch.Tensor): Position in local coordinates.

        Returns:
            torch.Tensor: Position in global coordinates.
        """
        return self.get_transform().to_global_pos(position)

    def to_local_pos(self,position):
        """
        Converts a position from global to local coordinates.

        Args:
            position (torch.Tensor): Position in global coordinates.

        Returns:
            torch.Tensor: Position in local coordinates.
        """
        return self.get_transform().to_local_pos(position)

    def get_transform(self):
        """
        Returns the transformation object associated with this physical object.

        Raises:
            NotImplementedError: If not implemented in subclass.

        Returns:
            object: Transformation object.
        """
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
        """
        Returns constraint functions for the surface that must be less than or equal to zero.

        Raises:
            NotImplementedError: If not implemented in subclass.

        Returns:
            list[Callable]: List of constraint functions.
        """
        raise NotImplementedError("PhysicalSurface: get_constraint_funs_geq_zero is not implemented")
    
    """
    def get_corners_in_parameter_space(self):
        raise NotImplementedError("get_corners: is not implemented")
    
    def get_edge_funcs_in_parameter_space(self):
        raise NotImplementedError("get_edge_funcs_in_parameter_space: is not implemented")
    """
    
    def parametric_sample(self, num_points: int, method: str = "sobol") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples points on the surface in parameter space.

        Args:
            num_points (int): Number of points to sample.
            method (str, optional): Sampling method. Defaults to "sobol".

        Raises:
            NotImplementedError: If not implemented in subclass.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Sampled parameter positions and corresponding surface positions.
        """
        raise NotImplementedError("PhysicalSurface: sample() not implemented")

    def parametric_surface(self, parametric_pos: torch.Tensor) -> torch.Tensor:
        """
        Maps parameter space positions to surface positions.

        Args:
            parametric_pos (torch.Tensor): Positions in parameter space.

        Raises:
            NotImplementedError: If not implemented in subclass.

        Returns:
            torch.Tensor: Surface positions.
        """
        raise NotImplementedError("PhysicalSurface: parametric_surface is not implemented")

