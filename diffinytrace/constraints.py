# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = ["Constraint", "EqualZero", "GEQZero", "LEQZero"]

from .physical_object import PhysicalSurface
import torch
from . integrators import Cube
from .optimize import minimize
from .utils.autograd import grad
#from .element import OpticalSurface

class Constraint():
    """
    Base class for optimization constraints.

    Attributes:
        fun (Callable): Function defining the constraint.
        type (str): Type of constraint ('eq' or 'ineq').
    """
    def __init__(self,fun,type):
        self.fun = fun
        self.type = type
    
class EqualZero(Constraint):
    """
    Equality constraint enforcing `fun() == 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(fun,'eq')

class GEQZero(Constraint):
    """
    Inequality constraint enforcing `fun() >= 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(fun,'ineq')

class LEQZero(Constraint):
    """
    Inequality constraint enforcing `fun() <= 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(lambda: -fun(),'ineq')

