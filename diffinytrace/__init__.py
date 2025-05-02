# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


"""
This module provides a collection of functions and classes for optical system design and analysis.
It includes modules for ray tracing, surface definitions, optimization, and more.
"""

__all__ = [
    # Submodules
    "source",
    "transforms",
    "target_grid",
    "utils",
    "plotting",
    "basis_functions",
    "nonimaging",
    "optimize",
    "integrators",
    "export",
    "render",
    "constraints",
    "spectrum",

    # intersection
    "cat_semi_functionals",
    "get_functional_param_args",
    "construct_surface_and_normal_func",
    "construct_surface_and_normal_func_with_params",
    "CustomAutogradRule_t",
    "get_ray_intersection_length",

    # surface
    "Plane",
    "Aspheric",
    "Bspline",
    "Legendre",
    "bspline_n_after_refinement",

    # element
    "OpticalSystem",
    "SequentialOpticalSystem",
    "OpticalElement",
    "OpticalSurface",
    "LensSurfaceTransmissionEnter",
    "LensSurfaceTransmissionLeave",
    "Lens",
    "Mirror",
    "Detector",
    "trace_to_detector",
    "get_unused_params_mask",
    "set_used_params_bounds_to_constant",
    "set_unused_params_to_zero",
    "set_unused_bspline_coeff_to_nearest",
    "FresnelVirtualLens",

    # config
    "set_tolerance",
    "get_tolerance",
    "set_max_iterations",
    "get_max_iterations",
    "restore_default_settings",
    "get_damping_factor",
    "set_damping_factor",
    "get_show_iteration_count",
    "set_show_iteration_count",

    # optimize
    "minimize",
    "make_parameter_from_input",
    
    # refractive_index
    "materials",
    "RefractiveIndex",

    # autograd
    "grad",
]

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.set_default_dtype(torch.float64)

from . import source
from . import transforms
from . import target_grid
from . import utils
#from . import refractive_index
from . import plotting
from . import basis_functions
from . import nonimaging
from . import optimize
from . import integrators
from . import export
from . import render
from . import constraints
from . import spectrum

from .intersection import cat_semi_functionals,get_functional_param_args,\
    construct_surface_and_normal_func,construct_surface_and_normal_func_with_params,\
    CustomAutogradRule_t,get_ray_intersection_length

from .surface import Plane,Aspheric,Bspline,Legendre,bspline_n_after_refinement

from .element import OpticalSystem,SequentialOpticalSystem,OpticalElement,OpticalSurface,\
    LensSurfaceTransmissionEnter,LensSurfaceTransmissionLeave,Lens,Mirror,Detector,trace_to_detector,\
    get_unused_params_mask,set_used_params_bounds_to_constant,\
    set_unused_params_to_zero,set_unused_bspline_coeff_to_nearest,\
    FresnelVirtualLens

from .config import set_tolerance,get_tolerance,set_max_iterations,\
    get_max_iterations,restore_default_settings,get_damping_factor,set_damping_factor,\
    get_show_iteration_count,set_show_iteration_count
from .optimize import minimize,make_parameter_from_input

from .refractive_index import materials
from .refractive_index import RefractiveIndex
from .utils.autograd import grad

