# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import torch
import torch.nn as nn
from .utils.autograd import grad
from .config import get_max_iterations,get_tolerance,get_damping_factor,get_show_iteration_count

class SemiFunctionalModule(nn.Module):
    r"""
    Abstract base class for semi-functional surface modules.

    These modules define a static method `functional` that computes a
    functional transformation on inputs and parameters, and a method to list
    their functional parameters for optimization purposes.
    """
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        raise NotImplementedError("params_list not implemented")

    @staticmethod
    def functional(O,*params):
        raise NotImplementedError("functional not implemented")

def cat_semi_functionals(functional_modules):
    r"""
    Recursively chains a list of `SemiFunctionalModule`s into a single composite function.

    Each module's `functional()` method is applied in sequence using the respective
    slice of the parameter list.

    Args:
        functional_modules (list[SemiFunctionalModule]): List of functional modules.

    Returns:
        Callable: A function f(O, *params) that applies all modules in sequence.
    """
    
    if len(functional_modules) == 0:
        return lambda O,*params: O
    current_func = functional_modules[0].functional
    other = functional_modules[1:]
    num_params = len(functional_modules[0].get_functional_param_args())
    def fun_out(O,*params):
        other_funs = cat_semi_functionals(other)
        return other_funs(current_func(O,*params[:num_params]),*params[num_params:])
    return fun_out

def get_functional_param_args(semi_functional_module_list):
    r"""
    Collects all functional parameters from a list of semi-functional modules.

    Args:
        semi_functional_module_list (list[SemiFunctionalModule]): List of modules.

    Returns:
        list[torch.nn.Parameter]: Flattened list of all parameters.
    """
    out = []
    for elem in semi_functional_module_list:
        out += elem.get_functional_param_args()
    return out


def construct_surface_and_normal_func(semi_functional_module_list):
    r"""
    Constructs a function to evaluate both the surface value and its gradient
    (normal direction) with respect to the ray origin `O`.

    The surface is defined by composing the provided semi-functional modules.

    Returns a callable:

    .. math::
        (O, p_1, ..., p_n) \\mapsto ( s(O), \\frac{\\partial s}{\\partial O} )

    Args:
        semi_functional_module_list (list[SemiFunctionalModule]): List of modules.

    Returns:
        Callable: A function `s_dsd(O, *params, only_s=False)` returning
        surface value `s` and optionally gradient `ds/dO`.
    """
    s = cat_semi_functionals(semi_functional_module_list)
    def s_dsd(O,*params,only_s = False):
        sval,dsdval= None,None
        with torch.enable_grad():
            if not O.requires_grad:
                O.requires_grad = True
            sval = s(O,*params)
            if only_s:
                return sval
            dsdval = grad(sval,inputs=O,grad_outputs=torch.ones_like(sval))
            dsdval = dsdval[0]
        return sval,dsdval
    return s_dsd

def construct_surface_and_normal_func_with_params(semi_functional_module_list):
    r"""
    Constructs both the surface function and a list of its functional parameters.

    Useful for optimization workflows that require parameter tracking.

    Args:
        semi_functional_module_list (list[SemiFunctionalModule]): List of modules.

    Returns:
        tuple:
            Callable: A function computing surface and its gradient.
            list[torch.nn.Parameter]: The list of parameters for the surface.
    """
    s_dsd = construct_surface_and_normal_func(semi_functional_module_list)
    args = get_functional_param_args(semi_functional_module_list)
    return s_dsd,args

        
        
    
class CustomAutogradRule_t(torch.autograd.Function):
    """
    Custom PyTorch autograd rule for ray-surface intersection.

    Computes a differentiable intersection length `t` such that:

    .. math::
        s(O + t D) = 0

    where `O` is the ray origin, `D` is the direction, and `s` is the surface function.

    This rule enables backpropagation through `t` with respect to `O`, `D`, and surface parameters.
    """
    @staticmethod
    def forward(ctx,O,D,surface_and_normal_func,t_detached,*param_args):
        """
        Stores inputs for backward pass and returns precomputed `t`.

        Args:
            O (torch.Tensor): Ray origin of shape (N, 3).
            D (torch.Tensor): Ray direction of shape (N, 3).
            surface_and_normal_func (Callable): Surface function returning (s, ds/dR).
            t_detached (torch.Tensor): Estimated intersection length (detached).
            *param_args: Surface parameters.

        Returns:
            torch.Tensor: Intersection length `t`.
        """
        ctx.save_for_backward(O,D,t_detached,*param_args)
        ctx.surface_and_normal_func = surface_and_normal_func
        return t_detached

    @staticmethod
    def backward(ctx, grad_outputs):
        """
        Computes gradients of intersection length `t` with respect to:
        - ray origin `O`
        - ray direction `D`
        - surface parameters

        Args:
            grad_outputs (torch.Tensor): Gradient of the loss w.r.t. output `t`.

        Returns:
            tuple: Gradients with respect to inputs (O, D, None, None, *param_args).
        """
        saved_tensors = ctx.saved_tensors
        O = saved_tensors[0]
        D = saved_tensors[1]
        t_detached = saved_tensors[2]
        param_args = saved_tensors[3:]
        surface_and_normal_func = ctx.surface_and_normal_func
        t = CustomAutogradRule_t.apply(O,D,surface_and_normal_func,t_detached,*param_args)
        R = O+t*D
        
        param_args_clone = []
        for elem in param_args:
            if torch.is_tensor(elem):
                elem = elem.clone()
            param_args_clone.append(elem)

        s_val,dsdR_val = surface_and_normal_func(R,*param_args_clone)
        dsdR_T_D = torch.sum(dsdR_val*D,axis=-1)
        v1 = -grad_outputs.reshape(-1)/dsdR_T_D.reshape(-1)
            
        jact_dtdp = None
        with torch.enable_grad():
            s_val = [s_val.reshape(-1)]
            jact_dtdp = grad(s_val,[*param_args_clone], grad_outputs=v1,create_graph=True,retain_graph=True)

        jact_dtdO = v1.reshape(-1,1)*dsdR_val
        jact_dtdD = jact_dtdO*t.reshape(-1,1)
        return jact_dtdO,jact_dtdD,None,None,*jact_dtdp
    
def get_ray_intersection_length(O,D,surface_and_normal_func,param_args,t_init=None):
    """
    Solves for the intersection length `t` such that:

    .. math::
        s(O + t D) = 0

    using a Newton-style iteration method with damping.

    This function finds the length `t` where a ray intersects a parametric surface,
    given by a composed function with normal information.

    Args:
        O (torch.Tensor): Ray origins of shape (N, 3).
        D (torch.Tensor): Ray directions of shape (N, 3).
        surface_and_normal_func (Callable): A function returning (s, ds/dR).
        param_args (list): List of surface parameters.
        t_init (torch.Tensor, optional): Initial guess for `t`. If None, starts from zero.

    Returns:
        torch.Tensor: Estimated intersection lengths `t` with autograd support.

    Raises:
        Warning is printed (not exception) if convergence fails within `max_iter`.
    """
    tolerance = get_tolerance()
    max_iter = get_max_iterations()
    damping = get_damping_factor()
    device = O.device
    dtype = O.dtype

    N = O.shape[0]

    #better initial value
    t_detached = None
    if t_init is not None:
        t_detached = t_init.detach().reshape(N,1) 
    else:
        t_detached = torch.zeros((N,1),device=device,dtype=dtype)
    
    O_detached = O.detach()
    D_detached = D.detach()
    

    converged = False
    
    smax_vals = []
    for k in range(max_iter):
        R_detached = O_detached+t_detached*D_detached
        s_val,dsdR_val = surface_and_normal_func(R_detached,*param_args)
        s_val,dsdR_val = s_val.detach(),dsdR_val.detach()
        t_detached = t_detached-damping*s_val.reshape(-1,1)/(torch.sum(dsdR_val*D_detached,dim=-1).reshape(-1,1))
        t_detached = t_detached.detach()
        smax_vals += [torch.max(torch.abs(s_val.detach()))]
        if (s_val<tolerance).all():
            converged = True
            if get_show_iteration_count():
                print(f"Ray intersection with surface completed in {k} iterations.")
            break
    if not converged:
        print(f"Ray intersection FAILED to converge after {max_iter} iterations!\nThis is totally normal durring optimization when a bad parameterset is chosen."+"maximum svals are: "+str(smax_vals))
    
    t_out =  CustomAutogradRule_t.apply(O,D,surface_and_normal_func,t_detached,*param_args)
    return t_out