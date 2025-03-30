# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

"""
Optimization Utilities for PyTorch-SciPy Integration
====================================================

This submodule provides a set of tools for constrained and unconstrained optimization
of PyTorch models using SciPy optimizers. It bridges the gap between SciPy’s powerful
optimization routines and PyTorch’s autograd system, enabling flexible and efficient
hybrid optimization workflows.

Key Features:
-------------
- Seamless wrapping of PyTorch-based objective functions for use with SciPy.
- Automatic gradient computation using PyTorch’s autograd.
- Support for parameter bounds, including custom mask-based bounds.
- Caching and reuse of recent function/gradient evaluations.
- Integration with SciPy's `minimize`.
- Optional tracking of optimization history (function values and gradient norms).
- Utility functions for flattening/unpacking tensor parameters.
- Conversion of PyTorch parameters to SciPy-compatible formats with bounds.
- Support for custom constraints and callback functions.
"""


import scipy
import scipy.optimize
from .utils.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import copy

def make_bounds_from_param(param):
    """
    Creates default bounds (-∞, ∞) for each element of the input tensor.

    This function returns a tensor of shape `param.shape + [2]`, where the last
    dimension represents the lower and upper bounds for each element in `param`.

    Args:
        param (torch.Tensor): A tensor for which bounds should be created.

    Returns:
        torch.Tensor: A tensor of shape `param.shape + [2]` where
        `[..., 0] = -inf` (lower bounds) and `[..., 1] = inf` (upper bounds),
        with the same dtype and device as `param`.
    """
    bounds = torch.zeros(list(param.shape)+[2],device=param.device,dtype=param.dtype)
    bounds[...,0] = -torch.inf
    bounds[...,1] = torch.inf
    return bounds


def make_parameter_from_input(input,bounds=None, dtype=None, device=None,bounds_attr_name="bounds"):
    """
    Converts input to a `torch.nn.Parameter` and attaches bounds as an attribute.

    Args:
        input (array-like or torch.Tensor): Input data.
        bounds (torch.Tensor, optional): Bounds to attach to the parameter.
        dtype (torch.dtype, optional): Desired tensor data type.
        device (torch.device, optional): Device to store the parameter on.
        bounds_attr_name (str): Attribute name used to store bounds.

    Returns:
        torch.nn.Parameter: The parameter with bounds attached as an attribute.
    """
    if not torch.is_tensor(input):
        input = torch.tensor(input, dtype=dtype, device=device)

    # If the input tensor has a different dtype or device, move it accordingly
    if dtype is not None or device is not None:
        input = input.to(device=device, dtype=dtype)

    # If the input is not already a Parameter, convert it to one
    if not isinstance(input, torch.nn.Parameter):
        input = torch.nn.Parameter(input)
    
    if bounds is None:
        bounds = make_bounds_from_param(input)
    #input.bounds = bounds
    setattr(input,bounds_attr_name,bounds)    
    return input

def pack_tensors(tensor_list):
    """
    Flattens and concatenates a list of tensors into a single 1D tensor.

    Args:
        tensor_list (list of torch.Tensor or torch.Tensor): Input tensor(s).

    Returns:
        torch.Tensor: A 1D tensor.
    """
    if torch.is_tensor(tensor_list):
        return tensor_list.reshape(-1)
    return torch.cat([t.reshape(-1) for t in tensor_list])

def unpack_tensors(packed_tensor, shapes):
    """
    Unpacks a 1D tensor into a list of tensors with specified shapes.

    Args:
        packed_tensor (torch.Tensor): The flat tensor to unpack.
        shapes (list of tuple): Target shapes for unpacked tensors.

    Returns:
        list of torch.Tensor: Unpacked tensors with original shapes.
    """
    unpacked_tensors = []
    start = 0
    for shape in shapes:
        size = torch.prod(torch.tensor(shape)).item()  # Calculate the size of the tensor
        size = int(max(size,1))
        # Reshape the portion of packed_tensor to the original shape
        tensor = packed_tensor[start:start + size]
        if len(shape) > 0:  # Only reshape if shape is not scalar
            tensor = tensor.reshape(*shape)
        unpacked_tensors.append(tensor)
        start += size  # Move to the next start index
    return unpacked_tensors

def apply_vec_to_params(vec,params,device=None,dtype = None):
    """
    Updates `params` with values from a flat NumPy vector.

    Args:
        vec (np.ndarray): A 1D NumPy array of new parameter values.
        params (list of torch.nn.Parameter): Parameters to update.
        device (torch.device, optional): Device to move data to.
        dtype (torch.dtype, optional): Data type for the new parameter values.

    Raises:
        RuntimeError: If `vec` is not a NumPy array.
    """
    if not isinstance(vec, np.ndarray):
        raise RuntimeError("vec should be a numpy vector")
    params = [elem for elem in params]
    if dtype is None:
        dtype = params[0].dtype
    if device is None:
        device = params[0].device
    unpacked_params = unpack_tensors(torch.tensor(vec,device=device,dtype=dtype), [elem.shape for elem in params])
    with torch.no_grad():
        for k,param in enumerate(params):
            param.data = unpacked_params[k]
    
def set_full_if_nan(input,fill_value):
    """
    Replaces NaNs in input with a specified fill value.

    Args:
        input (np.ndarray): A NumPy array or scalar.
        fill_value (float): Value to use in place of NaNs.

    Returns:
        np.ndarray or float: Modified input with no NaNs.
    """
    if not isinstance(input, np.ndarray):
        raise RuntimeError("set_full_if_nan,input should be a numpy vector")
    
    if len(input.shape) == 0:
        if np.isnan(input):
            return np.array(fill_value)
        else:
            return input
    else:
        if np.isnan(input).any():
            input = np.full_like(input, fill_value)
            return input
        else:
            return input

class ParameterFunHelper():
    """
    Helper class to evaluate a function and its gradient w.r.t. torch parameters.

    Attributes:
        orginal_fun (Callable): Function to be optimized.
        params (list of torch.nn.Parameter): Parameters for optimization.
        nan_fallback (float): Value to return if NaNs are detected.
    """    
    def __init__(self,orginal_fun,params,nan_fallback = float("inf")):
        self.last_x_fun_numpy = None
        self.last_fun_val_numpy = None
        self.last_fun_val_torch = None
        
        self.last_x_grad_numpy = None
        self.last_grad_val_numpy = None
        self.orginal_fun = orginal_fun

        self.params = [param for param in params]
        self.nan_fallback = nan_fallback
        
    def fun(self,x):
        """
        Evaluates the objective function at a given input.

        Args:
            x (np.ndarray): Flat input array.

        Returns:
            float: Function value with NaNs replaced if needed.
        """
        if not self.last_x_fun_numpy is None:
            if (x == self.last_x_fun_numpy).all():
                out = self.last_fun_val_numpy
                out = set_full_if_nan(out,self.nan_fallback)
                return out
        
        
        device = self.params[0].device
        dtype = self.params[0].dtype
        apply_vec_to_params(x,self.params,device,dtype)    
        self.last_x_fun_numpy = copy.deepcopy(x)
        fun_val = self.orginal_fun()
        self.last_fun_val_torch = fun_val
        self.last_fun_val_numpy = set_full_if_nan(fun_val.detach().cpu().numpy(),self.nan_fallback)
        out = self.last_fun_val_numpy
        out = set_full_if_nan(out,self.nan_fallback)
        return out
     
    def jac(self,x):
        """
        Computes the gradient of the objective function at input x.

        Args:
            x (np.ndarray): Flat input array.

        Returns:
            np.ndarray: Gradient with NaNs replaced if needed.
        """
        if not self.last_x_grad_numpy is None:
            if (x == self.last_x_grad_numpy).all():
                out = self.last_grad_val_numpy
                out = set_full_if_nan(out,self.nan_fallback)
                return out
        
        self.fun(x)
        self.last_x_grad_numpy = copy.deepcopy(x)
        dp = grad(self.last_fun_val_torch,inputs=self.params,materialize_grads=True,create_graph=False,retain_graph=False) 
        dp = pack_tensors(dp)
        dp_numpy = dp.detach().cpu().numpy()
        
        self.last_grad_val_numpy = set_full_if_nan(dp_numpy,self.nan_fallback)

        out = dp_numpy
        out = set_full_if_nan(out,self.nan_fallback)
        return out
            
    def fun_jac(self,x):
        """
        Evaluates both function value and gradient at once.

        Args:
            x (np.ndarray): Flat input array.

        Returns:
            Tuple[float, np.ndarray]: Function value and gradient.
        """
        fun_val_numpy = self.fun(x)
        grad_val_numpy = self.jac(x)
        return fun_val_numpy,grad_val_numpy
    """
    def hess(self,x,v):
        if not self.calc_hess:
            raise("ParameterFunHelper: calc_hess was initialized with False!")
        device = self.last_grad_val_torch.device
        dtype = self.last_grad_val_torch.dtype
        self.grad(x)
        v_torch = torch.tensor(v,device=device,dtype=dtype)
        Hv = grad(self.last_grad_val_torch,inputs=self.params,grad_outputs=v_torch,materialize_grads=True,create_graph=False,retain_graph=True)
        Hv_packed = pack_tensors(Hv)
        out = Hv_packed.detach().cpu().numpy()
        out = set_full_if_nan(out,self.nan_fallback)
        print("hess out ",out)
        return out"""

 
def create_fun_and_gradient(merit_fun,params,nan_fallback,device,dtype):
    """
    Wraps a PyTorch merit function and returns a callable that evaluates both
    the function and its gradient in NumPy format.

    Args:
        merit_fun (Callable): PyTorch function to optimize.
        params (list): List of `torch.nn.Parameter` objects.
        nan_fallback (float): Value to use if NaNs are encountered.
        device (torch.device): Target device.
        dtype (torch.dtype): Target dtype.

    Returns:
        Callable: Function that returns (value, gradient) as NumPy arrays.
    """
    def fun_and_gradient(input):
        apply_vec_to_params(input,params,device,dtype)    
        merit_val = merit_fun()
        dmdp = grad(merit_val,inputs=params,materialize_grads=True,create_graph=False,retain_graph=False)
        
        out_merit_val = merit_val.detach().cpu()
        out_dmdp = [elem.detach().cpu() for elem in dmdp]
        out_dmdp = pack_tensors(out_dmdp)
        
        out_dmdp = set_full_if_nan(out_dmdp.numpy(),nan_fallback)
        out_merit_val = set_full_if_nan(out_merit_val.numpy(),nan_fallback)
        
        #print("merit_val: ",out_merit_val)
        return out_merit_val,out_dmdp
    return fun_and_gradient


def remove_bounds(params,bounds_attr_name):
    """
    Removes the bounds attribute from parameters if present.

    Args:
        params (list): List of torch.nn.Parameter objects.
        bounds_attr_name (str): Attribute name of bounds to remove.
    """
    for elem in params:
        if hasattr(elem,bounds_attr_name):
            setattr(elem,bounds_attr_name,None)

def get_bounds(params,bounds_attr_name="bounds"):
    """
    Extracts and concatenates bounds for all parameters.

    Args:
        params (list): List of torch.nn.Parameter objects.
        bounds_attr_name (str): Name of attribute storing bounds.

    Returns:
        np.ndarray: Array of shape (N, 2) with all bounds.
    """
    out = []

    for elem in params:
        if not hasattr(elem,bounds_attr_name):
            bounds = make_bounds_from_param(elem) 
            setattr(elem,bounds_attr_name,bounds)
        tmp = getattr(elem,bounds_attr_name)
        if isinstance(tmp,list):
            tmp = torch.tensor(np.array(tmp),dtype=torch.get_default_dtype())
        if isinstance(tmp,np.ndarray):
            tmp = torch.tensor(tmp,dtype=torch.get_default_dtype())
        
        out += [tmp]
    out = torch.cat([t.reshape(-1,2) for t in out],dim=0)
    out = out.detach().cpu()
    #print("out",out)
    out = np.array(out)
    return out
    
def get_scipy_constraint(constraint,params,nan_fallback):
    """
    Converts a constraint into SciPy-compatible format.

    Args:
        constraint (Constraint): A custom constraint object.
        params (list): List of parameters for the optimization.
        nan_fallback (float): Fallback value for NaNs.

    Returns:
        dict: A dictionary compatible with SciPy constraints.
    """
    param_fun_helper = ParameterFunHelper(constraint.fun,params,nan_fallback)
    param_fun_helper.constraint=True

    scipy_data = {'type': constraint.type,'fun':param_fun_helper.fun,'jac':param_fun_helper.jac}
    return scipy_data


def create_callback(callback_fun,params,device,dtype):
    """
    Wraps a PyTorch callback function for use in SciPy.

    Args:
        callback_fun (Callable): A function taking no arguments.
        params (list): List of parameters to update before calling.
        device (torch.device): Device of the parameters.
        dtype (torch.dtype): Data type of the parameters.

    Returns:
        Callable: A callback function for SciPy optimizers.
    """
    def call_back(input):
        apply_vec_to_params(input,params,device,dtype)    
        return callback_fun()
    return call_back

#nlopt==2.6.2
"""
def global_dual_annealing(fun, 
                          params, 
                          constraints=[],
                          annealing_maxiter=1000,  
                          annealing_initial_temp=5230.0, 
                          annealing_restart_temp_ratio=2e-05, 
                          annealing_visit=2.62, 
                          annealing_accept=-5.0, 
                          annealing_maxfun=10000000.0,
                          bounds_attr_name="bounds",
                          local_tol=1e-6,
                          local_method=None):
    nan_fallback = annealing_maxfun
    
    from .constraints import Constraint

    if isinstance(constraints,Constraint):
        constraints = [constraints]

    if local_method is None:
        if len(constraints) == 0:
            local_method = 'L-BFGS-B'
        else:
            local_method = 'SLSQP'

    if (not local_method == 'SLSQP') and (len(constraints)>0):
        raise RuntimeError("Only for method SLSQP constraints are supported!")
    
    if isinstance(params, torch.nn.Parameter):
        params = [params]
        
    params = [param for param in params if param.requires_grad]

    if len(params) == 0:
        raise RuntimeError("Params is either an empty list or no parameter provided requires_grad!")

    constraints = [get_scipy_constraint(constraint,params,nan_fallback) for constraint in constraints]    

    device = params[0].device
    dtype = params[0].dtype
    
    bounds_numpy = get_bounds(params,bounds_attr_name)
    if np.isinf(bounds_numpy).any():
        raise RuntimeError("All bounds need to be non inf!")
    param_helper_main = ParameterFunHelper(fun,params,nan_fallback)
    #fun_helper = ParameterFunHelper(fun,params,False,nan_fallback)
  
    minimizer_kwargs = dict(
        #func=param_helper_main.fun,
        jac=param_helper_main.jac,
                            constraints=constraints,
                            tol=local_tol,
                            method=local_method)

    

    initial_params = pack_tensors([param.cpu().detach() for param in params])  # Pack the initial params
    
    result = scipy.optimize.dual_annealing(
        func=param_helper_main.fun,
        x0=initial_params, 
        bounds=bounds_numpy,
        maxiter = annealing_maxiter,  
        initial_temp=annealing_initial_temp, 
        restart_temp_ratio = annealing_restart_temp_ratio, 
        visit = annealing_visit, 
        accept = annealing_accept, 
        maxfun=annealing_maxfun,
        minimizer_kwargs=minimizer_kwargs)
    

    apply_vec_to_params(result["x"],[p for p in params],device,dtype)    
    return result
"""

    

def minimize(fun, params, constraints=[], method=None, tol=1e-9,callback=None, options=None,nan_fallback = float("inf"),bounds_attr_name="bounds",save_history=False,call_before_minimize=False):
    """
    Minimizes a function using SciPy's `minimize`, supporting bounds and constraints.

    Args:
        fun (Callable): Objective function.
        params (list): Parameters to optimize.
        constraints (list): List of constraints.
        method (str): SciPy optimization method (e.g., 'L-BFGS-B').
        tol (float): Tolerance for convergence.
        callback (Callable): Optional callback function.
        options (dict): Optimizer options.
        nan_fallback (float): Value to use if function returns NaN.
        bounds_attr_name (str): Name of bounds attribute.
        save_history (bool): If True, saves function values and gradient norms.
        call_before_minimize (bool): Whether to evaluate once before optimization.

    Returns:
        dict: Dictionary containing optimization results (and optionally history).
    """
    from .constraints import Constraint

    if isinstance(constraints,Constraint):
        constraints = [constraints]

    if method is None:
        if len(constraints) == 0:
            method = 'L-BFGS-B'
        else:
            method = 'SLSQP'

    if (not method == 'SLSQP') and (len(constraints)>0):
        raise RuntimeError("Only for method SLSQP constraints are supported!")
    
    if isinstance(params, torch.nn.Parameter):
        params = [params]
        
    params = [param for param in params if param.requires_grad]

    if len(params) == 0:
        raise RuntimeError("Params is either an empty list or no parameter provided requires_grad!")

    constraints = [get_scipy_constraint(constraint,params,nan_fallback) for constraint in constraints]    

    device = params[0].device
    dtype = params[0].dtype
    
    bounds_numpy = get_bounds(params,bounds_attr_name)

    initial_params = pack_tensors([param.cpu().detach() for param in params])  # Pack the initial params
    param_helper_main = ParameterFunHelper(fun,params,nan_fallback)
    #fun_helper = ParameterFunHelper(fun,params,False,nan_fallback)
    
    history = {"fun_vals":[],"fun_grads_norm":[]}

    fun_and_gradient = param_helper_main.fun_jac#create_fun_and_gradient(fun,params,nan_fallback,device=device,dtype=dtype)
    if save_history:
        
        def callback_history(input):
            
            out_merit_val,out_dmdp = fun_and_gradient(input)
            history["fun_vals"] += [out_merit_val]
            history["fun_grads_norm"] += [np.linalg.norm(out_dmdp)]
            
        if callback is None:
            callback = callback_history
        else:
            callback_tmp = create_callback(callback,params,device,dtype)
            def combined_callback(input):
                callback_tmp(input)
                callback_history(input)
            callback = combined_callback
    elif callback is not None:
        callback = create_callback(callback,params,device,dtype)

    initial_params = np.array(initial_params)
    if call_before_minimize:
        fun_and_gradient(initial_params)
        callback(initial_params)
    
    result = scipy.optimize.minimize(
            fun=fun_and_gradient,
            x0=initial_params,
            jac=True,  # Indicates that the function returns both value and gradient
            bounds=bounds_numpy,
            method=method,  # Choose an appropriate method
            tol=tol,
            callback=callback,
            options=options,
            constraints=constraints,
            #hessp=fun_helper.hess
        )
    apply_vec_to_params(result["x"],[p for p in params],device,dtype)    
    result = {key:result[key] for key in result.keys()}
    if len(history["fun_vals"])>0:
        history["fun_vals"] = np.array(history["fun_vals"])
        history["fun_grads_norm"] = np.array(history["fun_grads_norm"])
    
    if save_history:
        result["history"] = history
    return result


def copy_bounds_to_attr_name(params,bounds_attr_name_new,bounds_attr_name_old="bounds",replace_existing_once=True):
    """
    Copies bounds from one attribute name to another.

    Args:
        params (list): List of parameters.
        bounds_attr_name_new (str): New attribute name.
        bounds_attr_name_old (str): Existing attribute name.
        replace_existing_once (bool): Whether to skip copying if new attribute exists.
    """
    def copy_bounds(param,bounds_attr_name_new,bounds_attr_name_old="bounds"):
        bounds = None
        if hasattr(param,bounds_attr_name_old):
            bounds = getattr(param,bounds_attr_name_old)
        else:
            bounds = make_bounds_from_param(param)
        bounds = bounds.clone()
        setattr(param,bounds_attr_name_new,bounds)
    if isinstance(params,nn.Parameter):
        params = [params]
    params = [param for param in params]
    for param in params:
        if (not replace_existing_once) and (hasattr(param,bounds_attr_name_new)):
            continue
        else:
            copy_bounds(param,bounds_attr_name_new,bounds_attr_name_old)


def set_bounds_from_params_mask(params,mask:list|torch.Tensor,bounds_attr_name_new,bounds_attr_name_old="bounds"):
    """
    Sets bounds for parameters based on a mask. Parameters with `mask=False`
    get fixed bounds (equal lower and upper bounds).

    Args:
        params (list): List of parameters.
        mask (list or torch.Tensor): Mask specifying which elements are free.
        bounds_attr_name_new (str): Attribute name to store new bounds.
        bounds_attr_name_old (str): Attribute name to read old bounds from.
    """
    def set_new_bounds_from_param_mask(param,mask,bounds_attr_name_new,bounds_attr_name_old="bounds"):
        bounds = None
        if hasattr(param,bounds_attr_name_old):
            bounds = getattr(param,bounds_attr_name_old)
        else:
            bounds = make_bounds_from_param(param)
        bounds = bounds.clone()
        bounds_shape = bounds.shape
        mask = mask.reshape(-1)
        bounds = bounds.reshape(-1,2)
        data = param.data.clone()
        data = data.reshape(-1)
        #print("shapes",mask.shape,(mask==False).shape,bounds.shape)
        mask_false = mask==False
        bounds[mask_false,0] = data[mask_false]
        bounds[mask_false,1] = data[mask_false]
        bounds = bounds.reshape(*bounds_shape)
        setattr(param,bounds_attr_name_new,bounds)


    if isinstance(params,nn.Parameter):
        params = [params]
    params = [param for param in params]
    if isinstance(mask,(np.ndarray)) or torch.is_tensor(mask):
        mask = [mask]

    for k in range(len(params)):
        set_new_bounds_from_param_mask(params[k],mask[k],bounds_attr_name_new=bounds_attr_name_new,bounds_attr_name_old=bounds_attr_name_old)


