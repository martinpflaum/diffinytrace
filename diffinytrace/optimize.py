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

import scipy
import scipy.optimize
from .utils.autograd import grad
import torch
import numpy as np
import torch.nn as nn
import copy
def make_bounds_from_param(param):
    bounds = torch.zeros(list(param.shape)+[2],device=param.device,dtype=param.dtype)
    bounds[...,0] = -torch.inf
    bounds[...,1] = torch.inf
    return bounds


def make_parameter_from_input(input,bounds=None, dtype=None, device=None,bounds_attr_name="bounds"):
    # If input is not a tensor, convert it to a tensor

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
    Pack a list of tensors into a single 1D tensor.

    Parameters:
    tensor_list (list of torch.Tensor): List of tensors to be packed.

    Returns:
    torch.Tensor: A single 1D tensor containing all elements of the input tensors.
    """
    if torch.is_tensor(tensor_list):
        return tensor_list.reshape(-1)
    return torch.cat([t.reshape(-1) for t in tensor_list])

def unpack_tensors(packed_tensor, shapes):
    """
    Unpack a single 1D tensor into a list of tensors.

    Parameters:
    packed_tensor (torch.Tensor): The packed 1D tensor.
    shapes (list of tuple): List of shapes corresponding to each original tensor.

    Returns:
    list of torch.Tensor: A list of unpacked tensors with their original shapes.
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
    if not isinstance(vec, np.ndarray):
        raise RuntimeError("vec should be a numpy vector")
    params = [elem for elem in params]
    if dtype is None:
        dtype = params[0].dtype
    if device is None:
        device = params[0].device
    unpacked_params = unpack_tensors(torch.tensor(vec,device=device,dtype=dtype), [elem.shape for elem in params])
    #unpacked_params = [torch.tensor(elem,device=device,dtype=dtype) for elem in unpacked_params]
    with torch.no_grad():
        for k,param in enumerate(params):
            param.data = unpacked_params[k]
    
def set_full_if_nan(input,fill_value):
    #TODO implement axis - not necassary
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
    for elem in params:
        if hasattr(elem,bounds_attr_name):
            setattr(elem,bounds_attr_name,None)

def get_bounds(params,bounds_attr_name="bounds"):
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
        param_fun_helper = ParameterFunHelper(constraint.fun,params,nan_fallback)
        param_fun_helper.constraint=True

        scipy_data = {'type': constraint.type,'fun':param_fun_helper.fun,'jac':param_fun_helper.jac}
        return scipy_data


def create_callback(callback_fun,params,device,dtype):
    def call_back(input):
        apply_vec_to_params(input,params,device,dtype)    
        return callback_fun()
    return call_back

#nlopt==2.6.2

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


    

def minimize(fun, params, constraints=[], method=None, tol=1e-9,callback=None, options=None,nan_fallback = float("inf"),bounds_attr_name="bounds",save_history=False,call_before_minimize=False):
    #hessp=None,
    #constraints=(),
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


