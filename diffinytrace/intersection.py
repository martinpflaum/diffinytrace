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
from .utils.autograd import grad
from .config import get_max_iterations,get_tolerance,get_damping_factor,get_show_iteration_count

class SemiFunctionalModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_functional_param_args(self):
        raise NotImplementedError("params_list not implemented")

    @staticmethod
    def functional(O,*params):
        raise NotImplementedError("functional not implemented")

def cat_semi_functionals(functional_modules):
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
    out = []
    for elem in semi_functional_module_list:
        out += elem.get_functional_param_args()
    return out


def construct_surface_and_normal_func(semi_functional_module_list):
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
    s_dsd = construct_surface_and_normal_func(semi_functional_module_list)
    args = get_functional_param_args(semi_functional_module_list)
    return s_dsd,args

        
        
    
class CustomAutogradRule_t(torch.autograd.Function):
    @staticmethod
    def forward(ctx,O,D,surface_and_normal_func,t_detached,*param_args):
        ctx.save_for_backward(O,D,t_detached,*param_args)
        ctx.surface_and_normal_func = surface_and_normal_func
        return t_detached

    @staticmethod
    def backward(ctx, grad_outputs):
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