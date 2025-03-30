# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import typing
import torch

def grad(
    outputs: torch.types._TensorOrTensors,
    inputs: torch.types._TensorOrTensorsOrGradEdge,
    grad_outputs: typing.Optional[torch.types._TensorOrTensors] = None,
    retain_graph: typing.Optional[bool] = True,
    create_graph: bool = True,
    only_inputs: bool = True,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
    remove_no_grad_outputs: bool = True
):
    
    
    if torch.is_tensor(inputs):
        inputs = [inputs]
    inputs = [elem for elem in inputs]

    
    if remove_no_grad_outputs:
        if torch.is_tensor(grad_outputs) or torch.is_tensor(outputs):
            if torch.is_tensor(outputs):
                if not outputs.requires_grad:
                    if torch.is_tensor(inputs):
                        raise RuntimeError("this branch should not be called!")
                    else:
                        out = []
                        for elem in inputs:
                            if torch.is_tensor(elem):
                                out += [torch.zeros_like(elem)]
                            else:
                                out += [None]
                        return out
        else:                
            _grad_outputs = [elem for elem in grad_outputs]
            
            new_grad_outputs = []
            new_outputs = []
            for k,elem in enumerate(outputs):
                if torch.is_tensor(elem):
                    if elem.requires_grad:
                        grad_elem = _grad_outputs[k]
                        new_outputs += [elem]
                        new_grad_outputs += [grad_elem]
            grad_outputs = new_grad_outputs
            outputs = new_outputs
    
    inputs_requires_grad = []
    back_map_input = {}
    inputs_map_i = 0
    for k,param in enumerate(inputs):
        #param = inputs[k]
    
        if param is None:
            continue
        if param.requires_grad:
            inputs_requires_grad += [param]
            back_map_input[k] = inputs_map_i  
            inputs_map_i += 1
    grad_tmp = []
    if len(inputs_requires_grad)!=0:
        grad_tmp = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs_requires_grad,
                                    grad_outputs=grad_outputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph,
                                    only_inputs=only_inputs,
                                    allow_unused=True,
                                    is_grads_batched=is_grads_batched,
                                    materialize_grads=materialize_grads)
    else:
        
        pass
    grad = [None for input in inputs]
    for k in range(len(grad)):
        if k in back_map_input.keys():
            inputs_map_i = back_map_input[k]  
            grad[k] = grad_tmp[inputs_map_i]
        else:
            if materialize_grads:
                if inputs[k] is None:
                    grad[k] = None 
                else:
                    grad[k] = torch.zeros_like(inputs[k])
            else:
                grad[k] = None 

    return grad
