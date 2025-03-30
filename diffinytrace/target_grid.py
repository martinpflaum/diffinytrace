"""
MIT License

Copyright (c) 2025 Martin Pflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Grid():
    def __init__(self,y_range,x_range,y_grid_size,x_grid_size):
        super().__init__()
        self.y_range = np.array(y_range)
        self.x_range = np.array(x_range)
        

        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size
        self.x_delta = (self.x_range[1]-self.x_range[0])/x_grid_size
        self.y_delta = (self.y_range[1]-self.y_range[0])/y_grid_size
        
    """def get_all_midpoints(self):
        _x = self.__get_x_middle()
        _y = self.__get_y_middle()
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
        points = torch.zeros((x.shape[0],2))        
        points[:,0] = x
        points[:,1] = y
        return points
        
    """
    def get_area(self):
        return (self.x_range[1]-self.x_range[0])*(self.y_range[1]-self.y_range[0])
    
    def get_pixel_area(self):        
        return self.x_delta*self.y_delta

    def get_yi_xi(self,local_points,round_to_bounds=True):
        if len(local_points.shape) != 2 or local_points.shape[1] != 2:
            raise RuntimeError("The local_points must be in local coordinates and of shape [#points,2]")
        local_points = local_points.detach()
        
        ref_x = (local_points[:,0]-self.x_range[0])/self.x_delta
        ref_y = (local_points[:,1]-self.y_range[0])/self.y_delta
        
        xi = torch.floor(ref_x).long()
        yi = torch.floor(ref_y).long()
        

        valid = (xi>=self.x_grid_size).float()+(xi<0).float()+(yi>=self.y_grid_size).float()+(yi<0).float()
        valid = valid==0.0

        if ((xi>=self.x_grid_size).any() or (xi<0).any() or (yi>=self.y_grid_size).any() or (yi<0).any()):
            yi = torch.clamp(yi,min=0,max=(self.y_grid_size-1))
            xi = torch.clamp(xi,min=0,max=(self.x_grid_size-1))
            #else:
            #    raise RuntimeError(f"Target grid ERROR: points out of bounds! max xi={xi.max()}, min xi={xi.min()},max yi={yi.max()}, min yi={yi.min()}")
        
        if round_to_bounds:
            return (yi,xi)
        else:
            return (yi,xi),valid

    def get_k(self,local_points,round_to_bounds=True):
        if round_to_bounds:
            yi,xi = self.get_yi_xi(local_points,round_to_bounds=round_to_bounds)
            return (yi*self.x_grid_size+xi).long()
        else:
            (yi,xi),valid = self.get_yi_xi(local_points,round_to_bounds=round_to_bounds)
            k = (yi*self.x_grid_size+xi).long()
            return k,valid
        
    def map_matrix_to_ray(self,local_points,old_matrix):
        device = local_points.device
        dtype = local_points.dtype
        k = self.get_k(local_points)
        return old_matrix.reshape(-1)[k].reshape(local_points.shape[0],*old_matrix.shape[2:]) 
        
    def sum(self,local_points,values,old_matrix = None,round_to_bounds=False):
        device = local_points.device
        dtype = local_points.dtype
        out = torch.zeros((self.x_grid_size*self.y_grid_size),device=device,dtype=dtype)
        if not old_matrix is None:
           out = old_matrix
        
        if round_to_bounds:
            k = self.get_k(local_points,round_to_bounds)
            out.scatter_add_(0,k,values)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out
        else:
            k,valid = self.get_k(local_points,round_to_bounds)
            values = values[valid]
            k = k[valid]
            out.scatter_add_(0,k,values)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out


    def prod(self,local_points,values,old_matrix = None,round_to_bounds=False):
        device = local_points.device
        dtype = local_points.dtype
        out = torch.ones((self.y_grid_size*self.x_grid_size),device=device,dtype=dtype)
        if not old_matrix is None:
           out = old_matrix
        if round_to_bounds:
            k = self.get_k(local_points,round_to_bounds)
            out.scatter_reduce_(0,k,values,reduce='prod')
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out
        else:
            k,valid = self.get_k(local_points,round_to_bounds)
            values = values[valid]
            k = k[valid]
            out.scatter_reduce_(0,k,values,"prod")
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out

    def mean(self,local_points,values,old_matrix = None,round_to_bounds=False):
        device = local_points.device
        dtype = local_points.dtype
        out = torch.zeros((self.y_grid_size*self.x_grid_size),device=device,dtype=dtype)
        if not old_matrix is None:
           out = old_matrix
        if round_to_bounds:
            k = self.get_k(local_points)
            out.scatter_reduce_(0,k,values,reduce='mean',include_self=False)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out
        else:
            k,valid = self.get_k(local_points,round_to_bounds)
            values = values[valid]
            k = k[valid]
            out.scatter_reduce_(0,k,values,reduce='mean',include_self=False)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out

        

    def __get_args(self,M,b,v):
        device = v.device
        dtype = v.dtype
        M_argmin = torch.full((self.y_grid_size*self.x_grid_size,), -1, dtype=torch.long,device=device)
        mask = (v == M[b])
        indices = torch.arange(len(v))
        M_argmin.scatter_(0, b[mask], indices[mask])
        return M_argmin

    def min(self,local_points,values,old_matrix = None,return_args=False):
        device = local_points.device
        dtype = local_points.dtype
        out = torch.full((self.y_grid_size*self.x_grid_size,),float('inf'),device=device,dtype=dtype)
        if not old_matrix is None:
           out = old_matrix
        k = self.get_k(local_points)
        out.scatter_reduce_(0,k,values,reduce='amin')
        if return_args:
            out_args = self.__get_args(out,k,values)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            out_args = out_args.reshape(self.y_grid_size,self.x_grid_size)
            return out,out_args
        else:
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out

    def max(self,local_points,values,old_matrix = None,return_args=False):
        device = local_points.device
        dtype = local_points.dtype
        out = torch.full((self.y_grid_size*self.x_grid_size,),float('-inf'),device=device,dtype=dtype)
        if not old_matrix is None:
           out = old_matrix
        k = self.get_k(local_points)
        out.scatter_reduce_(0,k,values,reduce='amax')
        if return_args:
            out_args = self.__get_args(out,k,values)
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            out_args = out_args.reshape(self.y_grid_size,self.x_grid_size)
            return out,out_args
        else:
            out = out.reshape(self.y_grid_size,self.x_grid_size)
            return out

    def __get_x_middle(self):
        x_middle = self.x_delta*0.5+torch.arange(0,self.x_grid_size)*self.x_delta+self.x_range[0]
        return x_middle
    
    def __get_y_middle(self):
        y_middle = self.y_delta*0.5+torch.arange(0,self.y_grid_size)*self.y_delta+self.y_range[0]
        return y_middle
    
    def nearest(self,local_points,return_args=False):
        x_middle = self.__get_x_middle()
        y_middle = self.__get_y_middle()

        yi,xi = self.get_yi_xi(local_points)

        xdiff = (x_middle[xi]-local_points[:,0])**2.0
        ydiff = (y_middle[yi]-local_points[:,1])**2.0
        l2diff = xdiff+ydiff
        return self.min(local_points,l2diff,return_args=return_args)

    def get_pixel_centers(self):
        x_middle = self.__get_x_middle()
        y_middle = self.__get_y_middle()

        grid_y,grid_x = torch.meshgrid(y_middle, x_middle, indexing='ij')
        V = torch.cat([grid_x.reshape(-1,1),grid_y.reshape(-1,1)],dim=-1)
        return V.reshape(self.y_grid_size,self.x_grid_size,2)

    def get_nearest_ray(self,local_points):
        device = local_points.device
        dtype = local_points.dtype
        local_points = local_points.detach()
        with torch.no_grad():
            
            W = local_points
            V = self.get_pixel_centers().reshape(-1,2)
            
            
            nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            nn_model.fit(W) # Fit the model on W (the smaller collection)
            distances, indices = nn_model.kneighbors(V)
            indices = indices.flatten()
            #out = flat_args[indices].reshape(self.y_grid_size,self.x_grid_size)
            out = torch.tensor(indices.reshape(self.y_grid_size,self.x_grid_size),device = device)
            return out
            #implment nearest_ray


class GridSquare(Grid):
    def __init__(self,aperture_radius,grid_size):
        super().__init__(\
            [-aperture_radius,aperture_radius],\
            [-aperture_radius,aperture_radius],grid_size,grid_size)
        
#%%

"""import torch
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

grid_y,grid_x = torch.meshgrid(y, x, indexing='ij')
mid_pos = torch.cat([grid_x.reshape(-1,1),grid_y.reshape(-1,1)],dim=-1)
mid_pos

ix = torch.arange(0,x.shape[0])
iy = torch.arange(0,y.shape[0])
grid_ix, grid_iy = torch.meshgrid(iy, ix, indexing='ij')
mid_idx = grid_iy.reshape(-1,1)*ix.shape[0]+grid_ix.reshape(-1,1)
        
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Example: Initialize large V and W arrays (Replace with real data in practice)
V = np.random.randn(1000000, 128)  # V is very large (e.g., 1,000,000 vectors of 128 dimensions)
W = np.random.randn(1000, 128)     # W is smaller (e.g., 1,000 vectors of 128 dimensions)

# Initialize NearestNeighbors model with an efficient algorithm for large datasets
# 'auto' lets sklearn choose the best algorithm based on the input data
nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')

# Fit the model on W (the smaller collection)
nn_model.fit(W)

# Query the model with V to find the closest neighbors in W
distances, indices = nn_model.kneighbors(V)

# distances: The distance to the nearest neighbor in W for each vector in V
# indices: The index of the nearest neighbor in W for each vector in V
"""        

#%%


