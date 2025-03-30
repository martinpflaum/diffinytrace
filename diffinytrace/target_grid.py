# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

"""
This module implements grid-based spatial aggregation for ray optics.

Classes:
    - Grid: Represents a 2D grid for spatial aggregation and statistics.
    - GridSquare: Square variant of Grid for symmetric apertures.

Functions:
    - (none at top level)

Example:
    >>> grid = Grid([0, 1], [0, 1], 10, 10)
    >>> area = grid.get_area()
"""

import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
import numpy as np

class Grid():
    """
    Represents a 2D grid over a rectangular area with aggregation and indexing utilities.

    Args:
        y_range (tuple[float, float]): The range in y-direction, as (y_min, y_max).
        x_range (tuple[float, float]): The range in x-direction, as (x_min, x_max).
        y_grid_size (int): Number of grid cells in y-direction.
        x_grid_size (int): Number of grid cells in x-direction.
    """
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
        """
        Computes the total area of the grid.

        Returns:
            float: Total area of the grid.

        .. math::
            A = (x_{max} - x_{min}) \cdot (y_{max} - y_{min})
        """
        return (self.x_range[1]-self.x_range[0])*(self.y_range[1]-self.y_range[0])
    
    def get_pixel_area(self):     
        """
        Returns the area of a single pixel/grid cell.

        Returns:
            float: Area of a single grid cell.

        .. math::
            A_{pixel} = \Delta x \cdot \Delta y
        """   
        return self.x_delta*self.y_delta

    def get_yi_xi(self,local_points,round_to_bounds=True):
        """
        Converts 2D local coordinates to integer grid indices.

        Args:
            local_points (torch.Tensor): Tensor of shape (N, 2) representing 2D points.
            round_to_bounds (bool): If True, clamps indices to stay within grid bounds. If False, returns a mask indicating valid indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors (yi, xi) of shape (N,).
        """
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
        """
        Maps local coordinates to flattened grid indices.

        Args:
            local_points (torch.Tensor): Tensor of shape (N, 2).
            round_to_bounds (bool): Whether to clamp indices to grid bounds.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If `round_to_bounds` is True: Tensor of shape (N,).
            - Otherwise: Tuple (indices, validity_mask).
        """
        if round_to_bounds:
            yi,xi = self.get_yi_xi(local_points,round_to_bounds=round_to_bounds)
            return (yi*self.x_grid_size+xi).long()
        else:
            (yi,xi),valid = self.get_yi_xi(local_points,round_to_bounds=round_to_bounds)
            k = (yi*self.x_grid_size+xi).long()
            return k,valid
        
    def map_matrix_to_ray(self,local_points,old_matrix):
        """
        Maps a matrix defined on the grid to the given local points.

        Args:
            local_points (torch.Tensor): Points of shape (N, 2).
            old_matrix (torch.Tensor): Matrix of shape (H, W, ...).

        Returns:
            torch.Tensor: Resampled matrix values of shape (N, ...).
        """
        device = local_points.device
        dtype = local_points.dtype
        k = self.get_k(local_points)
        return old_matrix.reshape(-1)[k].reshape(local_points.shape[0],*old_matrix.shape[2:]) 
        
    def sum(self,local_points,values,old_matrix = None,round_to_bounds=False):
        """
        Sums values over the grid based on point locations.

        Args:
            local_points (torch.Tensor): Points of shape (N, 2).
            values (torch.Tensor): Values of shape (N,) or (N, D).
            old_matrix (torch.Tensor or None): Previous result for accumulation.
            round_to_bounds (bool): Clamp indices to bounds if True.

        Returns:
            torch.Tensor: Aggregated result of shape (H, W).
        """
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
        """
        Finds the nearest pixel for each local point using L2 distance.

        Args:
            local_points (torch.Tensor): Tensor of shape (N, 2).
            return_args (bool): If True, also return indices.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Minimum squared distances, optionally with indices.
        """
        x_middle = self.__get_x_middle()
        y_middle = self.__get_y_middle()

        yi,xi = self.get_yi_xi(local_points)

        xdiff = (x_middle[xi]-local_points[:,0])**2.0
        ydiff = (y_middle[yi]-local_points[:,1])**2.0
        l2diff = xdiff+ydiff
        return self.min(local_points,l2diff,return_args=return_args)

    def get_pixel_centers(self):
        """
        Returns the 2D center coordinates of each grid cell.

        Returns:
            torch.Tensor: Tensor of shape (H, W, 2) with pixel center coordinates.
        """
        x_middle = self.__get_x_middle()
        y_middle = self.__get_y_middle()

        grid_y,grid_x = torch.meshgrid(y_middle, x_middle, indexing='ij')
        V = torch.cat([grid_x.reshape(-1,1),grid_y.reshape(-1,1)],dim=-1)
        return V.reshape(self.y_grid_size,self.x_grid_size,2)

    def get_nearest_ray(self,local_points):
        """
        Finds the index of the nearest ray for each grid cell using `sklearn.neighbors.NearestNeighbors`.

        Args:
            local_points (torch.Tensor): Tensor of shape (N, 2) representing sampled rays.

        Returns:
            torch.Tensor: Tensor of shape (H, W) with ray indices.
        """
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
    """
    Convenience class for square grids centered at the origin.

    Args:
        aperture_radius (float): Half-width of the square domain.
        grid_size (int): Number of grid points in each direction.
    """
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


