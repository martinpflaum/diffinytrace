# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = ["Integrator", "Cube","Disc","IntegrationMethod"]

import torch
import numpy as np
import math
from scipy.stats import qmc
from enum import Enum

mersenne_twister = np.random.Generator(np.random.MT19937(seed=12345))

class IntegrationMethod(Enum):
    SIMPSON = "simpson"
    MIDPOINT = "midpoint"
    MONTE_CARLO = "monte_carlo"
    SOBOL = "sobol"
    SOBOL_POW2 = "sobol_pow2"

def check_2val(num_points):
    num_points = np.array(num_points)
    if len(num_points.shape)>0:
        return num_points[0]*num_points[1]
    return num_points

class Integrator():
    def __init__(self):
        pass

    def sample(self, num_points: int | list[int], method: IntegrationMethod) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points and weights using the specified method.
        Args:
            num_points (int or list): Number of points in each dimension.
            method (str): The integration method to use. Options are 'simpson', 'midpoint', 'monte_carlo', 'sobol', and 'sobol_pow2'.
        Returns:
            tuple: A tuple containing the sampled points and their corresponding weights.
        """
        raise NotImplementedError("sample() not implemented")

    def in_bounds(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("in_bounds() not implemented")

    def get_volume(self) -> float:
        raise NotImplementedError("get_volume() not implemented")

     
class Cube(Integrator):
    """
    Integrator for a multi-dimensional cube (hyperrectangle).

    Args:
        bounds (array-like): The bounds for each dimension of the cube. Should be a list or array of shape (n_dim, 2),
            where each row specifies [lower_bound, upper_bound] for a dimension.

    Example:
        >>> cube = dit.integrators.Cube([[0, 1], [0, 1]])
        >>> points, weights = cube.sample([10, 10], method=IntegrationMethod.MIDPOINT)
        >>> volume = cube.get_volume()
        >>> all_in_bounds = cube.in_bounds(points)
        >>> print("Sampled points:", points)
        >>> print("Integration weights:", weights)
        >>> print("Cube volume:", volume)
        >>> print("All points in bounds:", all_in_bounds)
    """
    
    def __init__(self,bounds):
        super().__init__()
        bounds = np.array(bounds)
        if len(bounds.shape)==1:
            bounds = np.array([bounds])
        
        self.bounds = torch.tensor(bounds)
        if len(self.bounds.shape)!=2:
            raise ValueError("len(self.bounds.shape)==2 must hold true!")

    def sample(self, num_points: int | list[int], method: IntegrationMethod = IntegrationMethod.MIDPOINT) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Sample points and weights using the specified method.
        
        Args:
            num_points (int or list): Number of points in each dimension.
            method (str): The integration method to use. Options are 'simpson', 'midpoint', 'monte_carlo', 'sobol', and 'sobol_pow2'.
        
        Returns:
            tuple: A tuple containing the sampled points and their corresponding weights.
        """
        if not isinstance(method, str):
            method = str(method.value)

        
        if method == 'simpson':
            return self._sample_simpson(num_points)
        elif method == 'midpoint':
            return self._sample_midpoint(num_points)
        elif method == 'monte_carlo':
            return self._sample_monte_carlo(num_points)
        elif method == 'sobol':
            return self._sample_sobol(num_points,False)
        elif method == 'sobol_pow2':
            return self._sample_sobol(num_points,True)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def in_bounds(self, x:torch.Tensor) -> torch.Tensor:
        out = torch.ones(x.shape[0],device=x.device,dtype=torch.bool).float()
        for k in range(self.bounds.shape[0]):
            out = out*((self.bounds[k,0]<=x[:,k]).float())*((x[:,k]<=self.bounds[k,1]).float())
        out = out==1.0
        return out

    def get_volume(self) -> float:
        """
        Returns:
            float: Volume of the Cube.
        """
        
        volume = torch.prod(self.bounds[:,1]-self.bounds[:,0])
        return volume
    
    def _sample_midpoint(self, num_points):
        """
        Sample points and weights using the midpoint rule.

        Args:
            num_points (list or array): Number of points in each dimension.

        Returns:
            sampled_points (torch.Tensor): Tensor of sampled points.
            weights (torch.Tensor): Tensor of weights associated with each point.
        """
        # Ensure num_points matches the number of dimensions in bounds
        num_points = np.array(num_points)
        if len(num_points) != self.bounds.shape[0]:
            raise ValueError("Using midpoint sampling expected num_points to match the number of dimensions.")
        
        midpoints = []
        
        # Calculate the midpoints for each dimension
        for i in range(self.bounds.shape[0]):
            lower_bound, upper_bound = self.bounds[i]
            # Calculate the size of each interval (dx) for the dimension
            dx = (upper_bound - lower_bound) / num_points[i]
            # Compute the midpoints in this dimension
            points = torch.linspace(lower_bound + dx / 2.0, upper_bound - dx / 2.0, num_points[i])
            midpoints.append(points)

        # Create a meshgrid of midpoints for all dimensions
        grid = torch.meshgrid(*midpoints, indexing='ij')
        sampled_points = torch.stack(grid, dim=-1).reshape(-1, self.bounds.shape[0])
        
        # Compute the weights based on the volume of each subregion
        weights = torch.ones(sampled_points.shape[0], dtype=torch.float32)

        for i in range(self.bounds.shape[0]):
            lower_bound, upper_bound = self.bounds[i]
            dx = (upper_bound - lower_bound) / num_points[i]
            # Each dimension contributes its own weight
            weights *= dx  # Multiply the weights by the width of the intervals

        return sampled_points, weights

            
    def _sample_simpson(self, num_points):
        # Ensure num_points matches the expected number of dimensions
        num_points = np.array(num_points)
        if len(num_points) != self.bounds.shape[0]:
            raise ValueError("Using Simpson's rule expected num_points to have the same number of entries as dimensions.")
        for elem in num_points:
            if elem % 2 == 0:
                raise ValueError("Simpson's rule only takes an odd number of points!")

        # Create sample points and weights
        sample_points = []
        weights = []

        for i in range(self.bounds.shape[0]):
            lower_bound, upper_bound = self.bounds[i]
            dx = (upper_bound - lower_bound) / (num_points[i] - 1)
            x = torch.linspace(lower_bound, upper_bound, num_points[i])  # Include endpoints
            sample_points.append(x)

            # Weights for Simpson's rule
            w = torch.ones(num_points[i])  # Initialize weights
            w[1:-1:2] *= 4  # Odd indices
            w[2:-1:2] *= 2  # Even indices
            weights.append(w * dx / 3)  # Multiply by dx/3

        # Create a meshgrid of sample points
        grid = torch.meshgrid(*sample_points)
        sampled_points = torch.stack(grid, dim=-1).reshape(-1, self.bounds.shape[0])

        # Total weight for each point
        total_weights = weights[0]
        for w in weights[1:]:
            total_weights = torch.ger(total_weights, w).reshape(-1)  # Use outer product and flatten


        points, weights = sampled_points, total_weights.reshape(-1)  # Ensure weights are a flat array
        points = points.to(torch.get_default_dtype())
        weights = weights.to(torch.get_default_dtype())
        return points, weights

    def _sample_trapezoidal(self, num_points):
        raise RuntimeError("DO NOT USE THIS method. It's more or less the same as midpoint rule....")
        num_points = np.array(num_points)
        if len(num_points) != self.bounds.shape[0]:
            raise ValueError("Using trapezoidal sampling expected num_points to match the number of dimensions.")
        
        sample_points = []
        weights = []

        for i in range(self.bounds.shape[0]):
            lower_bound, upper_bound = self.bounds[i]
            dx = (upper_bound - lower_bound) / (num_points[i] - 1)
            x = torch.linspace(lower_bound, upper_bound, num_points[i])
            sample_points.append(x)

            # Weights for the trapezoidal rule
            w = torch.ones(num_points[i])
            w[0] /= 2  # First point weight
            w[-1] /= 2  # Last point weight
            weights.append(w * dx)  # Multiply by dx to get the correct area contribution

        # Create a meshgrid of sample points
        grid = torch.meshgrid(*sample_points, indexing='ij')
        sampled_points = torch.stack(grid, dim=-1).reshape(-1, self.bounds.shape[0])

        # Calculate total weights
        total_weights = weights[0]
        for w in weights[1:]:
            total_weights = total_weights.unsqueeze(-1) * w.unsqueeze(0)  # Use broadcasting to multiply correctly

        points, weights = sampled_points, total_weights.reshape(-1)
        points = points.to(torch.get_default_dtype())
        weights = weights.to(torch.get_default_dtype())
        return points, weights
    
    def _sample_monte_carlo(self, num_points):
        num_points = check_2val(num_points)
        if len(num_points.shape)!=0:
            raise ValueError("num_points for monte_carlo needs to be a scalar")
        """Sample points uniformly using the Monte Carlo method."""
        points = torch.empty((num_points, self.bounds.shape[0]))
        
        for i in range(self.bounds.shape[0]):
            # Generate random points uniformly within the bounds for each dimension
            rand_points = mersenne_twister.uniform(0,1,size=num_points)
            rand_points = torch.tensor(rand_points, dtype=torch.float32)
            points[:, i] = rand_points * (self.bounds[i, 1] - self.bounds[i, 0]) + self.bounds[i, 0]
        
        # Calculate the volume of the cube
        volume = torch.prod(self.bounds[:,1]-self.bounds[:,0])
        constant_multi = volume/float(num_points)
        weights = torch.full((int(num_points),),fill_value=constant_multi)
        
        points = points.to(torch.get_default_dtype())
        weights = weights.to(torch.get_default_dtype())
        return points, weights

    def _sample_sobol(self, num_points,is_pow2):
        num_points = check_2val(num_points)
        if len(num_points.shape)!=0:
            raise ValueError("num_points for sobol needs to be a scalar")
        """Sample points using the Sobol sequence method."""
        points = None
        num_points_log2 = np.log2(num_points)
        if round(num_points_log2,0) != num_points_log2:
            if is_pow2:
                raise RuntimeError("round(num_points_log2,0) != num_points_log2"+ f",num_points_log2=={num_points_log2}")
            
            sobol = torch.quasirandom.SobolEngine(dimension=self.bounds.shape[0], scramble=True)
            points = sobol.draw(num_points,dtype=torch.float32)
        else:
            sampler = qmc.Sobol(d=self.bounds.shape[0], scramble=True)
            points = sampler.random_base2(m=int(num_points_log2))
            points = torch.tensor(points)
        # Scale points according to the cube bounds
        scaled_points = points * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        
        # Calculate the volume of the cube
        volume = torch.prod(self.bounds[:,1]-self.bounds[:,0])
        constant_multi = volume/float(num_points)
        weights = torch.full((int(num_points),),fill_value=constant_multi)
        
        points = points.to(torch.get_default_dtype())
        weights = weights.to(torch.get_default_dtype())
        return scaled_points, weights



class Disc(Integrator):
    """
    Integrator for a 2D disc (circle).

    Args:
        radius (float): The radius of the disc.

    Example:
        >>> disc = dit.integrators.Disc(1.0)
        >>> points, weights = disc.sample(2**4, method="sobol_pow2")
        >>> volume = disc.get_volume()
        >>> all_in_bounds = disc.in_bounds(points)
        >>> print("Sampled points:", points)
        >>> print("Integration weights:", weights)
        >>> print("Disc area:", volume)
        >>> print("All points in bounds:", all_in_bounds)
    """

    def __init__(self,radius):
        self.radius = float(radius)

    def sample(self, num_points: int | list[int], method: IntegrationMethod = IntegrationMethod.SOBOL) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points and weights using the specified method.
        
        Args:
            num_points (int or list): Number of points in each dimension.
            method (str): The integration method to use. Options are 'simpson', 'midpoint', 'monte_carlo', 'sobol', and 'sobol_pow2'.
        Returns:
            tuple: A tuple containing the sampled points and their corresponding weights.
        """
        if not isinstance(method, str):
            method = str(method.value)

        if method == 'simpson':
            return self._sample_simpson(num_points)
        elif method == 'monte_carlo':
            return self._sample_monte_carlo(num_points)
        elif method == 'sobol':
            return self._sample_sobol(num_points,False)
        elif method == 'sobol_pow2':
            return self._sample_sobol(num_points,True)
        elif method == 'midpoint':
            return self._sample_midpoint(num_points)
        else:
            raise ValueError(f"Unknown integration method: {method}")
    
    def _sample_midpoint(self, num_points):
        #raise RuntimeError("midpoint rule not implemted for disc")    
        num_points = np.array(num_points)
        
        midpoints = []
        
        # Calculate the midpoints for each dimension
        lower_bound, upper_bound = [-self.radius,self.radius]
        for i in range(2):
            # Calculate the size of each interval (dx) for the dimension
            dx = (upper_bound - lower_bound) / num_points[i]
            # Compute the midpoints in this dimension
            points = torch.linspace(lower_bound + dx / 2.0, upper_bound - dx / 2.0, num_points[i])
            midpoints.append(points)

        # Create a meshgrid of midpoints for all dimensions
        grid = torch.meshgrid(*midpoints, indexing='ij')
        sampled_points = torch.stack(grid, dim=-1).reshape(-1, 2)
        
        # Compute the weights based on the volume of each subregion
        weights = torch.ones(sampled_points.shape[0], dtype=torch.float32)

        lower_bound, upper_bound = [-self.radius,self.radius]
        for i in range(2):
            dx = (upper_bound - lower_bound) / num_points[i]
            # Each dimension contributes its own weight
            weights *= dx  # Multiply the weights by the width of the intervals
        in_bounds = self.in_bounds(sampled_points)
        sampled_points = sampled_points[in_bounds]
        weights = weights[in_bounds]
        return sampled_points, weights

    def _sample_simpson(self,num_points):
        raise RuntimeError("simpson's rule not implemted for disc")    
        if len(num_points) != self.bounds.shape[0]:
            raise ValueError("using simpson rule expected num_points to have the same number of entries as dimensions (4,3,2)")

    def _sample_weights_from_unif(self,points):
        num_points = points.shape[0]
        volume = (torch.pi*(self.radius**2.0))
        constant_multi = volume/float(num_points)
        weights = torch.full((int(num_points),),fill_value=constant_multi)
        weights = weights.to(torch.get_default_dtype())
        return weights

    def _sample_points_from_unif(self,points):
        # Scale points to the disc
        num_points = points.shape[0]
        r_points = self.radius * torch.sqrt(points[:, 0])  # Use sqrt to ensure uniform distribution
        theta = 2 * torch.pi * points[:, 1]

        # Convert polar to Cartesian coordinates
        x = r_points * torch.cos(theta)
        y = r_points * torch.sin(theta)

        # Stack x and y to get the final points
        points = torch.stack((x, y), dim=1)        
        points = points.to(torch.get_default_dtype())
        return points


    def _sample_monte_carlo(self, num_points):
        num_points = check_2val(num_points)
        #points = torch.rand(num_points,2)
        rand_points = mersenne_twister.uniform(0,1,size=num_points*2)
        rand_points = torch.tensor(rand_points, dtype=torch.float32)
        rand_points = rand_points.reshape(num_points,2)
        
        out_points = self._sample_points_from_unif(rand_points)
        out_weights = self._sample_weights_from_unif(rand_points)
        return out_points,out_weights
    

    def _sample_sobol(self, num_points,is_pow2):
        num_points = check_2val(num_points)
        
        points = None
        num_points_log2 = np.log2(num_points)
        if round(num_points_log2,0) != num_points_log2:
            if is_pow2:
                raise RuntimeError("round(num_points_log2,0) != num_points_log2"+ f",num_points_log2=={num_points_log2}")
            sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True)
            points = sobol.draw(num_points,dtype=torch.float32)
        else:
            sampler = qmc.Sobol(d=2, scramble=True)
            points = sampler.random_base2(m=int(num_points_log2))
            points = torch.tensor(points)
        
        out_points = self._sample_points_from_unif(points)
        out_weights = self._sample_weights_from_unif(points)
        return out_points,out_weights
       
    def in_bounds(self,x):
        """Check if points are within the disc.
        
        Args:
            x (torch.Tensor): Points to check.
        
        Returns:
            torch.Tensor: Boolean tensor indicating if points are within the disc.
        """
        device = x.device
        dtype = x.dtype
        return torch.linalg.norm(x,dim=1)<self.radius
    
    def get_volume(self):
        """Calculate the volume of the disc.
        
        Returns:
            float: Volume of the disc.
        """
        
        return math.pi*self.radius**2.