# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "is_valid_square_circle",
    "OpticalSystem",
    "SequentialOpticalSystem",
    "OpticalElement",
    "OpticalSurface",
    "LensSurfaceTransmissionEnter",
    "LensSurfaceTransmissionLeave",
    "LensSurfaceSide",
    "Lens",
    "Mirror",
    "Detector",
    "trace_to_detector",
    "set_unused_params_to_zero",
    "get_unused_params_mask",
    "set_used_params_bounds_to_constant",
    "FresnelOpticalSurface",
    "FresnelVirtualLensSurfaceTransmissionEnter",
    "FresnelVirtualLensSurfaceTransmissionLeave",
    "FresnelVirtualLens",
    "compute_reflected_directions",
    "get_refracted_directions",
    "set_unused_bspline_coeff_to_nearest"
]

import torch
import torch.nn as nn
from .plotting import Plotable
from .refractive_index import materials
#from .refractive_index import RefractiveIndex
from .intersection import construct_surface_and_normal_func_with_params,get_ray_intersection_length
from .optimize import make_parameter_from_input
from . import transforms
from .integrators import Disc,Cube
import numpy as np
import cadquery as cq
from .physical_object import PhysicalObject,PhysicalSurface
from . utils.autograd import grad
from . optimize import minimize,remove_bounds,set_bounds_from_params_mask
from . transforms import Transform
"""
import numpy as np
color_pallete = np.array([(218.0, 232, 252),
    (108, 142, 191),
    (248, 206, 204),
    (184, 84, 80),
    (213, 232, 212),
    (130, 179, 102),
    (255, 230, 204),
    (215, 155, 0),
    (225, 213, 231),
    (150, 115, 166),
    (255, 242, 204),
    (214, 182, 86)])/255.0
import matplotlib.colors as mcolors
"""
    

def is_valid_square_circle(transform,O,aperture_radius,is_square):
    r"""
    Checks whether points lie within a circular or square aperture after transformation.

    Args:
        transform (Transform): Transformation object to convert global to local coordinates.
        O (torch.Tensor): Points in global coordinates of shape (N, 3).
        aperture_radius (float or torch.Tensor): Radius of the circular or square aperture.
        is_square (bool): If True, aperture is square; if False, circular.

    Returns:
        torch.Tensor: Boolean tensor of shape (N,) indicating whether each point lies within the aperture.

    Note:
        For a square, checks if \( |x| < r \) and \( |y| < r \). For a circle, checks if \( \sqrt{x^2 + y^2} < r \).
    """
    aperture_radius = torch.abs(torch.tensor(aperture_radius))
    with torch.no_grad():
        O_local = transform.to_local_pos(O.detach())
        if is_square:
            return ((torch.abs(O_local[:,0])<aperture_radius).float()*(torch.abs(O_local[:,1])<aperture_radius).float())==1.0
        else:
            return torch.norm(O_local,dim=1)<aperture_radius

class OpticalSystem(nn.Module,Plotable):
    """
    Base class for optical systems composed of multiple optical modules.

    This class serves as a container for modules such as lenses, mirrors, and detectors.
    It supports visualization and modular organization.

    Attributes:
        modules_dict (nn.ModuleDict): Dictionary of named optical modules.
    """
    def __init__(self, modules_dict):
        nn.Module.__init__(self)
        Plotable.__init__(self)
        self.modules_dict = nn.ModuleDict(modules_dict)    


    def forward(self):
        raise RuntimeError("OpticalSystem: forward is not implemented")
    
    def get_plotable_childs(self):
        out = [[self.modules_dict[key],key] for key in self.modules_dict.keys()]
        return out
        
    def get_plot_points2D(self,resolution):
        return []
    
    def get_plot_points3D(self,resolution):
        return []

        
class SequentialOpticalSystem(OpticalSystem):
    """
    Optical system that processes rays in a defined sequence.

    Useful for simulating light propagation through a sequence of elements, e.g., source → lens → detector.

    Attributes:
        n_func_enviroment (Callable): Function returning refractive index of the surrounding medium.
    """ 
    def __init__(self,modules_dict,n_func_enviroment=materials["AIR"]):
        OpticalSystem.__init__(self,modules_dict)
        self.n_func_enviroment = n_func_enviroment #Edit wavelength dependent
        

    def forward(self,x,mapping_sequence):
        """
        Propagates rays through the defined sequence of modules.

        Args:
            x (Any): Input rays or sampling data.
            mapping_sequence (list[str]): Ordered list of module names defining propagation sequence.

        Returns:
            Any: Output after final module in the sequence.
        """
        for name in mapping_sequence:
            from .source import RaySource
            if isinstance(self.modules_dict[name],RaySource):
                x = self.modules_dict[name](x,self.n_func_enviroment)
            else:
                x = self.modules_dict[name](*x)
        return x 
    

class OpticalElement(PhysicalObject,Plotable):
    """
    Abstract base class for optical elements like lenses, mirrors, and detectors.

    Provides interface for geometric transformation and ray propagation.
    """

    def __init__(self,fill_color="white", outline_color="black",is_volume=False):
        PhysicalObject.__init__(self)
        Plotable.__init__(self,fill_color=fill_color,outline_color=outline_color,is_volume=is_volume)
        
    def forward(self,O2, D2, wl, n_func_enviroment, meta_data):
        """
        Propagates rays through the optical element.

        Args:
            O2 (torch.Tensor): Ray origins.
            D2 (torch.Tensor): Ray directions.
            wl (torch.Tensor): Wavelengths.
            n_func_enviroment (Callable): Function returning environmental refractive index.
            meta_data (dict): Dictionary with path length and validity information.

        Raises:
            NotImplementedError: Must be overridden by subclasses.
        """
        raise NotImplementedError("process_ray not implemented")
    
    def get_transform(self):
        raise NotImplementedError("get_transform not implemented")


class OpticalSurface(OpticalElement,PhysicalSurface):
    """
    Represents a surface in 3D space with a defined aperture and transformation.

    Supports both square and circular apertures, and provides methods for parametric sampling,
    CAD conversion, ray intersection, and plotting.

    Attributes:
        surface (object): Object with a method `explicit(parametric_pos)` returning z-values.
        aperture_radius (float): Radius of the circular or square aperture.
        is_square (bool): Whether the aperture is square-shaped.
        transform (Transform): Local-to-global transformation.
        integrator (Integrator): Integration object (Disc or Cube) for parametric sampling.
    """
    def __init__(self,transform:Transform,surface,aperture_radius,is_square=False,fill_color="white", outline_color="black"):
        """
        Initializes the optical surface.

        Args:
            transform (Transform): Local-to-global transformation of the surface.
            surface (object): Surface object with an `explicit()` method for height computation.
            aperture_radius (float): Radius or half-width of the aperture.
            is_square (bool, optional): If True, aperture is square. Defaults to False.
            fill_color (str, optional): Color for plotting. Defaults to "white".
            outline_color (str, optional): Outline color. Defaults to "black".
        """
        OpticalElement.__init__(self,fill_color=fill_color,outline_color=outline_color)
        PhysicalSurface.__init__(self)
        self.surface = surface
        self.aperture_radius = aperture_radius
        self.is_square = is_square
        self.transform = transform
        
        integrator = None
        if is_square:
            #print("parametric integrator Cube")
            integrator = Cube([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius]])
        else:
            #print("parametric integrator Disc")
            integrator = Disc(aperture_radius)
        self.integrator = integrator
        
    def get_constraint_funs_leq_zero(self):
        """
        Returns constraint functions used for integration and optimization over the surface.

        Returns:
            list[Callable]: List of functions f(param_pos) <= 0 indicating valid parametric regions.

        Raises:
            RuntimeError: If `is_square` is True (not yet implemented).
        """
        aperture_radius = self.aperture_radius
        if self.is_square:
            #lambda parametric_pos: aperture_radius-torch.abs(parametric_pos)[...,0]
            #lambda parametric_pos: aperture_radius-parametric_pos[...,1]
            raise RuntimeError("get_constraint_funs_geq_zero is not implemented")
        else:

            def out_func(parametric_pos):
                val = torch.linalg.norm(parametric_pos,dim=-1)-aperture_radius
                return val
            return [out_func]
    
    """
    def get_corners_in_parameter_space(self):
        aperture_radius = self.aperture_radius
        if self.is_square:
            return [[-aperture_radius,aperture_radius],\
                                 [aperture_radius,-aperture_radius],\
                                 [-aperture_radius,-aperture_radius],\
                                 [aperture_radius,aperture_radius]]
        else:
            return []
        
    def get_edge_funcs_in_parameter_space(self):
        aperture_radius = self.aperture_radius
        if self.is_square:
            func1D = lambda t: -aperture_radius+t*aperture_radius*2.0
            out = [lambda t: (-aperture_radius,func1D(t)),\
                lambda t: (aperture_radius,func1D(t)),\
                lambda t: (func1D(t),-aperture_radius),\
                lambda t: (func1D(t),aperture_radius)]
            return out
        else:
            def parameterize_circle(t):
                theta = 2 * torch.pi * t  # Convert t to radians
                x = torch.cos(theta)*self.aperture_radius
                y = torch.sin(theta)*self.aperture_radius
                return (x,y)
            return [parameterize_circle]
    """
            


    def get_plot_points2D(self,resolution):
        y = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
        x = torch.zeros_like(y)
        O = torch.zeros((resolution,2))
        O[:,0] = x
        O[:,1] = y

        points = None
        with torch.no_grad():
            points = self.parametric_surface(O)
        
        y = points[:,1]
        z = points[:,2]
        return [(z,y)]

    def get_plot_points3D(self,resolution):
        _x = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
        _y = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
        
        if not self.is_square:
            mul = (torch.sqrt(x*x+y*y)>self.aperture_radius).float()/torch.sqrt(x*x+y*y)*self.aperture_radius
            mul += (torch.sqrt(x*x+y*y)<self.aperture_radius).float()
            x = x*mul
            y = y*mul
                
        O = torch.zeros((x.shape[0],2))
        O[:,0] = x
        O[:,1] = y
        
        
        points = None
        with torch.no_grad():
            points = self.parametric_surface(O)
        
        x = points[:,0].reshape(_x.shape[0],_x.shape[0])
        y = points[:,1].reshape(_x.shape[0],_x.shape[0])
        z = points[:,2].reshape(_x.shape[0],_x.shape[0])
        return [(x,y,z)]

    def get_CAD_points(self,resolution):
        """
        Generates a 3D surface point grid for CAD conversion.

        Args:
            resolution (int): Sampling resolution.

        Returns:
            tuple: (x, y, z) coordinate grids for CAD modeling.
        """
        #TODO maybe implement this also for affine transforms in surface class itself
        _x = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
        _y = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
                
        O = torch.zeros((x.shape[0],2))
        O[:,0] = x
        O[:,1] = y
        
        
        points = None
        with torch.no_grad():
            points = self.parametric_surface(O)
        
        x = points[:,0].reshape(_x.shape[0],_x.shape[0])
        y = points[:,1].reshape(_x.shape[0],_x.shape[0])
        z = points[:,2].reshape(_x.shape[0],_x.shape[0])
        return (x,y,z)

    def get_CAD_face(self,resolution,tol=0.001,smoothing = None,minDeg: int = 1,maxDeg: int = 3):
        """
        Converts the surface into a CAD face using B-spline approximation.

        Args:
            resolution (int): Sampling resolution.
            tol (float, optional): Approximation tolerance. Defaults to 0.001.
            smoothing (Optional[int]): Smoothing value for fitting.
            minDeg (int): Minimum degree of the spline.
            maxDeg (int): Maximum degree of the spline.

        Returns:
            cadquery.Face: CAD face object.
        """
        if hasattr(self.surface,"get_CAD_face"):
            affine_transform = self.transform.get_transformation_matrix()
            out = self.surface.get_CAD_face(affine_transform)
            if out is None:
                print("get_CAD_face returned NONE fallback to fit method")
            else:
                return out
            
        cat_func_points = lambda tmp: torch.cat([telem.reshape(*telem.shape,1) for telem in tmp],dim=-1)

        surface1_points = cat_func_points(self.get_CAD_points(resolution))
        surface1_points = [[cq.Vector(elem[0],elem[1],elem[2]) for elem in row ]for row in surface1_points]
        face1 = cq.Face.makeSplineApprox(surface1_points,smoothing=smoothing,minDeg=minDeg,maxDeg=maxDeg,tol=tol)
        return face1
    
    def parametric_sample(self,num_points,method="sobol")-> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples parametric positions on the aperture using the integrator.

        Args:
            num_points (int): Number of sample points.
            method (str): Sampling method. Options: "sobol", "monte_carlo", "midpoint", etc.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled positions and integration weights.
        """
        return self.integrator.sample(num_points,method)

    def parametric_surface(self,parametric_pos)->torch.Tensor:
        """
        Maps 2D parametric coordinates to 3D global coordinates using the surface height and transform.

        Args:
            parametric_pos (torch.Tensor): 2D parametric positions of shape (N, 2).

        Returns:
            torch.Tensor: 3D positions of shape (N, 3) in global space.

        Raises:
            RuntimeError: If input does not have shape [..., 2].
        """
        if parametric_pos.shape[-1] !=2:
            raise RuntimeError("positions must be in local coordinates [:,2]")
        device = parametric_pos.device
        dtype = parametric_pos.dtype
        
        z = self.surface.explicit(parametric_pos)
        x = parametric_pos[:,0]
        y = parametric_pos[:,1]

        v = torch.zeros((x.shape[0],4),device=device,dtype=dtype)
        v[:,0] = x
        v[:,1] = y
        v[:,2] = z
        v[:,3] = torch.ones_like(v[:,3],device=device,dtype=dtype)   
            
        Mv = None
        M = self.transform.get_transformation_matrix(device=device,dtype=dtype)
        if (M.dtype != dtype) or M.device != device:
            M = M.to(device=device,dtype=dtype)
        Mv = v@M.T
        out = Mv[:,[0,1,2]]
        return out
    
    """
    def explicit_surface(self,parametric_pos)->torch.Tensor:
        z = self.surface.explicit(parametric_pos)
        return z
    """
    def get_surface_and_normal_func_with_params(self):
        """
        Constructs a callable for surface position and normal computation with parameter tracking.

        Returns:
            tuple: (callable, list) where the callable computes (position, normal),
                and the list contains parameters to be optimized.
        """
        surface_and_normal,param_args = construct_surface_and_normal_func_with_params([self.transform,self.surface])
        return surface_and_normal,param_args
    
    def get_ray_intersect_length(self,O,D)->torch.Tensor:
        """
        Computes intersection length along ray until hitting the surface.

        Args:
            O (torch.Tensor): Ray origins of shape (N, 3).
            D (torch.Tensor): Ray directions of shape (N, 3).

        Returns:
            torch.Tensor: Intersection distances t such that O + t*D lies on the surface.
        """
        device = O.device
        dtype = O.dtype
        surface_and_normal,param_args = self.get_surface_and_normal_func_with_params()
        global_pos_approx = self.get_transform().to_global_pos(torch.zeros_like(O))
        t_init = torch.linalg.norm((global_pos_approx.detach()-O.detach()),dim=-1)

        t = get_ray_intersection_length(O,D,surface_and_normal,param_args,t_init)
        return t

    def get_new_is_valid(self,O,valid)->torch.Tensor:
        """
        Updates a boolean mask indicating which rays are still valid after hitting the aperture.

        Args:
            O (torch.Tensor): Ray intersection points.
            valid (torch.Tensor): Previous boolean validity mask.

        Returns:
            torch.Tensor: Updated validity mask.
        """
        valid = valid.float()*is_valid_square_circle(self.transform,O,self.aperture_radius,self.is_square).float()
        valid = valid==1.0
        return valid
    
    def get_transform(self)->transforms.Transform:
        """
        Returns the transformation associated with the surface.

        Returns:
            Transform: The local-to-global transformation object.
        """
        return self.transform
    
def get_refracted_directions(D, N, n1, n2):
    r"""
    Computes refracted ray directions using Snell's law.

    At material interfaces, the transmitted direction :math:`\mathbf{D'}` is computed based on the surface normal 
    :math:`\mathbf{N} = \nabla s / \|\nabla s\|` and the incident direction :math:`\mathbf{D}`, using Snell's law (see :cite:`do`):

    .. math::

        \mathbf{D'} = \mathbf{N} \sqrt{1 - (1 - \cos^2 \psi_i) \eta^2} + \eta (\mathbf{D} - \mathbf{N} \cos \psi_i),

    where :math:`\cos \psi_i = \mathbf{D} \cdot \mathbf{N}` and :math:`\eta = n / n'` is the ratio of the refractive indices 
    of the two materials.

    Args:
        D (torch.Tensor): Incident directions of shape (M, 3), normalized.
        N (torch.Tensor): Surface normals at points of incidence, shape (M, 3).
        n1 (float or torch.Tensor): Refractive index of the incident medium.
        n2 (float or torch.Tensor): Refractive index of the transmission medium.

    Returns:
        torch.Tensor: Refracted directions of shape (M, 3).

    """
    # Ensure the input tensors are normalized (unit vectors)
    D = torch.nn.functional.normalize(D, dim=1)
    N = torch.nn.functional.normalize(N, dim=1)
    
    minus_DTN = -torch.sum(D * N, dim=1, keepdim=False)
    
    constant_surface_dir = torch.sign(minus_DTN)#this is supposed to be positive
    minus_DTN = minus_DTN*constant_surface_dir#Saves every thing
    N = N*(constant_surface_dir.reshape(-1,1))

    cos_theta_1 = minus_DTN #positive
    
    n1_divi_n2 = (n1/n2)
    sin_theta_2_squared = (n1_divi_n2**2.0)*(1.-cos_theta_1**2.)
    cos_theta_2 = torch.sqrt(1.0-sin_theta_2_squared)
    #N has specific sign
    out = (n1_divi_n2.reshape(-1,1))*D+((n1_divi_n2*cos_theta_1-cos_theta_2).reshape(-1,1))*N
    return out

class LensSurfaceTransmissionEnter(OpticalSurface):
    def __init__(self,transform,surface,aperture_radius,n_func,is_square=False):
        super().__init__(transform,surface,aperture_radius,is_square,'#dae8fc',"#6c8ebf")
        self.n_func = n_func

    def forward(self, O1, D1, wl, n_func_enviroment,meta_data):
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        surface_and_normal_func1,param_args1 = self.get_surface_and_normal_func_with_params()
        t1 = self.get_ray_intersect_length(O1,D1)
        O2 = O1+t1*D1
        valid = self.get_new_is_valid(O2,valid)
        
        _,N2 = surface_and_normal_func1(O2,*param_args1)
         
        n_enviroment = n_func_enviroment(wl)
        n = self.n_func(wl)
        D2 = get_refracted_directions(D1, N2, n_enviroment, n)
        PL+=t1.reshape(-1)
        OPL+=t1.reshape(-1)*n_enviroment
        ray_paths+=[O2.detach()]
        
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

class LensSurfaceTransmissionLeave(OpticalSurface):
    def __init__(self,transform,surface,aperture_radius,n_func,is_square=False):
        super().__init__(transform,surface,aperture_radius,is_square,'#dae8fc',"#6c8ebf")
        self.n_func = n_func

    def forward(self, O2, D2, wl, n_func_enviroment, meta_data):
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        surface_and_normal_func2,param_args2 = self.get_surface_and_normal_func_with_params()
        
        t2 = self.get_ray_intersect_length(O2,D2)
        O3 = O2+t2*D2
        valid = self.get_new_is_valid(O3,valid)
        
        n_enviroment = n_func_enviroment(wl)
        n = self.n_func(wl)
        
        _,N3 = surface_and_normal_func2(O3,*param_args2)
        D3 = get_refracted_directions(D2, N3,n, n_enviroment)
        
        PL+=t2.reshape(-1)
        OPL+=t2.reshape(-1)*n
        ray_paths+=[O3.detach()]
        
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O3,D3,wl,n_func_enviroment,meta_data
    

class LensSurfaceSide(PhysicalSurface,Plotable):
    """
    Non-optical surface connecting two curved lens surfaces for visualization.

    Used to render the full 3D body of the lens.
    
    Attributes:
        surface1 (PhysicalSurface): First lens surface.
        surface2 (PhysicalSurface): Second lens surface.
        aperture_radius (float): Radius or half-width of aperture.
        is_square (bool): Whether aperture is square.
    """
    def __init__(self,surface1:PhysicalSurface,surface2:PhysicalSurface,aperture_radius,is_square:bool):
        Plotable.__init__(self,'#dae8fc','#dae8fc')
        PhysicalSurface.__init__(self)
        self.aperture_radius = aperture_radius
        self.surface1,self.surface2 = surface1,surface2
        self.is_square = is_square
        
        self.integrator = Cube([[0.0,1.0],[0.0,1.0]])

              
    def parametric_sample(self, num_points, method="sobol"):
        if (method != "sobol") or (method != "monte_carlo") or (method != "midpoint"):
            raise RuntimeError("Only sobol,monte_carlo and midpoint supported for LensSurfaceSide parametric_sample")
        
        return self.integrator.sample(num_points, method)

    def parametric_surface(self, parametric_pos):
        device = parametric_pos.device
        dtype = parametric_pos.dtype

        def parameterize_circle(t):
            theta = 2 * torch.pi * t  # Convert t to radians
            x = torch.cos(theta)*self.aperture_radius
            y = torch.sin(theta)*self.aperture_radius
            return (x,y)
        
        def parameterize_height(x,y,param_height):
            device = x.device
            dtype = x.dtype
            local_pos = torch.zeros((x.shape[0],2),device=device,dtype=dtype)
            local_pos[:,0]=x
            local_pos[:,1]=y
            
            pos3Dlow = self.surface1.parametric_surface(local_pos)
            pos3Dhigh = self.surface2.parametric_surface(local_pos)
            
            out = pos3Dlow+(pos3Dhigh-pos3Dlow)*param_height
            return out
        
        t = parametric_pos[:,0]
        param_height = parametric_pos[:,1]
        out = parameterize_height(*parameterize_circle(t),param_height)
        return out

    def get_plot_points2D(self,resolution):
        """
        Returns 2D slices through the surface (z-y plane) for plotting.

        Args:
            resolution (int): Number of sample points along the y-axis.

        Returns:
            list[tuple]: List of (z, y) coordinate tuples.
        """
        def aperture_pass(surface,transformation):
            y = torch.tensor([-self.aperture_radius,self.aperture_radius])
            x = torch.zeros_like(y)
            O = torch.zeros((2,2))
            O[:,0] = x
            O[:,1] = y
            Mv = None
            with torch.no_grad():
                Mv = surface.parametric_surface(O)
            y = Mv[:,1]
            z = Mv[:,2]

            return z,y
        _z1,_y1 = aperture_pass(self.surface1,self.surface1.transform)
        _z2,_y2 = aperture_pass(self.surface2,self.surface2.transform)
        z1 = torch.zeros((2))
        y1 = torch.zeros((2))
        z1[0] = _z1[0]
        z1[1] = _z2[0]
        
        y1[0] = _y1[0]
        y1[1] = _y2[0]
        
        z2 = torch.zeros((2))
        y2 = torch.zeros((2))
        z2[0] = _z1[1]
        z2[1] = _z2[1]
        
        y2[0] = _y1[1]
        y2[1] = _y2[1]
        return [(z1,y1),(z2,y2)]
    
    def get_plot_points3D(self,resolution):
        """
        Returns 3D grid of surface points for visualization.

        Args:
            resolution (int): Grid resolution in x and y.

        Returns:
            list[tuple]: List of (x, y, z) meshgrids as torch tensors.
        """
        def make_sub_surface(x,y):
            O = torch.zeros((y.shape[0],2))
            O[:,0] = x
            O[:,1] = y
            lower = self.surface1.parametric_surface(O)
            upper = self.surface2.parametric_surface(O)
            
            out_x = torch.zeros((x.shape[0],2))
            out_y = torch.zeros_like(out_x)
            out_z = torch.zeros_like(out_x)

            out_x[:,0] = lower[:,0]
            out_x[:,1] = upper[:,0]
            
            out_y[:,0] = lower[:,1]
            out_y[:,1] = upper[:,1]
            
            out_z[:,0] = lower[:,2]
            out_z[:,1] = upper[:,2]
            out_x,out_y,out_z = out_x.detach().cpu(),out_y.detach().cpu(),out_z.detach().cpu()
            return (out_x,out_y,out_z)
        def parameterize_circle(t):
            theta = 2 * torch.pi * t  # Convert t to radians
            x = torch.cos(theta)*self.aperture_radius
            y = torch.sin(theta)*self.aperture_radius
            return (x,y)
        with torch.no_grad():
            if self.is_square:
                y = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
                x = torch.ones_like(y)*(-self.aperture_radius)
                out = []
                out.append(make_sub_surface(x,y))
                out.append(make_sub_surface(y,x))
                
                y = torch.linspace(-self.aperture_radius,self.aperture_radius,resolution)
                x = torch.ones_like(y)*(self.aperture_radius)
                out.append(make_sub_surface(x,y))
                out.append(make_sub_surface(y,x))
                return out
            else:
                t = torch.linspace(0.,1.0,resolution*4)
                x,y = parameterize_circle(t)
                out = []
                out.append(make_sub_surface(x,y))
                return out    
    def get_plotly_color_scale(self):
        if self.is_square:
            out = []
            for k in range(4):
                out += Plotable.get_plotly_color_scale(self)
            return out
        else:
            return Plotable.get_plotly_color_scale(self)

class Lens(OpticalElement):
    r"""
    Represents a transmissive lens consisting of two refractive surfaces.

    The lens is modeled as a sequence of:
    - Entry surface (refraction from external medium into the lens)
    - Exit surface (refraction from lens into external medium)
    - Side surface (purely for visualization)

    In our implementation, lenses consist of two *explicit surfaces*, a transformation matrix :math:`M`, a lens thickness, 
    an *aperture radius*, and a material. When the lens is initialized, one can also optionally specify whether the lens 
    is round or square. If the keyword **is_square** is not specified, the lens will default to being round.

    Example:
        Below is an example of initializing a square lens:

        >>> import diffinytrace as dit
        >>> aperture_half = 30.
        >>> lens_thickness = 8.
        >>> material = dit.materials["NBK7"]
        >>> transform = dit.transforms.Identity()
        >>> bspline = dit.Bspline(aperture_half, [3, 3], [8, 8])
        >>> plane = dit.Plane()
        >>> lens = dit.Lens(transform, lens_thickness,
        >>>          bspline, plane,
        >>>          material, aperture_half, is_square=True)

    Attributes:
        n_func (Callable): Function mapping wavelength to refractive index of the lens material.
        _transform1 (Transform): Transform for the first surface.
        _transform2 (Transform): Transform for the second surface.
        lens_thickness (torch.nn.Parameter): Learnable thickness of the lens.
        surface1 (LensSurfaceTransmissionEnter): Entry surface.
        surface2 (LensSurfaceTransmissionLeave): Exit surface.
        lens_surface_side (LensSurfaceSide): Side surface (for 3D rendering).
        aperture_radius (float): Radius (or half-width) of aperture.
        is_square (bool): Whether the aperture is square.
    """
    def __init__(self,transform,lens_thickness,surface1,surface2,n_func,aperture_radius,is_square=False):
        OpticalElement.__init__(self,'#dae8fc',"#6c8ebf",True)
        
        self.n_func = n_func 
        self._transform1 = transform
        self.lens_thickness = make_parameter_from_input(lens_thickness)
        self._transform2 = transforms.Distance(self.lens_thickness,parent_transform=self._transform1)
        self._transform2.distance.bounds=torch.tensor([0.,torch.inf])

        self.surface1 = LensSurfaceTransmissionEnter(self._transform1,surface1,aperture_radius,n_func,is_square)    
        self.surface2 = LensSurfaceTransmissionLeave(self._transform2,surface2,aperture_radius,n_func,is_square)
        self.lens_surface_side = LensSurfaceSide(self.surface1,self.surface2,aperture_radius,is_square)

        self.aperture_radius = aperture_radius
        self.is_square = is_square

    def get_plot_points2D(self,resolution):
        def inverse_points(input):
            z,y = input
            z = torch.tensor(np.array(np.array(z)[::-1]))
            y = torch.tensor(np.array(np.array(y)[::-1]))
            return (z,y)    
        
        psurface1 = self.surface1.get_plot_points2D(resolution)
        psurface2 = self.surface2.get_plot_points2D(resolution)
        psurfaceCy = self.lens_surface_side.get_plot_points2D(resolution)
        
        #return psurface1+psurface2+psurfaceCy
    
        out = [None for k in range(4)]
        out[0] = psurface1[0]
        out[1] = inverse_points(psurfaceCy[1])
        out[2] = inverse_points(psurface2[0])
        out[3] = psurfaceCy[0]
        
        """
             out = []
        out += self.surface1.get_plot_points2D(resolution)
        out += self.surface2.get_plot_points2D(resolution)
        out += self.cylinder_surface.get_plot_points2D(resolution)
        
        return out
    
        """
        return out
    
    def get_plot_points3D(self,resolution):
        out = []
        out += self.surface1.get_plot_points3D(resolution)
        out += self.surface2.get_plot_points3D(resolution)
        out += self.lens_surface_side.get_plot_points3D(resolution)
        return out
    
    def get_plotly_color_scale(self):
        out = []
        out += self.surface1.get_plotly_color_scale()
        out += self.surface2.get_plotly_color_scale()
        out += self.lens_surface_side.get_plotly_color_scale()
        return out

    def get_plotable_childs(self):
        return []
    
    def forward(self,O1,D1,wl,n_func_enviroment,meta_data):
        """
        Simulates light passing through the lens.

        Args:
            O1 (torch.Tensor): Ray origin positions.
            D1 (torch.Tensor): Ray directions.
            wl (torch.Tensor): Wavelengths.
            n_func_enviroment (Callable): Function returning external medium refractive index.
            meta_data (dict): Ray metadata (PL, OPL, paths, valid).

        Returns:
            Tuple[torch.Tensor]: Updated ray origins, directions, etc.
        """
        out = self.surface1(O1,D1,wl,n_func_enviroment,meta_data)
        return self.surface2(*out)

    def get_transform(self):
        return self.surface2.transform

    
def compute_reflected_directions(D, N):
    r"""
    Computes reflected ray directions using the reflection law.

    Args:
        D (torch.Tensor): Incident directions of shape (M, 3), normalized.
        N (torch.Tensor): Surface normals at points of incidence, shape (M, 3).

    Returns:
        torch.Tensor: Reflected directions of shape (M, 3).

    """
    # Ensure the input tensors are normalized (unit vectors)
    D = torch.nn.functional.normalize(D, dim=1)
    N = torch.nn.functional.normalize(N, dim=1)
    cos_theta_1 = -torch.sum(D * N, dim=1, keepdim=True)
    
    # Compute the reflected direction
    out = D + 2 * cos_theta_1 * N  # Shape: (M, 3)

    return out

class Mirror(OpticalSurface):
    """
    Reflective optical element that reflects rays according to the law of reflection.

    Visualization is colored in a warm gold tone.

    Inherits:
        - OpticalSurface: Full support for surface transformation and intersection.

    """
    def __init__(self,transform,surface,aperture_radius,is_square=False):
        super().__init__(transform,surface,aperture_radius,is_square,'#fff2cc','#d6b656')
        
    def forward(self,O1,D1,wl,n_func_enviroment,meta_data):
        #DONE: test mirror for 180° rotation.
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        surface_and_normal_func1,param_args1 = self.get_surface_and_normal_func_with_params()
        
        t1 = self.get_ray_intersect_length(O1,D1)
        O2 = O1+t1*D1
        valid = self.get_new_is_valid(O2,valid)

        _,N2 = surface_and_normal_func1(O2,*param_args1)
        
        D2 = compute_reflected_directions(D1, N2)

        n_enviroment = n_func_enviroment(wl)
        PL+=t1.reshape(-1)
        OPL+=t1.reshape(-1)*n_enviroment
        ray_paths += [O2.detach()]
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

   
class Detector(OpticalSurface):
    r"""
    Represents a terminal optical element that collects ray data.

    Detectors consist of an *explicit surface*, a transformation matrix :math:`M`, and an *aperture radius*. 
    The detector class represents a target surface used to track the rays that hit it. When the detector is initialized, 
    one can also optionally specify whether the detector is round or square. If the keyword **is_square** is not specified, 
    the detector defaults to being square.

    Example:
        Below is an example of how to initialize a detector:

        >>> import diffinytrace as dit
        >>> aperture_half = 30.
        >>> transform = dit.transforms.Identity()
        >>> plane = dit.Plane()
        >>> detector = dit.Detector(transform, plane,
        >>>                         aperture_half, is_square=False)

    """
    def __init__(self,transform,surface,aperture_radius,is_square=True):
        super().__init__(transform,surface,aperture_radius,is_square,'#d5e8d4','#82b366')
    
    def forward(self,O1,D1,wl,n_func_enviroment,meta_data):
        r"""
        Captures the final ray interaction without altering its direction.

        Args:
            O1 (torch.Tensor): Ray origin.
            D1 (torch.Tensor): Ray direction.
            wl (torch.Tensor): Wavelength.
            n_func_enviroment (Callable): Function for surrounding medium.
            meta_data (dict): Ray tracing metadata.

        Returns:
            Tuple[torch.Tensor]: Final ray data.
        """
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        t1 = self.get_ray_intersect_length(O1,D1)
        O2 = O1+t1*D1
        valid = self.get_new_is_valid(O2,valid)
        D2 = D1

        
        n_enviroment = n_func_enviroment(wl)
        PL+=t1.reshape(-1)
        OPL+=t1.reshape(-1)*n_enviroment
        
        ray_paths+=[O2.detach()]
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

    

def trace_to_detector(optical_system:SequentialOpticalSystem,sequence,source,detector:Detector,num_rays=200000,device=torch.get_default_device(),method_ray_tracing="sobol"):
    r"""
    Traces rays through a system to a detector and returns the impact coordinates.

    Args:
        optical_system (SequentialOpticalSystem): Ray-tracing pipeline.
        sequence (list[str]): Ordered names of system modules.
        source: Source object with `.sample()` method.
        detector (Detector): Final surface to collect rays.
        num_rays (int): Number of rays to simulate.
        device: Torch device (CPU/GPU).
        method_ray_tracing (str): Sampling method for source rays.

    Returns:
        Tuple[torch.Tensor]: (input samples, weights, detector plane hits, wavelengths)
    """
    def g_mapping(x):
        O,D,wl,_,_ = optical_system(x,sequence)
        O_local = detector.to_local_pos(O)
        return O_local[:,[0,1]],O,wl
    x,weights = source.sample(num_rays,method_ray_tracing)
    x = x.to(device)
    weights = weights.to(device)
    y,O,wl = g_mapping(x)
    return x,weights,y,wl



def set_unused_params_to_zero(optical_system:SequentialOpticalSystem,sequence,source,params,num_rays=200000,method_ray_tracing="sobol"):
    """
    Sets unused parameters (those with zero gradient across ray paths) to zero.

    Args:
        optical_system (SequentialOpticalSystem): Full system.
        sequence (list): Ordered module names.
        source: Ray source.
        params (list[torch.nn.Parameter] or torch.nn.Parameter): Parameters to clean.
        num_rays (int): Ray sample count.
        method_ray_tracing (str): Sampling method.
    """    
    if isinstance(params,nn.Parameter):
        params = [params]
    params = [param for param in params]
    device = params[0].device
    dtype = params[0].dtype
    
    x,weights = source.sample(num_rays,method_ray_tracing)

    x = x.to(device=device,dtype=dtype)
    weights = weights.to(device=device,dtype=dtype)
    O,D,wl,_,_ = optical_system(x,sequence)
    dOdp = grad(O,params,torch.randn_like(O),create_graph=False,materialize_grads=True)
    dDdp = grad(D,params,torch.randn_like(D),create_graph=False,materialize_grads=True)
    dwldp = grad(wl,params,torch.randn_like(wl),create_graph=False,materialize_grads=True)
    
    for k in range(len(params)):
        with torch.no_grad():
            dp_zero = (dOdp[k]==0.0).float()*(dDdp[k]==0.0).float()*(dwldp[k]==0.0).float()
            dp_zero = dp_zero==1.0
            param = params[k]
            param.data[dp_zero] = 0.0


def get_unused_params_mask(optical_system:SequentialOpticalSystem,sequence,source,params,num_rays=100000,method_ray_tracing="sobol"):
    """
    Returns a boolean mask identifying which parameters are unused in the ray tracing process.

    Args:
        optical_system (SequentialOpticalSystem): Full system.
        sequence (list): Ordered module names.
        source: Ray source.
        params (list[torch.nn.Parameter]): Parameter list.
        num_rays (int): Number of rays to test.
        method_ray_tracing (str): Sampling method.

    Returns:
        list[torch.BoolTensor]: Masks of the same shape as each parameter.
    """
    if isinstance(params,nn.Parameter):
        params = [params]
    params = [param for param in params]
    device= params[0].device
    x,weights = source.sample(num_rays,method_ray_tracing)

    x = x.to(device)
    weights = weights.to(device)
    O,D,wl,_,_ = optical_system(x,sequence)
    dOdp = grad(O,params,torch.randn_like(O),create_graph=False,materialize_grads=True)
    dDdp = grad(D,params,torch.randn_like(D),create_graph=False,materialize_grads=True)
    dwldp = grad(wl,params,torch.randn_like(wl),create_graph=False,materialize_grads=True)
    
    #print("len(params)",len(params))
    out = []
    for k in range(len(params)):
        with torch.no_grad():
            dp_zero = (dOdp[k]==0.0).float()*(dDdp[k]==0.0).float()*(dwldp[k]==0.0).float()
            dp_zero = dp_zero==1.0
            #print(dp_zero)
            out+=[dp_zero]
    return out


def set_used_params_bounds_to_constant(optical_system,sequence,source,params,bounds_attr_name_new,bounds_attr_name_old="bounds",num_rays=100000,method_ray_tracing="sobol"):
    """
    Locks unused parameters by copying their current value as bounds, making them constant.

    Args:
        bounds_attr_name_new (str): Name of the new bounds attribute to write.
        bounds_attr_name_old (str): Name of the original bounds attribute.
    """
    mask = get_unused_params_mask(optical_system,sequence,source,params,num_rays,method_ray_tracing)
    set_bounds_from_params_mask(params,mask,bounds_attr_name_new,bounds_attr_name_old)



class FresnelOpticalSurface(OpticalSurface):
    def __init__(self,transform,surface,aperture_radius,surface_derivative_x,surface_derivative_y,is_square=False):
        super().__init__(transform,surface,aperture_radius,is_square,'#dae8fc',"#6c8ebf")
        self.surface_derivative_x = surface_derivative_x
        self.surface_derivative_y = surface_derivative_y
    
    def get_virtual_normals(self,O):
        surface_and_normal1,param_args1 = construct_surface_and_normal_func_with_params([self.transform,self.surface_derivative_x])
        surface_and_normal2,param_args2 = construct_surface_and_normal_func_with_params([self.transform,self.surface_derivative_y])
        dx,_ = surface_and_normal1(O,*param_args1)
        dy,_ = surface_and_normal2(O,*param_args2)
        dx = dx.reshape(-1,1)
        dy = dy.reshape(-1,1)
        dz = torch.ones_like(dx)
        out = torch.cat([dx,dy,dz],dim=1)
        return out
        

class FresnelVirtualLensSurfaceTransmissionEnter(FresnelOpticalSurface):
    def __init__(self,transform,surface,aperture_radius,n_func,surface_derivative_x,surface_derivative_y,is_square=False):
        super().__init__(transform,surface,aperture_radius,surface_derivative_x,surface_derivative_y,is_square)
        self.n_func = n_func
        
    def forward(self, O1, D1, wl, n_func_enviroment,meta_data):
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        t1 = self.get_ray_intersect_length(O1,D1)
        O2 = O1+t1*D1
        valid = self.get_new_is_valid(O2,valid)
        
        N2 = self.get_virtual_normals(O2)
         
        n_enviroment = n_func_enviroment(wl)
        n = self.n_func(wl)
        D2 = get_refracted_directions(D1, N2, n_enviroment, n)
        PL+=t1.reshape(-1)
        OPL+=t1.reshape(-1)*n_enviroment
        ray_paths+=[O2.detach()]
        
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O2,D2,wl,n_func_enviroment,meta_data

class FresnelVirtualLensSurfaceTransmissionLeave(FresnelOpticalSurface):
    def __init__(self,transform,surface,aperture_radius,n_func,surface_derivative_x,surface_derivative_y,is_square=False):
        super().__init__(transform,surface,aperture_radius,surface_derivative_x,surface_derivative_y,is_square)
        self.n_func = n_func
        
    def forward(self, O2, D2, wl, n_func_enviroment, meta_data):
        PL, OPL, ray_paths, valid = meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"],meta_data["valid"]

        
        t2 = self.get_ray_intersect_length(O2,D2)
        O3 = O2+t2*D2
        valid = self.get_new_is_valid(O3,valid)
        
        n_enviroment = n_func_enviroment(wl)
        n = self.n_func(wl)
        
        N3 = self.get_virtual_normals(O3)
        D3 = get_refracted_directions(D2, N3,n, n_enviroment)
        
        PL+=t2.reshape(-1)
        OPL+=t2.reshape(-1)*n
        ray_paths+=[O3.detach()]
        
        meta_data["PL"],meta_data["OPL"],meta_data["ray_paths"], meta_data["valid"] = PL, OPL, ray_paths, valid
        return O3,D3,wl,n_func_enviroment,meta_data

class FresnelVirtualLens(OpticalElement):
    """
    """
    def __init__(self,transform,lens_thickness,surface1,surface2,n_func,aperture_radius,surface1_derivative_x=None,surface1_derivative_y=None,surface2_derivative_x=None,surface2_derivative_y=None,is_square=False):
        OpticalElement.__init__(self,'#dae8fc',"#6c8ebf",True)
        
        self.n_func = n_func 
        self._transform1 = transform
        self.lens_thickness = make_parameter_from_input(lens_thickness)
        self._transform2 = transforms.Distance(self.lens_thickness,parent_transform=self._transform1)
        self._transform2.distance.bounds=torch.tensor([0.,torch.inf])

        if (not surface1_derivative_x is None) or (not surface1_derivative_y is None):
            if (surface1_derivative_x is None or surface1_derivative_y is None):
                raise RuntimeError("if surface1_derivative_x is defined also surface1_derivative_y must be defined and the other way around!")
            self.surface1 = FresnelVirtualLensSurfaceTransmissionEnter(self._transform2,surface2,aperture_radius,n_func,surface1_derivative_x,surface1_derivative_y,is_square)
        else:
            self.surface1 = LensSurfaceTransmissionEnter(self._transform1,surface1,aperture_radius,n_func,is_square)    
        
        if (not surface2_derivative_x is None) or (not surface2_derivative_y is None):
            if (surface2_derivative_x is None or surface2_derivative_y is None):
                raise RuntimeError("if surface2_derivative_x is defined also surface2_derivative_y must be defined and the other way around!")
            self.surface2 = FresnelVirtualLensSurfaceTransmissionLeave(self._transform2,surface2,aperture_radius,n_func,surface2_derivative_x,surface2_derivative_y,is_square)
        
        else:
            self.surface2 = LensSurfaceTransmissionLeave(self._transform2,surface2,aperture_radius,n_func,is_square)
        self.lens_surface_side = LensSurfaceSide(self.surface1,self.surface2,aperture_radius,is_square)

        self.aperture_radius = aperture_radius
        self.is_square = is_square

    def get_plot_points2D(self,resolution):
        def inverse_points(input):
            z,y = input
            z = torch.tensor(np.array(np.array(z)[::-1]))
            y = torch.tensor(np.array(np.array(y)[::-1]))
            return (z,y)    
        
        psurface1 = self.surface1.get_plot_points2D(resolution)
        psurface2 = self.surface2.get_plot_points2D(resolution)
        psurfaceCy = self.lens_surface_side.get_plot_points2D(resolution)
        
        #return psurface1+psurface2+psurfaceCy
    
        out = [None for k in range(4)]
        out[0] = psurface1[0]
        out[1] = inverse_points(psurfaceCy[1])
        out[2] = inverse_points(psurface2[0])
        out[3] = psurfaceCy[0]
        
        """
             out = []
        out += self.surface1.get_plot_points2D(resolution)
        out += self.surface2.get_plot_points2D(resolution)
        out += self.cylinder_surface.get_plot_points2D(resolution)
        
        return out
    
        """
        return out
    
    def get_plot_points3D(self,resolution):
        out = []
        out += self.surface1.get_plot_points3D(resolution)
        out += self.surface2.get_plot_points3D(resolution)
        out += self.lens_surface_side.get_plot_points3D(resolution)
        return out
    
    def get_plotly_color_scale(self):
        out = []
        out += self.surface1.get_plotly_color_scale()
        out += self.surface2.get_plotly_color_scale()
        out += self.lens_surface_side.get_plotly_color_scale()
        return out

    def get_plotable_childs(self):
        return []
    
    def forward(self,O1,D1,wl,n_func_enviroment,meta_data):
        out = self.surface1(O1,D1,wl,n_func_enviroment,meta_data)
        return self.surface2(*out)

    def get_transform(self):
        return self.surface2.transform





"""
def smooth_optical_surface_with_unused_params(optical_surface:OpticalSurface,\
                                              optical_system:OpticalSystem,\
                                            sequence,\
                                            source,\
                                            params,\
                                            bounds_attr_name_old="bounds",\
                                            num_rays=100000,\
                                            method_ray_tracing="sobol",\
                                            num_points_surface=[701,701],\
                                            method_surface="simpson",\
                                            constraints=[],\
                                            minimization_method=None,\
                                            tol=1e-9):
    if isinstance(params,nn.Parameter):
        params = [params]
    
    params = [param for param in params]
    
    device = params[0].device
    dtype = params[0].dtype
    bounds_attr_name_new = "__used_bounds"
    set_used_params_bounds_to_constant(optical_system,sequence,source,params,bounds_attr_name_new,bounds_attr_name_old,num_rays,method_ray_tracing)
    def smoothness_func1():
        parametric_pos,weights = optical_surface.parametric_sample(num_points_surface,method_surface)
        weights = weights.to(device=device,dtype=dtype)
        parametric_pos = parametric_pos.detach()
        parametric_pos = parametric_pos.to(device=device,dtype=dtype)
        
        parametric_pos.requires_grad = True
        tmp = optical_surface.parametric_surface(parametric_pos)
        tmp = optical_surface.to_local_pos(tmp)
        dzdx, = grad(tmp[:,2],parametric_pos,torch.ones_like(tmp[:,2]),create_graph=True,retain_graph=True)
        
        ddzdx1dx, = grad(dzdx[:,0],parametric_pos,torch.ones_like(dzdx[:,0]),create_graph=True,retain_graph=True)
        ddzdx2dx, = grad(dzdx[:,1],parametric_pos,torch.ones_like(dzdx[:,1]),create_graph=True,retain_graph=True)
        
        

        smoothness = torch.sum((torch.abs(ddzdx1dx)+torch.abs(ddzdx2dx))*weights.reshape(-1,1))
        return smoothness

    def smoothness_func2():
        parametric_pos,weights = optical_surface.parametric_sample(num_points_surface,method_surface)
        weights = weights.to(device=device,dtype=dtype)
        parametric_pos = parametric_pos.detach()
        parametric_pos = parametric_pos.to(device=device,dtype=dtype)
        
        parametric_pos.requires_grad = True
        tmp = optical_surface.parametric_surface(parametric_pos)
        tmp = optical_surface.to_local_pos(tmp)
        dzdpos, = grad(tmp[:,2],parametric_pos,torch.ones_like(tmp[:,2]),create_graph=True,retain_graph=True)
        
        smoothness = torch.sum(torch.abs(dzdpos)*weights.reshape(-1,1))
        return smoothness


    out = minimize(smoothness_func1, params, constraints, minimization_method, tol=tol,bounds_attr_name="__used_bounds")
    out = minimize(smoothness_func2, params, constraints, minimization_method, tol=tol,bounds_attr_name="__used_bounds")
    remove_bounds(params,bounds_attr_name_new)
    return out

def smooth_lens_with_unused_params(lens:Lens,\
                                              optical_system:OpticalSystem,\
                                            sequence,\
                                            source,\
                                            params,\
                                            bounds_attr_name_old="bounds",\
                                            num_rays=100000,\
                                            method_ray_tracing="sobol",\
                                            num_points_surface=[701,701],\
                                            method_surface="simpson",\
                                            constraints=[],\
                                            minimization_method=None,\
                                            tol=1e-9):
    def run_on_surface(optical_surface):
        smooth_optical_surface_with_unused_params(optical_surface,\
                                                optical_system,\
                                                sequence,\
                                                source,\
                                                params,\
                                                bounds_attr_name_old,\
                                                num_rays,\
                                                method_ray_tracing,\
                                                num_points_surface,\
                                                method_surface,\
                                                constraints,\
                                                minimization_method,\
                                                tol)
    run_on_surface(lens.surface1)
    run_on_surface(lens.surface2)
    


"""
def set_unused_bspline_coeff_to_nearest(optical_system,\
                                            sequence,\
                                            source,\
                                            bspline_surface,\
                                            num_rays=100000,\
                                            method_ray_tracing="sobol"):
    
    """
    Fills only the unused B-spline coefficients with the nearest used value.

    This function identifies B-spline coefficients that have no influence on the ray paths
    (i.e., gradients are zero), and updates only those by copying the value from the closest
    neighboring coefficient that is used. Used coefficients remain unchanged.

    This is useful for having geometry that is simple to manifacture while not tempering with the overall performance.

    Args:
        optical_system (SequentialOpticalSystem): The optical system used for tracing.
        sequence (list[str]): Ordered list of module names for ray propagation.
        source: Ray source with a `.sample()` method.
        bspline_surface: Surface object with a `.coeff` tensor.
        num_rays (int, optional): Number of rays used to detect unused coefficients. Default is 100000.
        method_ray_tracing (str, optional): Sampling method (e.g., "sobol"). Default is "sobol".

    Raises:
        RuntimeError: If all coefficients are unused — likely due to insufficient ray coverage.
    """
    coeff = bspline_surface.coeff
    params = [coeff]

    mask = get_unused_params_mask(optical_system,sequence,source,params,num_rays=num_rays,method_ray_tracing=method_ray_tracing)
    mask = mask[0]
    mask = mask.reshape(*coeff.shape)
    
    dist = mask.float()
    shape = dist.shape
    dist = dist.reshape(-1)
    dist[dist==1.0] = torch.inf
    dist = dist.reshape(*shape)
    def valid_indices(yi, xi, mask):
        # Check if indices are within bounds
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
            return True
        return False
    if (dist==torch.inf).all():
        raise RuntimeError("all coeffs seem to be unused maybe try more rays?")
    with torch.no_grad():#<----------------------- this was changed
        while (dist==torch.inf).any():
            print("number of unset coefficients: ",torch.sum((dist==torch.inf).float()))
            for yi in range(mask.shape[0]):
                for xi in range(mask.shape[1]):
                    if mask[yi, xi]:  # Only operate if the mask is true
                        # Check each neighbor and update if valid
                        min_dist = torch.inf
                        min_dist_data = None
                        
                        if valid_indices(yi + 1, xi + 1, dist) and dist[yi + 1, xi + 1]!=torch.inf:
                            if min_dist>dist[yi + 1, xi + 1]:
                                min_dist_data = coeff.data[yi + 1, xi + 1]
                                min_dist = dist[yi + 1, xi + 1]
                            
                        if valid_indices(yi - 1, xi + 1, dist) and dist[yi - 1, xi + 1]!=torch.inf:
                            if min_dist>dist[yi - 1, xi + 1]:
                                min_dist_data = coeff.data[yi - 1, xi + 1]
                                min_dist = dist[yi - 1, xi + 1]
                                
                        if valid_indices(yi - 1, xi - 1, dist) and dist[yi - 1, xi - 1]!=torch.inf:
                            if min_dist>dist[yi - 1, xi - 1]:
                                min_dist_data = coeff.data[yi - 1, xi - 1]
                                min_dist = dist[yi - 1, xi - 1]
                            
                        if valid_indices(yi + 1, xi - 1, dist) and dist[yi + 1, xi - 1]!=torch.inf:
                            if min_dist>dist[yi + 1, xi - 1]:
                                min_dist_data = coeff.data[yi + 1, xi - 1]
                                min_dist = dist[yi + 1, xi - 1]
                        
                        if min_dist != torch.inf:
                            coeff.data[yi, xi] = min_dist_data
                            dist[yi, xi] = min_dist+1


