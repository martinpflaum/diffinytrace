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
from .plotting import Plotable
from .refractive_index import materials
#from .refractive_index import RefractiveIndex
from .intersection import construct_surface_and_normal_func_with_params,get_ray_intersection_length
from .optimize import make_parameter_from_input
from . import transforms
from . import export
from .integrators import Disc,Cube
from .surface import Bspline
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
    aperture_radius = torch.abs(torch.tensor(aperture_radius))
    with torch.no_grad():
        O_local = transform.to_local_pos(O.detach())
        if is_square:
            return ((torch.abs(O_local[:,0])<aperture_radius).float()*(torch.abs(O_local[:,1])<aperture_radius).float())==1.0
        else:
            return torch.norm(O_local,dim=1)<aperture_radius

class OpticalSystem(nn.Module,Plotable):
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
    This class is just used to register all lenses such that we get all parameters easily.
    TODO move mapping sequence to optical element??
    """ 
    def __init__(self,modules_dict,n_func_enviroment=materials["AIR"]):
        OpticalSystem.__init__(self,modules_dict)
        self.n_func_enviroment = n_func_enviroment #Edit wavelength dependent
        

    def forward(self,x,mapping_sequence):
        for name in mapping_sequence:
            from .source import RaySource
            if isinstance(self.modules_dict[name],RaySource):
                x = self.modules_dict[name](x,self.n_func_enviroment)
            else:
                x = self.modules_dict[name](*x)
        return x 
    

class OpticalElement(PhysicalObject,Plotable):
    def __init__(self,fill_color="white", outline_color="black",is_volume=False):
        PhysicalObject.__init__(self)
        Plotable.__init__(self,fill_color=fill_color,outline_color=outline_color,is_volume=is_volume)
        
    def forward(self,O2, D2, wl, n_func_enviroment, meta_data):
        raise NotImplementedError("process_ray not implemented")
    
    def get_transform(self):
        raise NotImplementedError("get_transform not implemented")


class OpticalSurface(OpticalElement,PhysicalSurface):
    def __init__(self,transform:Transform,surface,aperture_radius,is_square=False,fill_color="white", outline_color="black"):
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
        return self.integrator.sample(num_points,method)

    def parametric_surface(self,parametric_pos)->torch.Tensor:
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
        surface_and_normal,param_args = construct_surface_and_normal_func_with_params([self.transform,self.surface])
        return surface_and_normal,param_args
    
    def get_ray_intersect_length(self,O,D)->torch.Tensor:
        device = O.device
        dtype = O.dtype
        surface_and_normal,param_args = self.get_surface_and_normal_func_with_params()
        global_pos_approx = self.get_transform().to_global_pos(torch.zeros_like(O))
        t_init = torch.linalg.norm((global_pos_approx.detach()-O.detach()),dim=-1)

        t = get_ray_intersection_length(O,D,surface_and_normal,param_args,t_init)
        return t

    def get_new_is_valid(self,O,valid)->torch.Tensor:
        valid = valid.float()*is_valid_square_circle(self.transform,O,self.aperture_radius,self.is_square).float()
        valid = valid==1.0
        return valid
    
    def get_transform(self)->transforms.Transform:
        return self.transform
    
def get_refracted_directions(D, N, n1, n2):
    """
    Computes the refracted directions for multiple rays

    Parameters:
    - D: Tensor of shape (M, 3) representing M incident direction vectors (before refraction)
    - N: Tensor of shape (M, 3) representing M normal vectors at the points of refraction
    - n1: Refractive index of the initial medium (scalar)
    - n2: Refractive index of the second medium (scalar)

    Returns:
    - D_prime: Tensor of shape (M, 3) representing the refracted direction vectors
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
    """
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
        out = self.surface1(O1,D1,wl,n_func_enviroment,meta_data)
        return self.surface2(*out)

    def get_transform(self):
        return self.surface2.transform

    
def compute_reflected_directions(D, N):
    """
    Computes the reflected directions for multiple rays using PyTorch.

    Parameters:
    - D: Tensor of shape (M, 3) representing M incident direction vectors (before reflection)
    - N: Tensor of shape (M, 3) representing M normal vectors at the points of reflection

    Returns:
    - D_prime: Tensor of shape (M, 3) representing the reflected direction vectors
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
    tracker of rays
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
    """
    tracker of rays
    """
    def __init__(self,transform,surface,aperture_radius,is_square=True):
        super().__init__(transform,surface,aperture_radius,is_square,'#d5e8d4','#82b366')
    
    def forward(self,O1,D1,wl,n_func_enviroment,meta_data):
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


