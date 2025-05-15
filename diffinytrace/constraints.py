# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


from .physical_object import PhysicalSurface
import torch
from . integrators import Cube
from .optimize import minimize
from .utils.autograd import grad
#from .element import OpticalSurface
class Constraint():
    """
    Base class for optimization constraints.

    Attributes:
        fun (Callable): Function defining the constraint.
        type (str): Type of constraint ('eq' or 'ineq').
    """
    def __init__(self,fun,type):
        self.fun = fun
        self.type = type
    
class EqualZero(Constraint):
    """
    Equality constraint enforcing `fun() == 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(fun,'eq')

class GEQZero(Constraint):
    """
    Inequality constraint enforcing `fun() >= 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(fun,'ineq')

class LEQZero(Constraint):
    """
    Inequality constraint enforcing `fun() <= 0`.

    Args:
        fun (Callable): The constraint function.
    """
    def __init__(self,fun):
        super().__init__(lambda: -fun(),'ineq')

"""
class SurfaceDistance(Constraint):
    def __init__(self, fun, type):
        super().__init__(fun, type)
"""


"""class CustomAutogradRule_KKT_active_g(torch.autograd.Function):
    @staticmethod
    def forward(ctx,sol_kwargs,active_gs_tol,*params):
        f = sol_kwargs["f"]
        x = sol_kwargs["x"]
        gs = sol_kwargs["gs"]
        if len(params)==0:
            raise ValueError("len(params)==0")
        ctx.save_for_backward(x,*params)
        ctx.gs = gs
        ctx.f = f
        ctx.active_gs_tol = active_gs_tol

        out = f(x)
        return out
    
    @staticmethod
    def backward(ctx, grad_outputs):
        from . optimize import unpack_tensors,pack_tensors
        x = ctx.saved_tensors[0]
        params = ctx.saved_tensors[1:]
        active_gs_tol = ctx.active_gs_tol
        if torch.is_tensor(params):
            params = [params]
        params = [param for param in params]
        device = x.device
        dtype = x.dtype
          
        f = ctx.f
        gs = ctx.gs
        x = x.detach()
        #x = x.clone()
        with torch.enable_grad():
            x.requires_grad = True
            
            fval = f(x)
            gvals = [g(x)for g in gs]
            print("g1< active_gs_tol",torch.abs(gvals[0]).detach() < active_gs_tol)
            print("g2< active_gs_tol",torch.abs(gvals[1]).detach() < active_gs_tol)
            gvals = [gval for gval in gvals if torch.abs(gval).detach() < active_gs_tol]
            num_constraints = len(gvals)
            M = None
            N_no_derivative = None
            fx = None
            if num_constraints==0:
                fx = grad(fval,inputs=[x],create_graph=True)[0]
                Lx = fx
                Lxx = torch.zeros((Lx.shape[0],Lx.shape[0]),device=device,dtype=dtype)
                for k in range(Lx.shape[0]):
                    Lxx[:,k] = grad(Lx[k],inputs=[x],retain_graph=True,create_graph=False)[0]
                M = Lxx
                N_no_derivative = torch.zeros((Lxx.shape[0]),device=device,dtype=dtype)
                N_no_derivative[:Lx.shape[0]] = Lx
                
            else:
                fx = grad(fval,inputs=[x],create_graph=True)[0]
                gx = [grad(gval,inputs=[x],create_graph=True)[0] for gval in gvals]
                
                
                G = torch.stack(gx).T  # Shape: [*x.shape,num_constraints]
                lstsq_result = torch.linalg.lstsq(G.detach(), -fx.detach())
                mu = lstsq_result.solution

                Lx = torch.sum((mu.reshape(1,-1)*G),dim=-1)+fx
                Lxx = torch.zeros((Lx.shape[0],Lx.shape[0]),device=device,dtype=dtype)
                for k in range(Lx.shape[0]):
                    Lxx[:,k] = grad(Lx[k],inputs=[x],retain_graph=True,create_graph=False)[0]
                
                Gmu = (mu.reshape(1,-1)*G).T
                M = torch.zeros((num_constraints+Lxx.shape[0],num_constraints+Lxx.shape[0]),device=device,dtype=dtype)
                M[:Lxx.shape[0],:Lxx.shape[0]] = Lxx
                M[Lxx.shape[0]:,:Lxx.shape[0]] = Gmu
                M[:Lxx.shape[0],Lxx.shape[0]:] = G
                

            N_no_derivative = torch.zeros((num_constraints+Lxx.shape[0]),device=device,dtype=dtype)
            N_no_derivative[:Lx.shape[0]] = Lx
            for k in range(len(gvals)):
                N_no_derivative[k] = gvals[k]
            num_params = torch.sum(torch.tensor([torch.prod(torch.tensor(param.shape)).item() for param in params]))
            num_params = int(num_params.detach().cpu().item())
            Nshape = (num_params,num_constraints+Lxx.shape[0])
            N = torch.zeros(Nshape,device=device,dtype=dtype)
            for k in range(N.shape[1]):
                #print(N_no_derivative[k])
                tmp_grad = grad(N_no_derivative[k],inputs=params,create_graph=False,retain_graph=True,materialize_grads=True)
                #print("tmp_grad",tmp_grad)
                N[:,k] = pack_tensors(tmp_grad)
            M = M.detach()
            def is_singular(matrix, tol=1e-10):
                det = torch.linalg.det(matrix)
                return torch.abs(det) < tol
            
            if is_singular(M):
                dfdp = grad(fval,inputs=params,create_graph=False)
                dfdp = [elem for elem in dfdp]
                print("Matrix M is singular are these two flat surfaces???")
                return None,None,*dfdp
            #print("M",M)
            M_inv = -torch.linalg.inv(M)

            dxdp = N@M_inv.T
            dxdp = dxdp[:,:x.shape[0]]
            dfdp = grad(fval,inputs=params,create_graph=False,materialize_grads=True)
            fx = fx.detach()
            #print("dxdp",dxdp)
            dfdp = dxdp@fx+pack_tensors(dfdp)
            dfdp = unpack_tensors(dfdp, [elem.shape for elem in params])
            return None,None,*dfdp
"""        
class SurfaceDistanceConstraint(LEQZero):
    r"""
    Constraint that enforces a minimum distance between two 3D parametric surfaces.

    One very important constraint in any optimization procedure in the context of optical systems is the positive air spacing 
    and the minimum glass thickness constraint (see :cite:`do`). Such constraints were already implemented in the context of 
    differential ray tracing with bounding box constraints (see :cite:`do`). However, to always ensure that no surfaces overlap, 
    nonlinear constraints are needed instead of bounding box constraints. To the best of our knowledge, nonlinear constraints 
    have never been used before in the context of differential ray tracing to ensure positive air spacing or minimum glass thickness. 

    The positive air spacing and minimum glass thickness constraint are, in general, equivalent to a minimum surface distance 
    constraint (MSDC). This constraint ensures that the optical system, attained from the optimization procedure, makes 
    geometrical sense and does not have overlapping surfaces.

    Calculation of the Minimal Distance Between Two Surfaces
    ---------------------------------------------------------

    Given two parameterized surfaces :math:`S_1: \mathbb{R}^2 \to \mathbb{R}^3` and :math:`S_2: \mathbb{R}^2 \to \mathbb{R}^3`, 
    the minimum surface distance constraint function is given by:

    .. math::

        \hat{g}(p) = ||S_1(x_*, p) - S_2(x_*, p)|| - l,

    where :math:`l` is the smallest distance between the two surfaces that should be separated. :math:`x_*` is the solution 
    to the nonlinear program:

    .. math::

        \operatorname{MSDC}(p): \left\{
        \begin{aligned}
            &\min_{x} \quad \hat{f}^{MSDC}(x, p) \\
            &\text{subject to} \quad \hat{g}_i^{S_1}([x_1, x_2], p) \leq 0, \quad i = 1, \ldots, N_1, \\
            &\text{subject to} \quad \hat{g}_j^{S_2}([x_3, x_4], p) \leq 0, \quad j = 1, \ldots, N_2. \\
        \end{aligned}
        \right.

    Here:

    .. math::

        \hat{f}^{MSDC}(x, p) = ||S_1([x_1, x_2], p) - S_2([x_3, x_4], p)||.

    The constraints :math:`\hat{g}_i^{S_1}` and :math:`\hat{g}_j^{S_2}` ensure that :math:`x_*` lies within the *parametric domain* 
    of the *parametric surface description* of the two surfaces.

    In our library, we have only implemented the minimum surface distance constraint for round *optical surfaces*. In this case, 
    there are only two constraint functions :math:`\hat{g}^{S_1}` and :math:`\hat{g}^{S_2}` for the two surfaces. If the first 
    surface has radius :math:`\hat{r}_1` and the second surface has radius :math:`\hat{r}_2`, the constraints are given by:

    .. math::

        \hat{g}^{S_1}(x) := ||x||_2 - \hat{r}_1, \quad \hat{g}^{S_2}(x) := ||x||_2 - \hat{r}_2.

    Steps to Calculate :math:`x_*`
    ------------------------------

    1. **Sampling Points**  
    Sample :math:`N` points :math:`a_{(k)} \in \Xi_1` and :math:`b_{(i)} \in \Xi_2` uniformly from the *parametric domains* 
    :math:`\Xi_1` and :math:`\Xi_2` of both surfaces.

    2. **Evaluate Surface Points**  
    Evaluate :math:`v_{(k)} = S_1(a_{(k)}, p)` and :math:`w_{(i)} = S_2(b_{(i)}, p)` for all :math:`k` and :math:`i`.

    3. **Find Initial Solution**  
    Find the arguments :math:`k_*` and :math:`i_*` for which the expression:

    .. math::

        ||S_1(a_{(k_*)}, p) - S_2(b_{(i_*)}, p)||

    is minimized.

    4. **Local Optimization**  
    Perform a local optimization procedure for the *nonlinear program* :math:`\operatorname{MSDC}(p)` using the SLSQP method 
    in SciPy, with the initial solution:

    .. math::

        x_0 = \begin{bmatrix}
            a_{(k_*)} \\
            b_{(i_*)}
        \end{bmatrix}.

    Similar to the original *nonlinear program* for optimizing the optical system, one can use automatic differentiation to 
    compute the derivatives of the objective function and the constraints required to solve :math:`\operatorname{MSDC}(p)`.

    Visualization
    -------------

    The figure below shows a point :math:`x_*` that minimizes the nonlinear problem :math:`\operatorname{MSDC}(p)`:

    .. figure:: figures/constraints/surface_distance_constraint1.png
    :align: center
    :width: 40%
    :alt: Visualization of the minimum distance between two surfaces.

    Visualization of the minimum distance between two surfaces.

    Args:
        surface1 (PhysicalSurface): The first surface.
        surface2 (PhysicalSurface): The second surface.
        params (list[torch.nn.Parameter]): Parameters used in optimization.
        minimum_dist (float): Minimum allowed distance between surfaces.
        minimizer_tol (float): Tolerance for internal distance minimization.
        num_points (int): Number of sample points used.
        methdod (str): Sampling method ('sobol' or 'monte_carlo').
    """
    def __init__(self,surface1,
                 surface2,
                 params,
                 minimum_dist,
                 minimizer_tol=1e-9,
                 num_points=10000,
                 methdod="sobol"):
        super().__init__(self.get_constraint)
        #if (not isinstance(surface1,OpticalSurface))or(not isinstance(surface2,OpticalSurface)):
        #    raise RuntimeError("surface1 and surface2 must be OpticalSurface!")
        if torch.is_tensor(params):
            params = [params]

        params = [param for param in params]
        if len(params)==0:
            raise ValueError("The number of parameters provided is 0!")
        device = params[0].device
        dtype = params[0].dtype
        

        M1 = surface1.get_transformation_matrix().detach()
        M2 = surface2.get_transformation_matrix().detach()
        M3 = (torch.linalg.inv(M1)@M2)  
        if (M3[:3,:3]!=torch.eye(3,device=device,dtype=dtype)).all():
            print("SurfaceDistanceConstraintSimple: Make sure that surface1 and surface2 have the same rotation and only have a distance transformation!")
        if (M3[:2,3]!=0.0).all():
            print("SurfaceDistanceConstraintSimple: Make sure that surface1 and surface2 have the same rotation and only have a distance transformation!")
        
        self.surface1 = surface1
        self.surface2 = surface2
        self.minimum_dist = torch.tensor(minimum_dist)
        
        if not (methdod in ["sobol","monte_carlo"]):
            raise RuntimeError("SurfaceDistanceConstraint: only sobol and monte_carlo supported!")
        self.num_points = num_points
        self.methdod = methdod

        self.device = device
        self.dtype = dtype
        self.minimizer_tol = minimizer_tol
        #self.active_gs_tol = active_gs_tol
        self.params = params

    def get_closest_points3D(self):
        """
        Computes the closest 3D points between two surfaces by minimizing the Euclidean distance
        between sampled surface points under given constraints.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Closest 3D points on surface1 and surface2.
        """
        device = self.device
        dtype = self.dtype

        def find_minima(\
            parametric_surfac1,\
            parametric_surfac2,\
            kwargs1,\
            kwargs2):
            
            reference_getter1 = kwargs1["reference_getter"]
            reference_getter2 = kwargs2["reference_getter"]
            reference_to_param1 = kwargs1["reference_to_param"]
            reference_to_param2 = kwargs2["reference_to_param"]
            parametric_constraints1_leq_zero = kwargs1["parametric_constraints_leq_zero"]
            parametric_constraints2_leq_zero = kwargs2["parametric_constraints_leq_zero"]
            is_corner1 = kwargs1["is_corner"]
            is_corner2 = kwargs2["is_corner"]
            
            refval1 = None
            refval2 = None
            
            with torch.no_grad():
                reference1 = reference_getter1().to(device=device,dtype=dtype)
                reference2 = reference_getter2().to(device=device,dtype=dtype)
                param1 = reference_to_param1(reference1)
                param2 = reference_to_param2(reference2)
                points3D1 = parametric_surfac1(param1)
                points3D2 = parametric_surfac2(param2)
                arg = torch.argmin(torch.linalg.norm(points3D1-points3D2,dim=-1))
                refval1 = reference1[arg].detach().reshape(1,-1)
                refval2 = reference2[arg].detach().reshape(1,-1)
            
            refval1.requires_grad = not is_corner1
            refval2.requires_grad = not is_corner2
            constraints = [LEQZero(lambda : constraint_func(reference_to_param1(refval1))) for constraint_func in parametric_constraints1_leq_zero]
            constraints += [LEQZero(lambda : constraint_func(reference_to_param2(refval2))) for constraint_func in parametric_constraints2_leq_zero]
            def distance_fun():
                param1 = reference_to_param1(refval1)
                param2 = reference_to_param2(refval2)
                points3D1 = parametric_surfac1(param1)
                points3D2 = parametric_surfac2(param2)
                val = torch.linalg.norm(points3D1-points3D2,dim=-1).reshape(-1)
                return val
            

            minout = minimize(distance_fun,[refval1,refval2],constraints,tol=self.minimizer_tol,method="SLSQP")
            #print("minout nit",minout["nit"])
            param1 = reference_to_param1(refval1).detach().cpu()
            param2 = reference_to_param2(refval2).detach().cpu()
            return param1,param2
       
        def create_domain_kwargs(surface):
            out = {}
            out["reference_getter"] =  lambda:surface.parametric_sample(self.num_points,self.methdod)[0]
            out["reference_to_param"] =  lambda x:x
            out["parametric_constraints_leq_zero"] = surface.get_constraint_funs_leq_zero()
            out["is_corner"] = False
            return out

        surface1 = self.surface1
        surface2 = self.surface2
        
        parametric_surfac1 = lambda x: surface1.parametric_surface(x)
        parametric_surfac2 = lambda x: surface2.parametric_surface(x)
                    
        param1,param2 = find_minima(\
            parametric_surfac1,\
            parametric_surfac2,\
            create_domain_kwargs(surface1),\
            create_domain_kwargs(surface2))
        

        x = torch.cat([param1.reshape(-1),param2.reshape(-1)]).to(device=device,dtype=dtype)
        x = x.detach()        

        def g_leq_zero_wrapper1(x,g_leq_zero):
            x1 = x[:2]
            x2 = x[2:]
            return g_leq_zero(x1[None])

        def g_leq_zero_wrapper2(x,g_leq_zero):
            x1 = x[:2]
            x2 = x[2:]
            return g_leq_zero(x2[None])

        def fun(x):
            x1 = x[:2]
            x2 = x[2:]
            x1 = x1.reshape(1,-1)
            x2 = x2.reshape(1,-1)
            points3D1 = parametric_surfac1(x1)
            points3D2 = parametric_surfac2(x2)
            return points3D1,points3D2
        return fun(x)
    def get_closest_points_distance(self):
        """
        Computes the minimum distance between the two parametric surfaces.

        Returns:
            torch.Tensor: A 1D tensor with the smallest distance value.
        """
        points3D1,points3D2 = self.get_closest_points3D()
        return torch.linalg.norm(points3D1-points3D2,dim=-1).reshape(-1)
    def get_constraint(self):
        """
        Constraint function ensuring the closest points between the two surfaces
        remain at least `minimum_dist` apart.

        Returns:
            torch.Tensor: A constraint value. Should be `<= 0` for valid configurations.
        """
        device = self.device
        dtype = self.dtype
        
        def find_minima(\
            parametric_surfac1,\
            parametric_surfac2,\
            kwargs1,\
            kwargs2):
            
            reference_getter1 = kwargs1["reference_getter"]
            reference_getter2 = kwargs2["reference_getter"]
            reference_to_param1 = kwargs1["reference_to_param"]
            reference_to_param2 = kwargs2["reference_to_param"]
            parametric_constraints1_leq_zero = kwargs1["parametric_constraints_leq_zero"]
            parametric_constraints2_leq_zero = kwargs2["parametric_constraints_leq_zero"]
            is_corner1 = kwargs1["is_corner"]
            is_corner2 = kwargs2["is_corner"]
            
            refval1 = None
            refval2 = None
            
            with torch.no_grad():
                reference1 = reference_getter1().to(device=device,dtype=dtype)
                reference2 = reference_getter2().to(device=device,dtype=dtype)
                param1 = reference_to_param1(reference1)
                param2 = reference_to_param2(reference2)
                points3D1 = parametric_surfac1(param1)
                points3D2 = parametric_surfac2(param2)
                arg = torch.argmin(torch.linalg.norm(points3D1-points3D2,dim=-1))
                refval1 = reference1[arg].detach().reshape(1,-1)
                refval2 = reference2[arg].detach().reshape(1,-1)
            
            refval1.requires_grad = not is_corner1
            refval2.requires_grad = not is_corner2
            constraints = [LEQZero(lambda : constraint_func(reference_to_param1(refval1))) for constraint_func in parametric_constraints1_leq_zero]
            constraints += [LEQZero(lambda : constraint_func(reference_to_param2(refval2))) for constraint_func in parametric_constraints2_leq_zero]

            def distance_fun():
                param1 = reference_to_param1(refval1)
                param2 = reference_to_param2(refval2)
                points3D1 = parametric_surfac1(param1)
                points3D2 = parametric_surfac2(param2)
                val = torch.linalg.norm(points3D1-points3D2,dim=-1).reshape(-1)
                return val
            

            domain_result = minimize(distance_fun,[refval1,refval2],constraints,tol=self.minimizer_tol)
            param1 = reference_to_param1(refval1).detach().cpu()
            param2 = reference_to_param2(refval2).detach().cpu()
            return param1,param2
        
            functions = {}
            functions["f"] = fun
            functions["active_gs"] = self.surface1.get_constraint_funs_leq_zero()

        def create_domain_kwargs(surface):
            out = {}
            out["reference_getter"] =  lambda:surface.parametric_sample(self.num_points,self.methdod)[0]
            out["reference_to_param"] =  lambda x:x
            out["parametric_constraints_leq_zero"] = surface.get_constraint_funs_leq_zero()
            out["is_corner"] = False
            return out

        surface1 = self.surface1
        surface2 = self.surface2
        
        parametric_surfac1 = lambda x: surface1.parametric_surface(x)
        parametric_surfac2 = lambda x: surface2.parametric_surface(x)
                    
        param1,param2 = find_minima(\
            parametric_surfac1,\
            parametric_surfac2,\
            create_domain_kwargs(surface1),\
            create_domain_kwargs(surface2))
        

        x = torch.cat([param1.reshape(-1),param2.reshape(-1)]).to(device=device,dtype=dtype)
        x = x.detach()        

        def g_leq_zero_wrapper1(x,g_leq_zero):
            x1 = x[:2]
            x2 = x[2:]
            return g_leq_zero(x1[None])

        def g_leq_zero_wrapper2(x,g_leq_zero):
            x1 = x[:2]
            x2 = x[2:]
            return g_leq_zero(x2[None])

        def fun(x):
            x1 = x[:2]
            x2 = x[2:]
            x1 = x1.reshape(1,-1)
            x2 = x2.reshape(1,-1)
            points3D1 = parametric_surfac1(x1)
            points3D2 = parametric_surfac2(x2)
            val = self.minimum_dist-torch.linalg.norm(points3D1-points3D2,dim=-1).reshape(-1)
            return val
        
        def create_all_gs():
            g_leq_zero_all1 = self.surface1.get_constraint_funs_leq_zero()
            g_leq_zero_all2 = self.surface2.get_constraint_funs_leq_zero()
            
            out = [lambda x: g_leq_zero_wrapper1(x,g_leq_zero_fun) for g_leq_zero_fun in g_leq_zero_all1]
            out += [lambda x: g_leq_zero_wrapper2(x,g_leq_zero_fun) for g_leq_zero_fun in g_leq_zero_all2]
            return out
        
        sol_kwargs = {}
        sol_kwargs["f"] = fun
        sol_kwargs["gs"] = create_all_gs()
        sol_kwargs["x"] = x
        
        out= fun(x)
        if out >0.0:
            print("constraint not satisfied!",out)
        #print("constraintval == ",out)
        #out = CustomAutogradRule_KKT_active_g.apply(sol_kwargs,self.active_gs_tol,*self.params)
        return out
        

