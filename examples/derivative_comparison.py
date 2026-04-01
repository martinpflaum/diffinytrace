#%%
import sys
import os

sys.path.append(os.path.abspath(".."))

import diffinytrace as dit
import torch
import numpy as np
torch.set_default_dtype(torch.float64)

aperture_radius = 12.7
NBK7 = dit.materials["NBK7"]
dit.plotting.wavelength.plot(NBK7)
#%%
n_enviroment = 1.000293
wave_len = 0.5328

lens_pos1D = 0.5
lens_thickness = 6.5
curvature = 0.05
detector_distance = 25.
device = "cuda:0"
grid_size = 32


light_transform = dit.transforms.Offset(torch.tensor([0.0,0.0,0.0]))
light_source = dit.source.CollimatedMonochromatic(light_transform,aperture_radius,wave_len)
light_transform.pos.requires_grad = False
#get_anchor_transform
#0.5


lens_transform = dit.transforms.Distance(lens_pos1D,parent_transform=light_transform)
lens_transform.distance.requires_grad = False
surface1 = dit.Aspheric(curvature=curvature)
surface2 = dit.Plane()
lens1 = dit.Lens(lens_transform,lens_thickness,surface1,surface2,NBK7,aperture_radius)
lens1.lens_thickness.requires_grad = False

#detector_transform = dit.PositionTransform(torch.tensor([0.0,0.0,25.0+0.5]))
detector_transform = dit.transforms.Distance(detector_distance)#25.0+0.5
detector_transform.distance.requires_grad = False
plane_surface = dit.Plane()
detector = dit.Detector(detector_transform,plane_surface,aperture_radius)
gridxt = torch.linspace(-aperture_radius,aperture_radius,grid_size)
grid_delta =gridxt[1]-gridxt[0] 
grid_delta


system = dit.SequentialOpticalSystem({"source":light_source,"lens":lens1,"detector":detector})


x,weights = light_source.sample(50)
x.requires_grad = True
sequence = ["source","lens","detector"]
O,D,wave_len,_,RayPaths = system(x,sequence)


dit.plotting.system2D.plot(system,RayPaths,500,False)
#dit.plotting.system3D.plot(system,RayPaths)


num_rays = 100000

def gp2(x):
    O,D,wave_len,_,RayPaths = system(x,sequence)
    O_local = detector.to_local_pos(O)
    return O_local[:,[0,1]],O
x,weights = light_source.sample(num_rays)
num_rays = x.shape[0]
    
grid = dit.target_grid.GridSquare(aperture_radius,grid_size=grid_size)
constant_fac = weights/grid.get_pixel_area()

y,O = gp2(x)
    
irradiance = None
Qval = light_source.get_flux(x.detach()).detach()
sigma=grid_delta*0.5
irradiance = dit.gaussian_smoother.gaussian_func_2D(detector.to_local_pos(O)[:,[0,1]],[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],grid_size,grid_size,sigma,val_multi=Qval*constant_fac,include_boundary=False)

tmp = torch.ones_like(irradiance)
tmp.requires_grad = True

dp_summed, = dit.grad(irradiance,inputs=[surface1.curvature],grad_outputs=tmp,materialize_grads=True)
    
dIdp_classical, = dit.grad(dp_summed,inputs=[tmp])
irradiance_smoothed = irradiance


def calc_gradientA(light_source,optical_system,sequence,target_grid,num_rays):    
    device = "cpu"#optical_system.device
    parameter = [surface1.curvature]
    def gp2(x):
        O,D,wave_len,_,RayPaths = optical_system(x,sequence)
        O_local = detector.to_local_pos(O)
        return O_local[:,[0,1]],O
    
    x,weights = light_source.sample(num_rays,"sobol")
    if len(x.shape) > 2:
        raise RuntimeError("light source sample function should be off shape [num_rays,dims]")
    num_rays = x.shape[0]
    x.requires_grad = True
    

    constant_fac = weights/target_grid.get_pixel_area()
        
    
    y,O = gp2(x)
    
    irradiance = None
    with torch.no_grad():
        Qval = light_source.get_flux(x.detach()).detach()
        irradiance = target_grid.sum(detector.to_local_pos(O)[:,[0,1]],Qval*constant_fac)
        irradiance = irradiance.detach()
        
    dmdI = torch.ones_like(irradiance)
    dmdI.requires_grad = True
    dmdIk = target_grid.map_matrix_to_ray(detector.to_local_pos(O.detach())[:,[0,1]],dmdI).reshape(-1)
    dmdIk = dmdIk*constant_fac
    
    dy1_dx, = dit.grad(y[:,0],inputs=[x],grad_outputs=torch.ones_like(y[:,0]))
    dy2_dx, = dit.grad(y[:,1],inputs=[x],grad_outputs=torch.ones_like(y[:,1]))
    dy_dx = torch.zeros((x.shape[0],2,2),device=device)
    dy_dx[:,0] = dy1_dx
    dy_dx[:,1] = dy2_dx
    H = torch.det(dy_dx).reshape(-1)
    #H = H+torch.sign(H)*0.001

    Q = light_source.get_flux(x)
    dQdx, = dit.grad(Q,inputs=[x],grad_outputs=torch.ones_like(Q))

    dHdx, = dit.grad(H,inputs=[x],grad_outputs=torch.ones_like(H))
    
    dy_dx_inv = torch.zeros((num_rays,2,2),device=device)
    dy_dx_inv[:,0,0 ]= dy_dx[:,1,1]
    dy_dx_inv[:,0,1] = -dy_dx[:,0,1]
    dy_dx_inv[:,1,0] = -dy_dx[:,1,0]
    dy_dx_inv[:,1,1] = dy_dx[:,0,0]
    dy_dx_inv = dy_dx_inv/H.reshape(-1,1,1)
    

    dy_dx_inv_dHdx = torch.bmm(dy_dx_inv,dHdx.reshape(-1,2,1)).reshape(-1,2)
    grad_outputs_dgdp_H = -dmdIk.reshape(-1,1)*(((1.0/H.reshape(-1,1))*dy_dx_inv_dHdx*Q.reshape(-1,1)).detach())
    grad_outputs_dHdp = -dmdIk.reshape(-1)*(((1.0/H.reshape(-1))*Q.reshape(-1)).detach())
    
    dy_dx_inv_dQdx = torch.bmm(dy_dx_inv,dQdx.reshape(-1,2,1)).reshape(-1,2)
    grad_outputs_dgdp_Q = dmdIk.reshape(-1,1)*(dy_dx_inv_dQdx.detach())
    grad_outputs_dQdp = dmdIk.reshape(-1)

    grad_outputs_dgdp = grad_outputs_dgdp_H+grad_outputs_dgdp_Q
    
    dHdp, = dit.grad(H,inputs=parameter,grad_outputs=grad_outputs_dHdp,materialize_grads=True)
    dQdp, = dit.grad(Q,inputs=parameter,grad_outputs=grad_outputs_dQdp,materialize_grads=True)
    dgdp_H_Q, = dit.grad(y,inputs=parameter,grad_outputs=grad_outputs_dgdp,materialize_grads=True)
    
    pgrad = dHdp+dQdp+dgdp_H_Q
    out, = dit.grad(pgrad,inputs=dmdI)

    return out.detach(),irradiance.detach()

dIdp_second_order,true_irradiance = calc_gradientA(light_source,system,sequence,grid,num_rays)
#%%
dit.plotting.quantity2D.plot(true_irradiance.detach(),"irradiance true",[-aperture_radius,aperture_radius])
dit.plotting.quantity2D.plot(dIdp_second_order.detach(),"irradiance derivative",[-aperture_radius,aperture_radius])

#%%
dit.plotting.quantity2D.plot(irradiance.detach(),f"irradiance distribution",[-aperture_radius,aperture_radius])
dit.plotting.quantity2D.plot(dIdp_classical.detach(),f"dIdp_classical",[-aperture_radius,aperture_radius])
#%%
system.to(device)

"""
nearest_ray = grid.get_nearest_ray(O_local)
num_rays_on_grid = grid.sum(O_local,torch.ones((x.shape[0]),dtype=x.dtype,device=x.device))
dit.plotting.quantity2D.plot(nearest_ray,"",[-detector.aperture_radius,detector.aperture_radius],show=False,cmap="Spectral")
nearest_ray = nearest_ray.reshape(-1)
plt.scatter(O_local[nearest_ray,0].detach(),O_local[nearest_ray,1].detach(),c="black",s=10,marker="x")
"""
#%%


import torch.nn as nn
device = "cpu"#optical_system.device
parameter = [surface1.curvature]



def get_x_closest_to_pix(tol): 
    all_pix_centers = grid.get_pixel_centers()
    all_pix_centers = all_pix_centers.reshape(-1,2)
    num_rays = 400000
    x,weights = light_source.sample(num_rays)
    O,D,wave_len,_,RayPaths = system(x,sequence)
    O_local = detector.to_local_pos(O)
    O_local = O_local[:,[0,1]]
    nearest_has_hit = grid.nearest(O_local,return_args=True)[1]
    nearest_has_hit = nearest_has_hit.reshape(-1)


    x_inits = x[nearest_has_hit[nearest_has_hit.reshape(-1)!=-1]]
    all_pix_centers = all_pix_centers[nearest_has_hit.reshape(-1)!=-1]

    x_opti = nn.Parameter(torch.tensor([[0.0,0.0]]))
    x_opti.bounds = torch.tensor([[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius]])
    def in_circle():
        return aperture_radius-torch.linalg.norm(x_opti)
        
    
    constraint = dit.constraints.GEQZero(in_circle)

    def make_objective_fun(pix_center):
        def fun():
            #print(x_opti)
            O,D,wave_len,_,RayPaths = system(x_opti,sequence)
            O_local = detector.to_local_pos(O)
            out = torch.linalg.norm(O_local[:,[0,1]]-pix_center)
            #print(O_local[:,[0,1]],pix_center)
            return out
        
        return fun


    all_pix_centers = all_pix_centers.reshape(-1,2)
    out = []
    from tqdm import tqdm
    for k in tqdm(range(all_pix_centers.shape[0])):
        x_init = x_inits[k]
        with torch.no_grad():
            x_opti.data = x_init
        #dit.optimize.apply_vec_to_params(np.array(x_init),x)
        pix_center = all_pix_centers[k]
        tmp = dit.minimize(make_objective_fun(pix_center),x_opti,tol=tol)
        out += [tmp]
        #print("converged")
    xvals = [out[k]["x"] for k in range(len(out))]
    xvals = np.array(xvals)
    xvals = torch.tensor(xvals,dtype=torch.get_default_dtype())
    xvals = xvals[light_source.integrator.in_bounds(xvals)]

    return xvals


def calc_irradiance_fast(x):
    x = x.detach()
    x.requires_grad=True
    y,O = gp2(x)
    
    dy1_dx, = dit.grad(y[:,0],inputs=[x],grad_outputs=torch.ones_like(y[:,0]))
    dy2_dx, = dit.grad(y[:,1],inputs=[x],grad_outputs=torch.ones_like(y[:,1]))
    dy_dx = torch.zeros((x.shape[0],2,2),device=device)
    dy_dx[:,0] = dy1_dx
    dy_dx[:,1] = dy2_dx
    H = torch.det(dy_dx).reshape(-1)
    Q = light_source.get_flux(x)
    vals = Q/torch.abs(H)
    return grid.sum(y,vals)

h = 1e-100

with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature)


with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature+h*0.5)

irr_pplus = calc_irradiance_fast(get_x_closest_to_pix(tol=1e-12))

with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature-h*0.5)

irr_pminus = calc_irradiance_fast(get_x_closest_to_pix(tol=1e-12))

with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature)


shape = irr_pminus.shape
irr_pminus = irr_pminus.reshape(-1)
irr_pplus = irr_pplus.reshape(-1)
irr_pminus[irr_pplus==0.0] = torch.nan
irr_pplus[irr_pminus==torch.nan] = torch.nan

irr_pplus = irr_pplus.reshape(*shape)
irr_pminus = irr_pminus.reshape(*shape)

#%%
dit.plotting.quantity2D.plot(irr_pminus,"irr_pminus",[-aperture_radius,aperture_radius])
dit.plotting.quantity2D.plot(irr_pplus,"irr_pplus",[-aperture_radius,aperture_radius])

#%%
dit.plotting.quantity2D.plot((irr_pplus-irr_pminus)/h,"finite differences accurate",[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],xlabel="x",ylabel="y")

#%%

num_rays = 800000
h = 1e-3

with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature+h*0.5)

irr_pplus = dit.render.binned_irradiance(system,sequence,light_source,detector,grid,num_rays=num_rays,method_ray_tracing="sobol")
    
with torch.no_grad():
    surface1.curvature.data = torch.tensor(curvature-h*0.5)

irr_pminus = dit.render.binned_irradiance(system,sequence,light_source,detector,grid,num_rays=num_rays,method_ray_tracing="sobol")
    
# %%
dit.plotting.quantity2D.plot((irr_pplus-irr_pminus)/h,"finite differences smooth",[-aperture_radius,aperture_radius])

# %%
dit.plotting.quantity2D.plot(irr_pminus,"irr_pminus",[-aperture_radius,aperture_radius])
dit.plotting.quantity2D.plot(irr_pplus,"irr_pplus",[-aperture_radius,aperture_radius])


# %%
