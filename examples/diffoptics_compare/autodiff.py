#%%
"""
MIT License

Copyright (c) 2023 vccimaging

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

import sys
sys.path.append(os.path.abspath("../.."))
import diffinytrace as dit

#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

import diffoptics as do

# initialize a lens
device = torch.device('cpu')
lens = do.Lensgroup(device=device)

save_dir = './autodiff_demo/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

R = 12.7
surfaces = [
    do.Aspheric(R, 0.0, c=0.05, device=device),
    do.Aspheric(R, 6.5, c=0., device=device)
]
materials = [
    do.Material('air'),
    do.Material('N-BK7'),
    do.Material('air')
]
lens.load(surfaces, materials)
lens.d_sensor = 25.0
lens.r_last = 12.7

# generate array of rays
wavelength = torch.Tensor([532.8]).to(device) # [nm]
R = 10.0 # [mm]
def render():
    ray_init = lens.sample_ray(wavelength, M=9, R=R, sampling='grid')
    ps = lens.trace_to_sensor(ray_init)
    return ps[...,:2]

def trace_all():
    ray_init = lens.sample_ray_2D(R, wavelength, M=11)
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[...,:2], oss

def compute_Jacobian(ps):
    Js = []
    for i in range(1):
        J = torch.zeros(torch.numel(ps))
        for j in range(torch.numel(ps)):
            mask = torch.zeros(torch.numel(ps))
            mask[j] = 1
            ps.backward(mask.reshape(ps.shape), retain_graph=True)
            J[j] = lens.surfaces[i].c.grad.item()
            lens.surfaces[i].c.grad.data.zero_()
        J = J.reshape(ps.shape)

    # get data to numpy
    Js.append(J.cpu().detach().numpy())
    return Js


N = 1
cs = [0.05]#[0.045]
Iss = []
Jss = []
for index, c in enumerate(cs):
    index_string = str(index).zfill(3)
    # load optics
    lens.surfaces[0].c = torch.Tensor(np.array(c))
    lens.surfaces[0].c.requires_grad = True
    
    # show trace figure
    ps, oss = trace_all()
    ax, fig = lens.plot_raytraces(oss, color='b-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(save_dir + "layout_trace_" + index_string + ".png", bbox_inches='tight')

    # show spot diagram
    RMS = lambda ps: torch.sqrt(torch.mean(torch.sum(torch.square(ps), axis=-1)))
    ps = render()
    rms_org = RMS(ps)
    print(f'RMS: {rms_org}')
    lens.spot_diagram(ps, xlims=[-4, 4], ylims=[-4, 4], savepath=save_dir + "spotdiagram_" + index_string + ".png", show=False)

    # compute Jacobian
    Js = compute_Jacobian(ps)[0]
    print(Js.max())
    print(Js.min())
    ps_ = ps.cpu().detach().numpy()
    fig = plt.figure()
    x, y = ps_[:,0], ps_[:,1]
    plt.plot(x, y, 'b.', zorder=0)
    plt.quiver(x, y, Js[:,0], Js[:,1], color='b', zorder=1)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    fig.savefig(save_dir + "flow_" + index_string + ".png", bbox_inches='tight')

    # compute images
    ray = lens.sample_ray(wavelength.item(), view=0.0, M=2049*2, sampling='grid')
    print(ray.o.shape)
    lens.film_size = [512, 512]
    lens.pixel_size = 50.0e-3/2
    I = lens.render(ray)
    I = I.cpu().detach().numpy()
    lm = do.LM(lens, ['surfaces[0].c'], 1e-2, option='diag')
    JI = lm.jacobian(lambda: lens.render(ray)).squeeze()
    J = JI.abs().cpu().detach().numpy()

    Iss.append(I)
    Jss.append(J)
    plt.close()

Iss = np.array(Iss)
Jss = np.array(Jss)
for i in range(len(cs)):
    
    plt.imsave(save_dir + "I_" + str(i).zfill(3) + ".png", Iss[i], cmap='gray')
    plt.imsave(save_dir + "J_" + str(i).zfill(3) + ".png", Jss[i], cmap='gray')

names = [
    'spotdiagram',
    'layout_trace',
    'I',
    'J',
    'flow'
]

# %%


final_plot_radius = 512*(50.0e-3/2)/2

#dit.plotting.quantity2D.plot(Iss[0]/(Iss[0]*lens.pixel_size**2).sum(),"Irradiance",[-final_plot_radius,final_plot_radius])
#dit.plotting.quantity2D.plot(Jss[0]/(Iss[0]*lens.pixel_size**2).sum(),"Derivative",[-final_plot_radius,final_plot_radius])

val = Jss[0]/(Iss[0]*lens.pixel_size**2).sum()
x_range = [-final_plot_radius,final_plot_radius]
y_range = [-final_plot_radius,final_plot_radius]
from matplotlib.colors import LogNorm
norm = LogNorm()
plt.imshow(val,cmap="jet",extent=list(x_range)+list(y_range),norm=norm)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.colorbar()    
#%%

val = Iss[0]/(Iss[0]*lens.pixel_size**2).sum()
x_range = [-final_plot_radius,final_plot_radius]
y_range = [-final_plot_radius,final_plot_radius]
from matplotlib.colors import LogNorm
norm = LogNorm()
plt.imshow(val,cmap="jet",extent=list(x_range)+list(y_range))
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.colorbar()    


# %%

aperture_radius = 12.7
do_air = do.Material('air')
do_nbk7 = do.Material('N-BK7')
dit_nbk7 = dit.RefractiveIndex(lambda x:do_nbk7.A + do_nbk7.B / (x*1000)**2,[0.1,1.5])
dit_air = dit.RefractiveIndex(lambda x:do_air.A + do_air.B / (x*1000)**2,[0.1,1.5])


wave_len = 0.5328

lens_pos1D = 0.5
lens_thickness = 6.5
curvature = cs[0]
detector_distance = 25.
device = "cuda:0"
grid_size = 512


light_transform = dit.transforms.Offset(torch.tensor([0.0,0.0,0.0]))
light_source = dit.source.CollimatedMonochromatic(light_transform,aperture_radius,wave_len)
light_transform.pos.requires_grad = False
#get_anchor_transform
#0.5


lens_transform = dit.transforms.Distance(lens_pos1D,parent_transform=light_transform)
lens_transform.distance.requires_grad = False
surface1 = dit.Aspheric(curvature=curvature)
surface2 = dit.Plane()
lens1 = dit.Lens(lens_transform,lens_thickness,surface1,surface2,dit_nbk7,aperture_radius)
lens1.lens_thickness.requires_grad = False

#detector_transform = dit.PositionTransform(torch.tensor([0.0,0.0,25.0+0.5]))
detector_transform = dit.transforms.Distance(detector_distance)#25.0+0.5
detector_transform.distance.requires_grad = False
plane_surface = dit.Plane()

detector = dit.Detector(detector_transform,plane_surface,final_plot_radius)


system = dit.SequentialOpticalSystem({"source":light_source,"lens":lens1,"detector":detector},n_func_enviroment = dit_air)


x,weights = light_source.sample(50)
x.requires_grad = True
sequence = ["source","lens","detector"]
O,D,wave_len,_,RayPaths = system(x,sequence)


dit.plotting.system2D.plot(system,RayPaths,500,False)
# %%
#M=2049*2**2
num_rays = (2049*2)**2
grid = dit.target_grid.GridSquare(final_plot_radius,grid_size=grid_size)
binned_irradiance = dit.render.binned_irradiance(system,sequence,light_source,detector,grid,num_rays=num_rays,method_ray_tracing="monte_carlo")
dit.plotting.quantity2D.plot(binned_irradiance, "Irradiance", [-final_plot_radius, final_plot_radius], [-final_plot_radius, final_plot_radius])
#%%

#smoother = dit.gaussian_smoother.GaussianSmootherSquare(final_plot_radius,grid_size=grid_size,sigma=0.1,desired_irradiance_fun=lambda x: torch.zeros_like(x),smoothed_num_integration_points=8,smoothed_num_splits=1)
#dit.render.smoothed_irradiance(system,sequence,light_source,detector,smoother,num_rays=num_rays,method_ray_tracing="monte_carlo")

#Qval = light_source.get_flux(x.detach()).detach()
#sigma=grid_delta*0.5
#irradiance = dit.gaussian_smoother.gaussian_func_2D(detector.to_local_pos(O)[:,[0,1]],[-aperture_radius,aperture_radius],[-aperture_radius,aperture_radius],grid_size,grid_size,sigma,val_multi=Qval*constant_fac,include_boundary=False)

#%%
num_rays = 2**19
def gp2(x):
    O,D,wave_len,_,RayPaths = system(x,sequence)
    O_local = detector.to_local_pos(O)
    return O_local[:,[0,1]],O
x,weights = light_source.sample(num_rays)
num_rays = x.shape[0]




#%%
grid = dit.target_grid.GridSquare(aperture_radius,grid_size=grid_size)


#%%

(binned_irradiance*lens.pixel_size**2).sum()
# %%
