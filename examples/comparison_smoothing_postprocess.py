#%%
import sys
import os
import gc
import tqdm
sys.path.insert(0, os.path.abspath(".."))
device = "cuda:0"

import diffinytrace as dit
from diffinytrace.source import CollimatedMonochromatic,CollimatedGaussianBeam
from diffinytrace.gaussian_smoother import GaussianSmootherSquare,make_merit_function,make_evaluation_function
import torch
import matplotlib.pyplot as plt


import pickle
path = "results/"
results_folder = path + "collimated_compare_server/"


# Save
baseline_results_file_name = results_folder+"baseline_results.pkl"

baseline_results = None
with open(baseline_results_file_name, "rb") as f:
    baseline_results = pickle.load(f)


our_results_file_name = results_folder+"our_results.pkl"

our_results = None
with open(our_results_file_name, "rb") as f:
    our_results = pickle.load(f)


plt.figure(figsize=(10,4))

baseline_conv = [result["convergence_list"][-1] for result in baseline_results]
our_conv = [result["convergence_list"][-1] for result in our_results]
sigmas = [result["sigma"] for result in our_results]
ax = plt.gca()
ax.grid(True, which='major', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
ax.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
ax.set_xlabel("$\\sigma$ [mm]")
ax.set_ylabel("Error")
plt.plot(sigmas,baseline_conv,label="Partially Smoothed")
plt.plot(sigmas,our_conv,label="Ours")
plt.title("Relationship Between Error and the Kernel Width")
plt.legend()
#plt.yscale("log")
plt.savefig(results_folder+"relationshipVA.png", dpi=400, bbox_inches='tight')


idx = [0,10,20,30,38]


plt.figure(figsize=(12,4))

baseline_conv = [result["convergence_list"][-1] for result in baseline_results]
our_conv = [result["convergence_list"][-1] for result in our_results]
sigmas = [result["sigma"] for result in our_results]
ax = plt.gca()
ax.grid(True, which='major', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
ax.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
ax.set_xlabel("$\\sigma$ [mm]")
ax.set_ylabel("Error")
plt.plot(sigmas,baseline_conv,label="Partially Smoothed")
plt.plot(sigmas,our_conv,label="Ours")
plt.title("Relationship Between Error and the Kernel Width")
plt.legend()
#plt.yscale("log")
plt.savefig(results_folder+"relationshipVB.png", dpi=400, bbox_inches='tight')


idx = torch.arange(len(baseline_results)//4)*5
idx = [0,10,20,30,38]

xbaseline_results = [baseline_results[i] for i in idx]
xour_results = [our_results[i]for i in idx]
xbaseline_results
data_grid = [[]]*4
#result["smooth_irr"] = smooth_irr.cpu().detach()
#result["none_smooth_irr"] = none_smooth_irr.cpu().detach()

data_grid[0] = [result["smooth_irr"] for result in xbaseline_results]
data_grid[1] = [result["smooth_irr"] for result in xour_results]
data_grid[2] = [result["none_smooth_irr"] for result in xbaseline_results]
data_grid[3] = [result["none_smooth_irr"] for result in xour_results]


from PIL import Image, ImageDraw, ImageFont
from importlib import reload

from image_grid_maker import image_from_grid
import image_grid_maker
reload(image_grid_maker)
from image_grid_maker import image_from_grid
import image_grid_maker

out_aperture = 8.0
vmin = 0.0
vmax = 0.03

rows_extent = [[-out_aperture,out_aperture,-out_aperture,out_aperture]]*4
rows_vidx = ["x","x","x","x"]
rows_cmap = ["jet"]*4
cbar_titles = ["[W/mm²]"]*4
columns_title = [f'σ={result["sigma"]} mm' for result in baseline_results]
columns_title = [columns_title[i] for i in idx]
rows_title = ["(Partially Smoothed)\nSmoothed Irr.","(Ours)\nSmoothed Irr.","(Partially Smoothed)\nIrr. RC","(Ours)\nIrr. RC"]

data_grid = [data_grid[0],data_grid[2],data_grid[1],data_grid[3]]
rows_title = [rows_title[0],rows_title[2],rows_title[1],rows_title[3]]

rows_vmin = [vmin]*4
rows_vmax = [vmax]*4

kwargs = dict(
        image_grid=data_grid,
        rows_extent=rows_extent,
        rows_vidx=rows_vidx,
        rows_cmap=rows_cmap,
        rows_title=rows_title,
        cbar_titles=cbar_titles,
        columns_title=columns_title,
        rows_vmin=rows_vmin,
        rows_vmax=rows_vmax,
)
out = image_from_grid(
    **kwargs,
    max_num_column=len(columns_title),
    font_size_PIL=40,
    cbar_labelsize=20,
    cbar_title_fontsize=20,
    column_title_ratio=0.3
    )
out = out[0]



# Load an image from the file path
image = Image.open(out)
image.save(results_folder+"comparison_collimated.png")

out_aperture = 8.0

import numpy as np
import plotly.graph_objects as go


in_aperture = 4.0
in_aperture_lens = 5.0

desired_width_square = 4.0

out_aperture = 8.0
source_wl = 0.589
source_gaussian_constant = 0.035

light_transform = dit.transforms.Identity()
source = CollimatedGaussianBeam(light_transform,in_aperture,source_wl,source_gaussian_constant)

lens_mat = dit.materials["NBK7"]
env_mat = dit.materials["NONE"]

lens1_thickness = 2.
lens1_surf1 = dit.Bspline(in_aperture_lens,[4,4],[11,11])#dit.Legendre(in_aperture_lens,20)#
lens1_surf2 = dit.Plane()

det_surf = dit.Plane()

lens1_transform = dit.transforms.Distance(5.0,parent_transform=source)
lens1_transform.distance.requires_grad = False

lens1 = dit.Lens(lens1_transform,lens1_thickness,lens1_surf1,lens1_surf2,lens_mat,in_aperture_lens,is_square=False)


det_transform = dit.transforms.Distance(10.,parent_transform=lens1)
det_transform.distance.requires_grad = False
detector = dit.Detector(det_transform,det_surf,out_aperture)

system = dit.SequentialOpticalSystem({"source":source,"lens1":lens1,"detector":detector},env_mat)
    
system.cpu()
source = system.modules_dict["source"]
lens1 = system.modules_dict["lens1"]
detector = system.modules_dict["detector"]

in_aperture = 4.0
num_bins = 512

x = np.linspace(-in_aperture, in_aperture, num_bins)  # Width
y = np.linspace(-in_aperture, in_aperture, num_bins)  # Height
z = torch.zeros((num_bins, num_bins))

tmp = np.meshgrid(x,y)
tmp = torch.tensor([tmp])[0].reshape(2,-1).T

irr_source = source.get_flux(tmp)
irr_source[torch.linalg.norm(tmp,dim=1)>in_aperture]=torch.nan
irr_source = irr_source.reshape(num_bins,num_bins)
print("irr_source",irr_source.shape)

desired_width_square = 4.
def desired_irradiance_func(y):
    out = (torch.abs(y[:,0]) < desired_width_square).float() * (torch.abs(y[:,1]) < desired_width_square).float()
    return out/((desired_width_square*2)**2)
#dit.plotting.quantity2D.plot(irr_source,title="Radiant Exitance [W/mm²]",x_range=[-out_aperture,out_aperture],cmap="hot")

x = np.linspace(-out_aperture, out_aperture, num_bins)  # Width
y = np.linspace(-out_aperture, out_aperture, num_bins)  # Height
z = torch.zeros((num_bins, num_bins))

tmp = np.meshgrid(x,y)
tmp = torch.tensor([tmp])[0].reshape(2,-1).T

desired_irr = desired_irradiance_func(tmp.reshape(-1,2)).reshape(num_bins,num_bins)
print("desired_irr",desired_irr.shape,tmp.shape)

#vmax = torch.max(irr_source).item()

import matplotlib.pyplot as plt

# Assume you already have:
# - irrs: list of 2D arrays (irradiance maps)
# - rows_extent: list of [xmin, xmax, ymin, ymax] per image
# - sigmas: list of sigma values used for smoothing

cbar_labelsize=12
cbar_title_fontsize=15
# Grid dimensions
rows_extent = [[-in_aperture, in_aperture, -in_aperture, in_aperture]] +[[-out_aperture, out_aperture, -out_aperture, out_aperture]]
irrs = [irr_source,desired_irr]
irrs = [irr.cpu() for irr in irrs]

num_rows = 1
num_cols = len(irrs)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4), constrained_layout=True)

# Titles for each column
columns_title = ["Radiant Exitance"] + [f"Desired Irradiance" for sigma in sigmas]
cmaps = ["jet","jet"]
cbar_title = "[W/mm²]"

for k in range(num_cols):
    ax = axes[k]
    img = irrs[k]
    cmap = cmaps[k]
    im = ax.imshow(img, extent=rows_extent[k],origin='lower', cmap=cmap,interpolation="nearest",vmin=vmin,vmax=vmax)
    ax.set_title(columns_title[k],fontsize=cbar_title_fontsize)
    if k != 0:
        ax.set_xticks([-4,0,4])
    ax.set_yticks([])
    ax.tick_params(labelsize=cbar_labelsize)

    cbar = plt.colorbar(im,ax=ax,shrink=0.65,aspect=9)  # Add a colorbar for reference
    cbar.ax.tick_params(labelsize=cbar_labelsize)
    cbar.ax.set_title(cbar_title, fontsize=cbar_title_fontsize, pad=10,loc="left")  # Set label above
    offset_text = cbar.ax.yaxis.offsetText
    offset_text.set_size(cbar_labelsize)  # Set the font size
    offset_text.set_ha('left')  # Align the text to the left

#plt.suptitle("Irradiance Maps from Ray Counting and Smoothing", fontsize=16)
plt.savefig(results_folder+"radiant_exitance_desired_irr.png", dpi=400, bbox_inches='tight')

plt.show()

font_multi = 1.3
for k in range(num_cols):
    ax = plt.gca()
    img = irrs[k]
    cmap = "jet"
    im = ax.imshow(img, extent=rows_extent[k],origin='lower', cmap=cmap,interpolation="nearest",vmin=vmin,vmax=vmax)
    ax.set_title(columns_title[k],fontsize=cbar_title_fontsize*font_multi)
    if k != 0:
        ax.set_xticks([-4,0,4])
    ax.set_yticks([])
    ax.tick_params(labelsize=cbar_labelsize*font_multi)

    cbar = plt.colorbar(im,ax=ax,shrink=0.65,aspect=9)  # Add a colorbar for reference
    cbar.ax.tick_params(labelsize=cbar_labelsize*font_multi)
    cbar.ax.set_title(cbar_title, fontsize=cbar_title_fontsize*font_multi, pad=10,loc="left")  # Set label above
    offset_text = cbar.ax.yaxis.offsetText
    offset_text.set_size(cbar_labelsize*font_multi)  # Set the font size
    offset_text.set_ha('left')  # Align the text to the left

    plt.savefig(results_folder+f"radiant_exitance_desired_irr_sep{k}.png", dpi=400, bbox_inches='tight')
    plt.show()

# %%
