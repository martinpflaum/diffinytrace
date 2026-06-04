#%%
import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import diffinytrace as dit
from examples.sunlight_picture import create_lens
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
from image_grid_maker import image_from_grid,concatenate_images_vertical
import os
import pickle
import gc

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return data

def create_folder(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        return f"Folder created successfully at: {folder_path}" if not os.path.exists(folder_path) else f"Folder already exists at: {folder_path}"
    except Exception as e:
        return f"An error occurred: {e}"

device = "cuda:0"
#image_file_name = "image_vertical.png"

results_folder_in = "results/results_final_lens_picture/lens_picture/"#.2"
results_folder_out = "results/results_final_lens_picture/lens_picture_plots/"
create_folder(results_folder_out)


import torch
all_main_subfolders = ["results_classical","results_desired_irr_smoothing","results_margin"]

def get_folder(idx):
    out = results_folder_in
    main_subfolder = all_main_subfolders[idx]
    out += main_subfolder
    return out

def load_results(idx):
    folder = get_folder(idx)
    results = load_data(folder+"/results_dict.pkl")
    return results

results_classical_e = load_results(0)
results_ours_e = load_results(1)
results_margin_e = load_results(2)
results_ours_e.keys()


subtitle_classical_short_e = "Partially Smoothed"
subtitle_ours_short_e = "Fully Smoothed"
subtitle_margin_short_e = "Sliding Window"
    
columns_title = [subtitle_classical_short_e,subtitle_ours_short_e,subtitle_margin_short_e]
all_results = [results_classical_e,results_ours_e,results_margin_e]

title_all = {}
title_all["Irradiance MC showcase"] = "Irradiance RC"
title_all["Irr. Smooth"] = f"Smoothed Irr."
title_all["Relative Surface Profile"] = f"Surface Profile"
    
keys1 = ["Irradiance MC showcase","Irr. Smooth", "Relative Surface Profile"]


def create_2d4x4_plots():
    results_classical = load_results(0)
    results_desired_irr_smoothing = load_results(1)
    results_margin = load_results(2)
    
    
    subtitle_classical = "Method: Classical Algorithmic Differentiable Ray Tracing"
    subtitle_desired_irr_smoothing = "Method: Desired Irradiance Smoothing"
    subtitle_margin = "Method: Sliding Window"
    

    subtitle_classical_short = "Method: CADRT"
    subtitle_desired_irr_smoothing_short = "Method: DIS"
    subtitle_margin_short = "Method: Sliding Window"
    

    all_subtitles = [subtitle_classical,subtitle_desired_irr_smoothing,subtitle_margin]
    all_subtitles_short = [subtitle_classical_short,subtitle_desired_irr_smoothing_short,subtitle_margin_short]
    all_results = [results_classical,results_desired_irr_smoothing,results_margin]

    def get_kwards_from_index(index):
        if index > len(all_subtitles):
            raise ValueError("index must be > len(all_subtitles)")
        subtitle = all_subtitles[index]
        subtitle_short = all_subtitles_short[index]
        result = all_results[index]
        return result,subtitle,subtitle_short

    def create_convergence_plot_res(title,quantity_key,y_label, file_name_out):
        plt.figure(figsize=(3,4))
        ax = plt.gca()  # Slightly wider for space
        

        refine_iters_all = []
        refine_fun_all = []
        point_style_all = []
        labels_all = []

        for k in [1,2,0]:
            result, subtitle, _ = get_kwards_from_index(k)
            results_minimize = result["results_minimize"]
            all_fun_vals = []
            all_iters = []
            refine_iter = []
            refine_fun = []
            
            current_iter = 0
            for i in range(len(results_minimize)):
                history = results_minimize[i]["history"]
                fun_vals = history[quantity_key]
                fun_vals = np.array(fun_vals)
                iters = np.arange(len(fun_vals)) + current_iter
                current_iter = iters[-1]
                sigma_val = results_minimize[i]["sigma"]
                coeff_shape = results_minimize[i]["coeff_shape"]
                #ax.plot(iters, fun_vals,linestyle=linestyle, label=f"{label_prefix} (n={coeff_shape[0]})")
                #ax.axvline(x=current_iter, color='gray', linestyle='--', linewidth=1.2)
                if i != len(results_minimize)-1:
                    refine_iter.append(iters[-1])
                    refine_fun.append(fun_vals[-1])
                    
                all_fun_vals.append(fun_vals)
                all_iters.append(iters)
            label_prefix = "Fully Smoothed"
            point_style = "o"
            if k == 0:
                label_prefix="Partially Smoothed"
                point_style = "o"
            elif k == 2:
                label_prefix="Sliding Window"
                point_style = "s"
            
            all_iters = np.concatenate(all_iters)
            all_fun_vals = np.concatenate(all_fun_vals)
            ax.plot(all_iters, all_fun_vals,linestyle="-", label=f"{label_prefix}")
            labels_all.append("Refinements ("+label_prefix+")")
            point_style_all.append(point_style)
            refine_iters_all += [refine_iter]
            refine_fun_all += [refine_fun]
            

        for k in range(3):
            color = "red"
            if k == 0:
                color = "black"
            elif k == 2:
                color = "green"
            
            label = labels_all[k]
            point_style = point_style_all[k]
            refine_iter = refine_iters_all[k]
            refine_fun = refine_fun_all[k]
            ax.plot(refine_iter,refine_fun,point_style,color=color, label=label)

            #ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='-', linewidth=1)  # Minor grid lines (finer)
        ax.set_title(title)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(y_label)

            # Move legend outside
        #, fontsize=8
        ax.legend()
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        print("hello there")
        
        #plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legends
        plt.savefig(file_name_out, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # Free up memory

    def create_convergence_plot_default(title,quantity_key,y_label, file_name_out):
        plt.figure(figsize=(3,4))
        ax = plt.gca()  # Slightly wider for space
        refine_iters_all = []
        refine_fun_all = []
        point_style_all = []
        labels_all = []
        
        for k in [1,2,0]:
            result, subtitle, _ = get_kwards_from_index(k)
            results_minimize = result["results_minimize"]
            refine_iter = []
            refine_fun = []
            
            current_iter = 0
            all_iters = []
            all_fun_vals = []
            for i in range(len(results_minimize)):
                history = results_minimize[i]["history"]
                fun_vals = history[quantity_key]
                fun_vals = np.array(fun_vals)
                iters = np.arange(len(fun_vals)) + current_iter
                current_iter = iters[-1]
                sigma_val = results_minimize[i]["sigma"]
                coeff_shape = results_minimize[i]["coeff_shape"]
                all_fun_vals.append(fun_vals)
                all_iters.append(iters)

                if i != len(results_minimize)-1:
                    refine_iter.append(iters[-1])
                    refine_fun.append(fun_vals[-1])
                
            
            label_prefix = "Fully Smoothed"
            point_style = "o"
            if k == 0:
                label_prefix="Partially Smoothed"
                point_style = "o"
            elif k == 2:
                label_prefix="Sliding Window"
                point_style = "s"
            
            all_iters = np.concatenate(all_iters)
            all_fun_vals = np.concatenate(all_fun_vals)
            ax.plot(all_iters, all_fun_vals,linestyle="-", label=f"{label_prefix}")
                #ax.axvline(x=current_iter, color='gray', linestyle='--', linewidth=1.2)
            labels_all.append("Refinements ("+label_prefix+")")
            point_style_all.append(point_style)
            refine_iters_all += [refine_iter]
            refine_fun_all += [refine_fun]
            

        for k in range(3):
            color = "red"
            if k == 0:
                color = "black"
            elif k == 2:
                color = "green"
            
            label = labels_all[k]
            point_style = point_style_all[k]
            refine_iter = refine_iters_all[k]
            refine_fun = refine_fun_all[k]
            ax.plot(refine_iter,refine_fun,point_style,color=color, label=label)

            #ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='-', linewidth=1)  # Minor grid lines (finer)
        ax.set_title(title)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(y_label)

            # Move legend outside
        #, fontsize=8
        ax.legend()
        #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        print("hello there")
        
        #plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to fit legends
        plt.savefig(file_name_out, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # Free up memory


        
    file_name_out_merit = results_folder_out + f"/convergence_merit_fun.png"
    create_convergence_plot_res("Convergence of Merit Function","fun_vals","$m$", file_name_out_merit)

    file_name_out_error = results_folder_out + f"/convergence_error.png"
    create_convergence_plot_default("Convergence of $L_2$ Error","convergence","$L_2$ Error", file_name_out_error)

    file_name_out_ssim = results_folder_out + f"/convergence_ssim.png"
    create_convergence_plot_default("Convergence of SSIM","ssim","SSIM", file_name_out_ssim)

    file_name_out_rmse = results_folder_out + f"/convergence_rmse.png"
    create_convergence_plot_default("Convergence of RMSE","rmse","RMSE", file_name_out_rmse)


create_2d4x4_plots()

results_classical_e["results_minimize"][-1].keys()

aperture_radius_detector = results_classical_e["settings"]["aperture_radius_detector"]


from image_grid_maker import (make_row,
                              concatenate_images_tempfile_vertical,
                              create_image_with_text_orientation,
                              concatenate_images_tempfile_row,
                              create_white_image_with_dimensions)


max_num_column=3
font_size_PIL=40
cbar_labelsize=20
cbar_title_fontsize=20
column_title_ratio=0.2

discrete_desired_irradiance = results_classical_e["results_minimize"][-1]["discrete_desired_irradiance"]
smoothed_desired_irradiance = results_ours_e["results_minimize"][-1]["smoothed_desired_irradiance"]
#dit.plotting.quantity2D.plot(smoothed_desired_irradiance,"Smoothed Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
#plt.savefig(results_folder_out + "/smoothed_desired_irradiance.png")
#dit.plotting.quantity2D.plot(discrete_desired_irradiance,"Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
#plt.savefig(results_folder_out + "/discrete_desired_irradiance.png")

vmax = max(discrete_desired_irradiance.max(),smoothed_desired_irradiance.max())
vmin = min(discrete_desired_irradiance.min(),smoothed_desired_irradiance.min())

image_desired = make_row([discrete_desired_irradiance,smoothed_desired_irradiance],[-aperture_radius_detector,aperture_radius_detector,-aperture_radius_detector,aperture_radius_detector],"gray",cbar_labelsize,"[W/mm²]",cbar_title_fontsize,show_x_axis_first=True,vmin=vmin,vmax=vmax)
#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
bottom = concatenate_images_tempfile_row(image_desired)
text1 = create_image_with_text_orientation(image_desired[0],"Desired Irradiance", 0.12,30,vertical=False)
text2 = create_image_with_text_orientation(image_desired[0],"Smoothed Desired Irr.", 0.12,30,vertical=False)
top = concatenate_images_tempfile_row([text1,text2])
out = concatenate_images_tempfile_vertical([top,bottom])
Image.open(out).save(results_folder_out + "/smooth_desired_both0.png")

classical_binned_irradiance = results_classical_e["final_irr_results"]["binned_irradiance"]
classical_smooth_irradiance = results_classical_e["final_irr_results"]["smooth_irradiance"]

ours_binned_irradiance = results_ours_e["final_irr_results"]["binned_irradiance"]
ours_smooth_irradiance = results_ours_e["final_irr_results"]["smooth_irradiance"]

margin_binned_irradiance = results_margin_e["final_irr_results"]["binned_irradiance"]
margin_smooth_irradiance = results_margin_e["final_irr_results"]["smooth_irradiance"]

ours_lens_offset = results_ours_e["lens_offset"]["z"]
classical_lens_offset = results_classical_e["lens_offset"]["z"]
margin_lens_offset = results_margin_e["lens_offset"]["z"]


dit.plotting.quantity2D.plot(ours_binned_irradiance,"ours_binned_irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)

aperture_radius_lens = results_classical_e["settings"]["aperture_radius_lens"]

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

font_multi = 1.3
cbar_labelsize=16
cbar_title_fontsize=20
fontsize=cbar_title_fontsize

irr_classical_row = [classical_binned_irradiance, classical_smooth_irradiance]
irr_ours_row = [ours_binned_irradiance, ours_smooth_irradiance]
irr_margin_row = [margin_binned_irradiance, margin_smooth_irradiance]

irr_vmin = min(irr_classical_row[0].min(), irr_ours_row[0].min(), irr_margin_row[0].min(),
               irr_classical_row[1].min(), irr_ours_row[1].min(), irr_margin_row[1].min())
irr_vmax = max(irr_classical_row[0].max(), irr_ours_row[0].max(), irr_margin_row[0].max(),
               irr_classical_row[1].max(), irr_ours_row[1].max(), irr_margin_row[1].max())
irr_vmax = 0.0015
surf_vmin = min(ours_lens_offset.min(), classical_lens_offset.min(), margin_lens_offset.min())
surf_vmax = max(ours_lens_offset.max(), classical_lens_offset.max(), margin_lens_offset.max())


irr_extent = [
    -aperture_radius_detector, aperture_radius_detector,
    -aperture_radius_detector, aperture_radius_detector
]

surf_extent = [-aperture_radius_lens, aperture_radius_lens,-aperture_radius_lens,aperture_radius_lens]


# Titles corresponding to `irrs`
column_title1 = "Irradiance RC"
column_title2 = "Smoothed Irradiance"
column_title3 = "Surface Profile"

column_titles = [column_title1, column_title2, column_title3]

row_title1 = "Partially Smoothed"
row_title2 = "Fully Smoothed"

row_titles = [row_title1, row_title2]


surf_cmap = "coolwarm"
irr_cmap = "gray"

surf_cmap_title = "[mm]"
irr_cmap_title = "[W/mm²]"

surface1 = make_row([classical_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_vmin,vmax=surf_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface_text = create_image_with_text_orientation(tmp[0],"Surface Profile", column_title_ratio,font_size_PIL,vertical=False)
surface_img = concatenate_images_tempfile_vertical([surface_text,surface1,surface2,surface3])

row1 = make_row(irr_classical_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row2 = make_row(irr_ours_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row3 = make_row(irr_margin_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=irr_vmin,vmax=irr_vmax)
#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
baseline_text = create_image_with_text_orientation(tmp[0],"Partially Smoothed", column_title_ratio,font_size_PIL,vertical=True)
ours_text = create_image_with_text_orientation(tmp[0],"Fully Smoothed", column_title_ratio,font_size_PIL,vertical=True)
margin_text = create_image_with_text_orientation(tmp[0],"Sliding Window", column_title_ratio,font_size_PIL,vertical=True)

row1 = [baseline_text]+row1
row2 = [ours_text]+row2
row3 = [margin_text]+row3
row1 = concatenate_images_tempfile_row(row1)
row2 = concatenate_images_tempfile_row(row2)
row3 = concatenate_images_tempfile_row(row3)

culomn1 = create_image_with_text_orientation(tmp[0],"Irradiance RC", column_title_ratio,font_size_PIL,vertical=False)
culomn2 = create_image_with_text_orientation(tmp[0],"Smoothed Irr.", column_title_ratio,font_size_PIL,vertical=False)
corner = create_white_image_with_dimensions(baseline_text,culomn1)
row0 = concatenate_images_tempfile_row([corner,culomn1,culomn2])

out = concatenate_images_tempfile_vertical([row0,row1,row2,row3])


xout = [out,surface_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plot.png"
image = Image.open(xout)
image.save(file_name_out)
# %%
ours_lens_offset_dx = results_ours_e["lens_offset"]["dz_dx"]
ours_lens_offset_dy = results_ours_e["lens_offset"]["dz_dy"]

classical_lens_offset_dx = results_classical_e["lens_offset"]["dz_dx"]
classical_lens_offset_dy = results_classical_e["lens_offset"]["dz_dy"]

margin_lens_offset_dx = results_margin_e["lens_offset"]["dz_dx"]
margin_lens_offset_dy = results_margin_e["lens_offset"]["dz_dy"]

#%%

dit.plotting.quantity2D.plot(ours_lens_offset_dy.T,"Fully Smoothed Lens Offset Y Diff [1]",[-aperture_radius_lens, aperture_radius_lens],cmap="jet",show=True)

# %%
dit.plotting.quantity2D.plot(ours_lens_offset_dx.T,"Fully Smoothed Lens Offset X Diff [1]",[-aperture_radius_lens, aperture_radius_lens],cmap="jet",show=True)

# %%
from matplotlib.colors import LogNorm
ours_size_grad = np.sqrt(ours_lens_offset_dx**2+ours_lens_offset_dy**2)

dit.plotting.quantity2D.plot(ours_size_grad,"Fully Smoothed Lens Offset Size Grad [1]",[-aperture_radius_lens, aperture_radius_lens],cmap="jet",show=True)
# %%
classical_size_grad = np.sqrt(classical_lens_offset_dx**2+classical_lens_offset_dy**2)
margin_size_grad = np.sqrt(margin_lens_offset_dx**2+margin_lens_offset_dy**2)


surf_grad_cmap_title = "[1]"
surf_grad_cmap = "jet"

surf_grad_vmin = 0
surf_grad_vmax = 0.15#max(np.max(classical_size_grad),np.max(ours_size_grad))

surface1 = make_row([classical_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset]+[ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface_grad_text = create_image_with_text_orientation(tmp[0],"Surface Gradient", column_title_ratio,font_size_PIL,vertical=False)
surface_grad_img = concatenate_images_tempfile_vertical([surface_grad_text,surface1,surface2,surface3])


xout = [out,surface_img,surface_grad_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plotB.png"
image = Image.open(xout)
image.save(file_name_out)
# %%

high_res_irradiance_classical = results_classical_e["high_res_irradiance"]
high_res_irradiance_ours = results_ours_e["high_res_irradiance"]
high_res_irradiance_margin = results_margin_e["high_res_irradiance"]
plt.imshow(high_res_irradiance_ours,cmap="gray")
#%%

"""
High res irradiance final plot

"""


column_title_ratio = 0.3
font_size_PIL = 37

irr_ours_row2 = irr_ours_row+[high_res_irradiance_ours]
irr_classical_row2 = irr_classical_row+[high_res_irradiance_classical]
irr_margin_row2 = irr_margin_row+[high_res_irradiance_margin]

surface1 = make_row([classical_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_vmin,vmax=surf_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row2+[ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface_text = create_image_with_text_orientation(tmp[0],"Surface Profile", column_title_ratio,font_size_PIL,vertical=False)
surface_img = concatenate_images_tempfile_vertical([surface_text,surface1,surface2,surface3])

row1 = make_row(irr_classical_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row2 = make_row(irr_ours_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row3 = make_row(irr_margin_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=irr_vmin,vmax=irr_vmax)


#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
baseline_text = create_image_with_text_orientation(tmp[0],"Partially Smoothed", column_title_ratio,font_size_PIL,vertical=True)
ours_text = create_image_with_text_orientation(tmp[0],"Fully Smoothed", column_title_ratio,font_size_PIL,vertical=True)
margin_text = create_image_with_text_orientation(tmp[0],"Sliding Window", column_title_ratio,font_size_PIL,vertical=True)

row1 = [baseline_text]+row1
row2 = [ours_text]+row2
row3 = [margin_text]+row3
#(²)
row1 = concatenate_images_tempfile_row(row1)
row2 = concatenate_images_tempfile_row(row2)
row3 = concatenate_images_tempfile_row(row3)


culomn1 = create_image_with_text_orientation(tmp[0],"Irradiance RC\n(10M rays, 300² pixels)", column_title_ratio,font_size_PIL,vertical=False)
culomn2 = create_image_with_text_orientation(tmp[0],"Smoothed Irr.\n(1M rays, 300² pixels)", column_title_ratio,font_size_PIL,vertical=False)
culomn3 = create_image_with_text_orientation(tmp[0],"Irradiance RC\n(1B rays, 1200² pixels)", column_title_ratio,font_size_PIL,vertical=False)
corner = create_white_image_with_dimensions(baseline_text,culomn1)

row0 = concatenate_images_tempfile_row([corner,culomn1,culomn2,culomn3])

out = concatenate_images_tempfile_vertical([row0,row1,row2,row3])


xout = [out,surface_img]
xout = concatenate_images_tempfile_row(xout)

file_name_out = results_folder_out + f"/final_plot_highres.png"
image = Image.open(xout)
image.save(file_name_out)
image
# %%

"""
High res irradiance final plot B

"""



surf_grad_cmap_title = "[1]"
surf_grad_cmap = "jet"

surf_grad_vmin = 0
surf_grad_vmax = 0.15#max(np.max(classical_size_grad),np.max(ours_size_grad))

surface1 = make_row([classical_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset]+[ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface_grad_text = create_image_with_text_orientation(tmp[0],"Surface Gradient", column_title_ratio,font_size_PIL,vertical=False)
surface_grad_img = concatenate_images_tempfile_vertical([surface_grad_text,surface1,surface2,surface3])


xout = [out,surface_img,surface_grad_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plot_highresB.png"
image = Image.open(xout)
image.save(file_name_out)
# %%
image
# %%
300*4
# %%

"""
********************* no clamp
High res irradiance final plot

"""

irr_vmax = max(discrete_desired_irradiance.max(),smoothed_desired_irradiance.max(),high_res_irradiance_ours.max(),high_res_irradiance_classical.max(),high_res_irradiance_margin.max())
irr_vmin = 0


column_title_ratio = 0.3
font_size_PIL = 37

irr_ours_row2 = irr_ours_row+[high_res_irradiance_ours]
irr_classical_row2 = irr_classical_row+[high_res_irradiance_classical]
irr_margin_row2 = irr_margin_row+[high_res_irradiance_margin]

surface1 = make_row([classical_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_vmin,vmax=surf_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row2+[ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface_text = create_image_with_text_orientation(tmp[0],"Surface Profile", column_title_ratio,font_size_PIL,vertical=False)
surface_img = concatenate_images_tempfile_vertical([surface_text,surface1,surface2,surface3])

row1 = make_row(irr_classical_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row2 = make_row(irr_ours_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row3 = make_row(irr_margin_row2,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=irr_vmin,vmax=irr_vmax)


#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
baseline_text = create_image_with_text_orientation(tmp[0],"Partially Smoothed", column_title_ratio,font_size_PIL,vertical=True)
ours_text = create_image_with_text_orientation(tmp[0],"Fully Smoothed", column_title_ratio,font_size_PIL,vertical=True)
margin_text = create_image_with_text_orientation(tmp[0],"Sliding Window", column_title_ratio,font_size_PIL,vertical=True)

row1 = [baseline_text]+row1
row2 = [ours_text]+row2
row3 = [margin_text]+row3
#(²)
row1 = concatenate_images_tempfile_row(row1)
row2 = concatenate_images_tempfile_row(row2)
row3 = concatenate_images_tempfile_row(row3)


culomn1 = create_image_with_text_orientation(tmp[0],"Irradiance RC\n(10M rays, 300² pixels)", column_title_ratio,font_size_PIL,vertical=False)
culomn2 = create_image_with_text_orientation(tmp[0],"Smoothed Irr.\n(1M rays, 300² pixels)", column_title_ratio,font_size_PIL,vertical=False)
culomn3 = create_image_with_text_orientation(tmp[0],"Irradiance RC\n(1B rays, 1200² pixels)", column_title_ratio,font_size_PIL,vertical=False)
corner = create_white_image_with_dimensions(baseline_text,culomn1)

row0 = concatenate_images_tempfile_row([corner,culomn1,culomn2,culomn3])

out = concatenate_images_tempfile_vertical([row0,row1,row2,row3])


xout = [out,surface_img]
xout = concatenate_images_tempfile_row(xout)

file_name_out = results_folder_out + f"/final_plot_highrescoclamp.png"
image = Image.open(xout)
image.save(file_name_out)
image
# %%

"""
High res irradiance final plot B no clamp

"""



surf_grad_cmap_title = "[1]"
surf_grad_cmap = "jet"

surf_grad_vmin = 0
surf_grad_vmax = 0.15#max(np.max(classical_size_grad),np.max(ours_size_grad))

surface1 = make_row([classical_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset]+[ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface_grad_text = create_image_with_text_orientation(tmp[0],"Surface Gradient", column_title_ratio,font_size_PIL,vertical=False)
surface_grad_img = concatenate_images_tempfile_vertical([surface_grad_text,surface1,surface2,surface3])


xout = [out,surface_img,surface_grad_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plot_highresBnoclamp.png"
image = Image.open(xout)
image.save(file_name_out)
# %%


"""

no clamp
"""


surface1 = make_row([classical_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_vmin,vmax=surf_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset],surf_extent,surf_cmap,cbar_labelsize,surf_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_vmin,vmax=surf_vmax)
surface_text = create_image_with_text_orientation(tmp[0],"Surface Profile", column_title_ratio,font_size_PIL,vertical=False)
surface_img = concatenate_images_tempfile_vertical([surface_text,surface1,surface2,surface3])

row1 = make_row(irr_classical_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row2 = make_row(irr_ours_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=irr_vmin,vmax=irr_vmax)
row3 = make_row(irr_margin_row,irr_extent,irr_cmap,cbar_labelsize,irr_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=irr_vmin,vmax=irr_vmax)
#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
baseline_text = create_image_with_text_orientation(tmp[0],"Partially Smoothed", column_title_ratio,font_size_PIL,vertical=True)
ours_text = create_image_with_text_orientation(tmp[0],"Fully Smoothed", column_title_ratio,font_size_PIL,vertical=True)
margin_text = create_image_with_text_orientation(tmp[0],"Sliding Window", column_title_ratio,font_size_PIL,vertical=True)

row1 = [baseline_text]+row1
row2 = [ours_text]+row2
row3 = [margin_text]+row3
row1 = concatenate_images_tempfile_row(row1)
row2 = concatenate_images_tempfile_row(row2)
row3 = concatenate_images_tempfile_row(row3)

culomn1 = create_image_with_text_orientation(tmp[0],"Irradiance RC", column_title_ratio,font_size_PIL,vertical=False)
culomn2 = create_image_with_text_orientation(tmp[0],"Smoothed Irr.", column_title_ratio,font_size_PIL,vertical=False)
corner = create_white_image_with_dimensions(baseline_text,culomn1)
row0 = concatenate_images_tempfile_row([corner,culomn1,culomn2])

out = concatenate_images_tempfile_vertical([row0,row1,row2,row3])


xout = [out,surface_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plot_noclamp.png"
image = Image.open(xout)
image.save(file_name_out)


irr_vmax = max(discrete_desired_irradiance.max(),smoothed_desired_irradiance.max(),ours_binned_irradiance.max(),classical_binned_irradiance.max(),margin_binned_irradiance.max())

surface1 = make_row([classical_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface1 = surface1[0]
surface2 = make_row([ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface2 = surface2[0]
surface3 = make_row([margin_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=True,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface3 = surface3[0]
tmp = make_row(irr_ours_row+[ours_lens_offset]+[ours_size_grad],surf_extent,surf_grad_cmap,cbar_labelsize,surf_grad_cmap_title,cbar_title_fontsize,show_x_axis_first=False,vmin=surf_grad_vmin,vmax=surf_grad_vmax)
surface_grad_text = create_image_with_text_orientation(tmp[0],"Surface Gradient", column_title_ratio,font_size_PIL,vertical=False)
surface_grad_img = concatenate_images_tempfile_vertical([surface_grad_text,surface1,surface2,surface3])


xout = [out,surface_img,surface_grad_img]
xout = concatenate_images_tempfile_row(xout)

#file_name_out = results_folder_main + f"/final_plot2.png"
file_name_out = results_folder_out + f"/final_plotBnoclamp.png"
image = Image.open(xout)
image.save(file_name_out)
# %%
