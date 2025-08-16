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
results_folder_in = "results/server_lens_picture/"#.2"
results_folder_out = "results/server_lens_picture_plots/"
create_folder(results_folder_out)


# %%

import torch
all_main_subfolders = ["results_classical","results_desired_irr_smoothing"]

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
results_ours_e.keys()


subtitle_classical_short_e = "Partially Smoothed"
subtitle_ours_short_e = "Ours"
    
columns_title = [subtitle_classical_short_e,subtitle_ours_short_e]
all_results = [results_classical_e,results_ours_e]

title_all = {}
title_all["Irradiance MC showcase"] = "Irradiance RC"
title_all["Irr. Smooth"] = f"Smoothed Irr."
title_all["Relative Surface Profile"] = f"Surface Profile"
    
keys1 = ["Irradiance MC showcase","Irr. Smooth", "Relative Surface Profile"]


# %%
def create_2d4x4_plots():
    results_classical = load_results(0)
    results_desired_irr_smoothing = load_results(1)
    
    
    subtitle_classical = "Method: Classical Algorithmic Differentiable Ray Tracing"
    subtitle_desired_irr_smoothing = "Method: Desired Irradiance Smoothing"
    

    subtitle_classical_short = "Method: CADRT"
    subtitle_desired_irr_smoothing_short = "Method: DIS"
    

    all_subtitles = [subtitle_classical,subtitle_desired_irr_smoothing]
    all_subtitles_short = [subtitle_classical_short,subtitle_desired_irr_smoothing_short]
    all_results = [results_classical,results_desired_irr_smoothing]

    def get_kwards_from_index(index):
        if index > len(all_subtitles):
            raise ValueError("index must be > len(all_subtitles)")
        subtitle = all_subtitles[index]
        subtitle_short = all_subtitles_short[index]
        result = all_results[index]
        return result,subtitle,subtitle_short

    def create_convergence_plot_res(title,quantity_key,y_label, file_name_out):
        ax = plt.gca()  # Slightly wider for space
        

        refine_iters_all = []
        refine_fun_all = []
        point_style_all = []
        labels_all = []

        for k in [1,0]:
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
            label_prefix = "Ours"
            point_style = "o"
            if k % 2 == 0:
                label_prefix="Partially Smoothed"
                point_style = "o"
            
            all_iters = np.concatenate(all_iters)
            all_fun_vals = np.concatenate(all_fun_vals)
            ax.plot(all_iters, all_fun_vals,linestyle="-", label=f"{label_prefix}")
            labels_all.append("Refinements ("+label_prefix+")")
            point_style_all.append(point_style)
            refine_iters_all += [refine_iter]
            refine_fun_all += [refine_fun]
            

        for k in range(2):
            color = "red"
            if k % 2 == 0:
                color = "black"
            
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
        ax = plt.gca()  # Slightly wider for space
        refine_iters_all = []
        refine_fun_all = []
        point_style_all = []
        labels_all = []
        
        for k in [1,0]:
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
                
            
            label_prefix = "Ours"
            point_style = "o"
            if k % 2 == 0:
                label_prefix="Partially Smoothed"
                point_style = "o"
            
            all_iters = np.concatenate(all_iters)
            all_fun_vals = np.concatenate(all_fun_vals)
            ax.plot(all_iters, all_fun_vals,linestyle="-", label=f"{label_prefix}")
                #ax.axvline(x=current_iter, color='gray', linestyle='--', linewidth=1.2)
            labels_all.append("Refinements ("+label_prefix+")")
            point_style_all.append(point_style)
            refine_iters_all += [refine_iter]
            refine_fun_all += [refine_fun]
            

        for k in range(2):
            color = "red"
            if k % 2 == 0:
                color = "black"
            
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
    create_convergence_plot_default("Convergence of Error","convergence","Error", file_name_out_error)

create_2d4x4_plots()
# %%
results_classical_e["results_minimize"][-1].keys()
#%%
aperture_radius_detector = results_classical_e["settings"]["aperture_radius_detector"]
#%%
discrete_desired_irradiance = results_classical_e["results_minimize"][-1]["discrete_desired_irradiance"]
smoothed_desired_irradiance = results_ours_e["results_minimize"][-1]["smoothed_desired_irradiance"]
dit.plotting.quantity2D.plot(smoothed_desired_irradiance,"Smoothed Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
plt.savefig(results_folder_out + "/smoothed_desired_irradiance.png")


dit.plotting.quantity2D.plot(discrete_desired_irradiance,"Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
plt.savefig(results_folder_out + "/discrete_desired_irradiance.png")

# %%
