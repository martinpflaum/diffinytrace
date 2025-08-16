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
image_file_name = "image_vertical.png"
results_folder_main = "results/results_NV1"#.2"
create_folder(results_folder_main)


import torch
all_main_subfolders = ["results_classical","results_desired_irr_smoothing","results_sigma_refinement"]
#,"results_power_correction_pure","results_power_correction_R"]

def get_folder(idx,etendue):
    out = results_folder_main+"/"
    main_subfolder = all_main_subfolders[idx]
    out += main_subfolder
    if etendue:
        out += "/etendue"
    else:
        out += "/no_etendue"
    return out
def load_results(idx,etendue):
    folder = get_folder(idx,etendue)
    results = load_data(folder+"/results_dict.pkl")
    return results


#etendue = True,results_index = 3
def create_all_plots_from_run(etendue,results_index):

    results_classical = load_results(0,etendue)
    results_desired_irr_smoothing = load_results(1,etendue)
    
    #results_power_correction_pure = load_results(2,etendue)
    #results_power_correction_R = load_results(3,etendue)

    subtitle_classical = "Partially Smoothed"
    subtitle_desired_irr_smoothing = "Ours"
    
    #subtitle_power_correction_pure = "Method: Power correction"
    #subtitle_power_correction_R = "Method: Power correction with Refinement"


    subtitle_classical_short = "Method: CADRT"
    subtitle_desired_irr_smoothing_short = "Method: DIS"
    #subtitle_power_correction_pure_short = "Method: DISPC"
    #subtitle_power_correction_R_short = "Method: DISRPC"


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

    result,subtitle,subtitle_short = get_kwards_from_index(results_index)
    
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_convergence_merit_function(result,subtitle):
        results_minimize = result["results_minimize"]
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        
        fig, ax = plt.subplots()
        current_iter = 0
        for k in range(len(results_minimize)):
            history = results_minimize[k]["history"]
            fun_vals = history["fun_vals"]
            fun_grads_norm = history["fun_grads_norm"]
            fun_vals = np.array(fun_vals)
            fun_grads_norm = np.array(fun_grads_norm)
            iters = np.arange(len(fun_vals))+current_iter
            current_iter = iters[-1]
            sigma_val = results_minimize[k]["sigma"]
            coeff_shape = results_minimize[k]["coeff_shape"]
            ax.plot(iters,fun_vals,label=f"n={coeff_shape[0]}")
            ax.axvline(x=current_iter, color='gray', linestyle='--', linewidth=1.2)


        #ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()  # Enable minor ticks
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='--', linewidth=1)  # Minor grid lines (finer)
        fig.suptitle("Convergence of Merit Function $m$")
        ax.set_title(subtitle)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(r"$m$")
        ax.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.tight_layout()
        #plt.show()

    def plot_convergence_error_function(result,subtitle):
        results_minimize = result["results_minimize"]
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        
        fig, ax = plt.subplots()
        
        current_iter = 0
        for k in range(len(results_minimize)):
            history = results_minimize[k]["history"]
            fun_vals = history["convergence"]
            fun_vals = np.array(fun_vals)
            iters = np.arange(len(fun_vals))+current_iter
            current_iter = iters[-1]
            sigma_val = results_minimize[k]["sigma"]
            coeff_shape = results_minimize[k]["coeff_shape"]
            plt.plot(iters,fun_vals,label=f"n={coeff_shape[0]}")
            plt.axvline(x=current_iter, color='gray', linestyle='--', linewidth=1.2)

        # Adjust the grid to have a finer resolution by changing the major and minor ticks
        ax.minorticks_on()  # Enable minor ticks
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='--', linewidth=1)  # Minor grid lines (finer)
        
        
        fig.suptitle("Convergence of Error Per Iteration")
        ax.set_title(subtitle)
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Error")
        ax.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.tight_layout()
        #plt.show()

    plot_convergence_merit_function(result,subtitle)
    current_result_folder = get_folder(results_index,etendue)
    plt.savefig(current_result_folder+"/merit_plot.png",format='png')

    plot_convergence_error_function(result,subtitle)
    current_result_folder = get_folder(results_index,etendue)
    plt.savefig(current_result_folder+"/error_plot.png",format='png')


    results_minimize = result["results_minimize"]
        
    def get_surface_data(lens,resolution):
        aperture_radius = lens.aperture_radius
        surface = lens.surface2.surface

        x_range = (-aperture_radius,aperture_radius)
        y_range = (-aperture_radius,aperture_radius)
        _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
        _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
        O = torch.zeros((x.shape[0],2))        
            
        O[:,0] = x
        O[:,1] = y
        z = None
            
        with torch.no_grad():
            z = surface.explicit(O)
            
        if not lens.is_square:
            z[O[:,[0,1]].norm(dim=-1)>aperture_radius] = float("nan")

        z = z.detach().reshape(resolution,resolution).numpy()
        z = z.T
        return z

    lens_resolution = 512
    results_minimize = result["results_minimize"]
    lens_heights = [get_surface_data(elem["lens"],lens_resolution) for elem in results_minimize]    
    lens_heights.append(get_surface_data(result["final_lens"],lens_resolution))

    def get_quantity_default(name):
        out = [elem[name] for elem in results_minimize]
        #out.append(result["final_irr_results"][name])
        return out
        
    binned_irradiances = get_quantity_default("binned_irradiance")
    smooth_irradiances = get_quantity_default("smooth_irradiance")
    binned_irradiance_eval = get_quantity_default("binned_irradiance_eval")
    desired_none_smooth_irradiance_eval = get_quantity_default("desired_none_smooth_irradiance_eval")
    desired_none_smooth_irradiance_opti = get_quantity_default("desired_none_smooth_irradiance_opti")
    desired_smooth_irradiance = get_quantity_default("desired_smooth_irradiance")


    def get_residuals_MC():
        out = []
        for k in range(len(binned_irradiance_eval)):
            out.append((binned_irradiance_eval[k]-desired_none_smooth_irradiance_eval[k])**2)  
        return out

    def get_residuals_smooth():
        out = []
        if not result["settings"]["use_desired_irradiance_smoothing"]:
            for k in range(len(binned_irradiance_eval)):
                out.append((smooth_irradiances[k]-desired_none_smooth_irradiance_opti[k])**2)  
        else:
            for k in range(len(binned_irradiance_eval)):
                out.append((smooth_irradiances[k]-desired_smooth_irradiance[k])**2)  
        return out


    aperture_radius_lens = result["settings"]['aperture_radius_lens']
    aperture_radius_detector = result["settings"]['aperture_radius_detector']
    def get_extent(elem):
        return [-elem,elem,-elem,elem]

    def create_irradiance_dict(title,data):
        return {"vidx":"irr","title":title,"extent":get_extent(aperture_radius_detector),"data":data,"cmap":"grey","cbar_title":"[W/mm²]"}
    data_all = {}
    #data_all["Relative Surface Profile"] = {"vidx":"13018913","title":"Surface Profile","extent":get_extent(aperture_radius_lens),"data":lens_heights,"cmap":"coolwarm","cbar_title":"[mm]"}
    data_all["Irradiance MC showcase"] = create_irradiance_dict("Irradiance RC",binned_irradiances)
    data_all["Irr. Smooth opti"] = create_irradiance_dict("Irr. Smooth",smooth_irradiances)
    #data_all["Smoothed Desired Irr."] = create_irradiance_dict("Smoothed desired Irr.",desired_smooth_irradiance)
    #data_all["Desired Irr."] = create_irradiance_dict("Desired Irr.",desired_none_smooth_irradiance_opti)
    #data_all["Residual Squared Eval."] = {"vidx":"2424526","title":"Residual² Eval.","data":get_residuals_MC(),"extent":get_extent(aperture_radius_detector),"cmap":"jet","cbar_title":"[W²/mm⁴]"}
    #data_all["Residual Squared Opti."] = {"vidx":"8325923","title":"Residual² Opti.","data":get_residuals_smooth(),"extent":get_extent(aperture_radius_detector),"cmap":"jet","cbar_title":"[W²/mm⁴]"}

    current_result_folder = get_folder(results_index,etendue)
    def run_data_grid(data_all,keys,prefix_filename):
        def process_short(quant_key):
            return [data_all[key][quant_key]for key in keys]
        image_grid = process_short("data")
        rows_cmap = process_short("cmap")
        rows_title = process_short("title")
        rows_extent = process_short("extent")
        cbar_titles = process_short("cbar_title")
        rows_vidx = process_short("vidx")

        columns_title = [f'n={elem["coeff_shape"][0]}' for elem in results_minimize]#+["Final Result"]
        
        kwargs = dict(image_grid=image_grid,
                rows_extent=rows_extent,
                rows_vidx=rows_vidx,
                rows_cmap=rows_cmap,
                rows_title=rows_title,
                cbar_titles=cbar_titles,
                columns_title=columns_title)
        
        out = image_from_grid(**kwargs,
                                max_num_column=5,
                            font_size_PIL=40,
                            cbar_labelsize=20,
                            cbar_title_fontsize=20)
        
        file_name_getter = lambda i: current_result_folder+"/"+prefix_filename+f"_{i}.png"
        for k in range(len(out)):
            if k% 2 ==1:
                concatenate_images_vertical([out[k-1],out[k]],file_name_getter(k))
            elif k == len(out)-1:
                concatenate_images_vertical([out[k]],file_name_getter(k))
    keys1 = ["Irradiance MC showcase","Irr. Smooth opti"]
    run_data_grid(data_all,keys1,"page1")
    
    #keys2 = None
    #if result["settings"]["use_desired_irradiance_smoothing"]:
    #    keys2 = ["Irr. Smooth opti","Smoothed Desired Irr.","Residual Squared Opti."]
    #else:
    #    keys2 = ["Irr. Smooth opti","Desired Irr.","Residual Squared Opti."]
    #run_data_grid(data_all,keys2,"page2")


def add_title_to_plot_with_auto_height(image_path, title, output_path, font_size=20):
    """
    Adds a title on top of an existing plot image, automatically estimating the required whitespace height.

    Parameters:
    - image_path: Path to the input image file.
    - title: The title text to be added.
    - output_path: Path to save the resulting image.
    - font_size: Size of the font for the title.

    Returns:
    - Saves the modified image with the title.
    """
    # Open the original image
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size

    # Prepare the font (using Arial)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception as e:
        print(f"Error loading font: {e}. Using default font.")
        font = ImageFont.load_default()

    # Estimate the whitespace height based on font size and title text
    text_bbox = font.getbbox(title)
    text_width, text_height = text_bbox[2], text_bbox[3]  # Get width and height
    whitespace_height = int(1.5 * text_height)  # Add some padding above and below the text

    # Create a new image with extra whitespace on top
    new_height = original_height + whitespace_height
    new_image = Image.new("RGB", (original_width, new_height), "white")

    # Paste the original image onto the new image
    new_image.paste(original_image, (0, whitespace_height))

    # Draw the title on the new image
    draw = ImageDraw.Draw(new_image)
    text_position = ((original_width - text_width) // 2, (whitespace_height - text_height) // 2)
    draw.text(text_position, title, fill="black", font=font)

    # Save the result
    new_image.save(output_path)
    print(f"Title added and saved to {output_path}")

# Example usage (uncomment and adjust paths as needed):
# add_title_to_plot_with_auto_height("plot.png", "Example Plot Title", "plot_with_auto_title.png", font_size=30)
#%%


etendue = True
results_classical_e = load_results(0,etendue)
results_ours_e = load_results(1,etendue)
    
subtitle_classical_short_e = "Partially Smoothed"
subtitle_ours_short_e = "Ours"
    
columns_title = [subtitle_classical_short_e,subtitle_ours_short_e]
all_results = [results_classical_e,results_ours_e]

title_all = {}
title_all["Irradiance MC showcase"] = "Irradiance RC"
title_all["Irr. Smooth"] = f"Smoothed Irr."
title_all["Relative Surface Profile"] = f"Surface Profile"
    
keys1 = ["Irradiance MC showcase","Irr. Smooth", "Relative Surface Profile"]

def get_surface_data(lens,resolution):
    aperture_radius = lens.aperture_radius
    surface = lens.surface2.surface

    _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
    _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
    mesh = torch.meshgrid(_x,_y)
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    O = torch.zeros((x.shape[0],2))        
            
    O[:,0] = x
    O[:,1] = y
    z = None
            
    with torch.no_grad():
        z = surface.explicit(O)
        
    if not lens.is_square:
        z[O[:,[0,1]].norm(dim=-1)>aperture_radius] = float("nan")

    z = z.detach().reshape(resolution,resolution).numpy()
    z = z.T
    return z


def extract_data_for_key(result, key):
    if key == "Relative Surface Profile":
        lens = result["results_minimize"][-1]["lens"]
        return get_surface_data(lens, 512)
    elif key == "Irradiance MC showcase":
        return result["final_irr_results"]["binned_irradiance"]
    elif key == "Desired Irr.":
        return result["final_irr_results"]["desired_none_smooth_irradiance_opti"]
    elif key == "Residual Squared Eval.":
        return (result["final_irr_results"]["binned_irradiance_eval"] - 
        result["final_irr_results"]["desired_none_smooth_irradiance_eval"]) ** 2
    elif key== "Irr. Smooth":
        return result["final_irr_results"]["smooth_irradiance"]
    else:
        raise ValueError(f"Unknown key: {key}")

data_grid = {key: [extract_data_for_key(result, key) for result in all_results] for key in keys1}
rows_vidx = ["irr","irr","surface"]
rows_title = keys1
aperture_radius_lens =results_classical_e["settings"]['aperture_radius_lens']
aperture_radius_detector = results_classical_e["settings"]['aperture_radius_detector']
def get_extent(elem):
    return [-elem,elem,-elem,elem]

rows_extent = [get_extent(aperture_radius_detector),get_extent(aperture_radius_detector),get_extent(aperture_radius_lens)]
rows_cmap = ["gray", "gray", "coolwarm"]  # Color maps for each row
cbar_titles = ["[W/mm²]", "[W/mm²]", "[mm]"]  # Colorbar titles



# %%

keys1
#%%
rows_vmin_idx_dict = {}
rows_vmax_idx_dict = {}

image_grid=[data_grid[key] for key in keys1]
from image_grid_maker import get_row_vmin_vmax
    
for i,row in enumerate(image_grid):
    idx = rows_vidx[i]
    vmin,vmax = get_row_vmin_vmax(row)
    if not idx in rows_vmin_idx_dict.keys():
        rows_vmin_idx_dict[idx] = vmin 
    else:
        rows_vmin_idx_dict[idx] = min(vmin,rows_vmin_idx_dict[idx]) 
        
    if not idx in rows_vmax_idx_dict.keys():
        rows_vmax_idx_dict[idx] = vmax 
    else:
        rows_vmax_idx_dict[idx] = max(vmax,rows_vmax_idx_dict[idx]) 
           

rows_vmin = []
rows_vmax = []

for i,row in enumerate(image_grid):
    idx = rows_vidx[i]
    vmin = rows_vmin_idx_dict[idx]
    vmax = rows_vmax_idx_dict[idx]
    rows_vmin.append(vmin)
    rows_vmax.append(vmax)
rows_vmax
#%%
max_num_column=len(columns_title)
font_size_PIL=40
cbar_labelsize=20
cbar_title_fontsize=20
column_title_ratio=0.2

#%%
from image_grid_maker import make_row,concatenate_images_tempfile_vertical,create_image_with_text_orientation
i=2
matrices_row = image_grid[i]
#baseline

#title_all["Irradiance MC showcase"] = "Irradiance RC"
#title_all["Irr. Smooth"] = f"Irr. Smooth"
#title_all["Relative Surface Profile"] = f"Surface Profile"
    

surface1 = make_row([matrices_row[0]],rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
surface1 = surface1[0]
surface2 = make_row([matrices_row[1]],rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=True,vmin=rows_vmin[i],vmax=rows_vmax[i])
surface2 = surface2[0]
tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
surface_text = create_image_with_text_orientation(tmp[0],"Surface Profile", column_title_ratio,font_size_PIL,vertical=False)
surface_img = concatenate_images_tempfile_vertical([surface_text,surface1,surface2])

#create_white_image_with_dimensions2(wdith_plus_img,width_minus_img, image2_path)
#create_image_with_text_orientation(row_files[1], rows_title[i], column_title_ratio,font_size_PIL,vertical=True)


#%%
surface_img_pil = Image.open(surface_img)
surface_img_pil.save(results_folder_main + "/CCsurface_img_PIL0.png")

#%%
from image_grid_maker import concatenate_images_tempfile_row,create_white_image_with_dimensions
#baseline

#title_all["Irradiance MC showcase"] = "Irradiance RC"
#title_all["Irr. Smooth"] = f"Irr. Smooth"
#title_all["Relative Surface Profile"] = f"Surface Profile"
i=0

row1 = make_row([image_grid[0][0],image_grid[1][0]],rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
row2 = make_row([image_grid[0][1],image_grid[1][1]],rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=True,vmin=rows_vmin[i],vmax=rows_vmax[i])
#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
baseline_text = create_image_with_text_orientation(tmp[0],"Partially Smoothed", column_title_ratio,font_size_PIL,vertical=True)
ours_text = create_image_with_text_orientation(tmp[0],"Ours", column_title_ratio,font_size_PIL,vertical=True)

row1 = [baseline_text]+row1
row2 = [ours_text]+row2
row1 = concatenate_images_tempfile_row(row1)
row2 = concatenate_images_tempfile_row(row2)

culomn1 = create_image_with_text_orientation(tmp[0],"Irradiance RC", column_title_ratio,font_size_PIL,vertical=False)
culomn2 = create_image_with_text_orientation(tmp[0],"Smoothed Irr.", column_title_ratio,font_size_PIL,vertical=False)
corner = create_white_image_with_dimensions(baseline_text,culomn1)
row0 = concatenate_images_tempfile_row([corner,culomn1,culomn2])

out = concatenate_images_tempfile_vertical([row0,row1,row2])

#%%
xout = [out,surface_img]
xout = concatenate_images_tempfile_row(xout)

file_name_out = results_folder_main + f"/final_plot2.png"
image = Image.open(xout)
image.save(file_name_out)
#%%

def create_2d4x4_plots(etendue):
    results_classical = load_results(0,etendue)
    results_desired_irr_smoothing = load_results(1,etendue)
    
    #results_power_correction_pure = load_results(2,etendue)
    #results_power_correction_R = load_results(3,etendue)

    subtitle_classical = "Method: Classical Algorithmic Differentiable Ray Tracing"
    subtitle_desired_irr_smoothing = "Method: Desired Irradiance Smoothing"
    
    #subtitle_power_correction_pure = "Method: Power correction"
    #subtitle_power_correction_R = "Method: Power correction with Refinement"


    subtitle_classical_short = "Method: CADRT"
    subtitle_desired_irr_smoothing_short = "Method: DIS"
    #subtitle_power_correction_pure_short = "Method: DISPC"
    #subtitle_power_correction_R_short = "Method: DISRPC"


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
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='--', linewidth=1)  # Minor grid lines (finer)
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
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5)  # Minor grid lines (finer)
        ax.grid(True, which='major', linestyle='--', linewidth=1)  # Minor grid lines (finer)
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


    etendue_add_on = "with_etendue"
    if not etendue:
        etendue_add_on = "without_etendue"

    _etendue_add_on = " (with étendue)"
    if not etendue:
        _etendue_add_on = " (without étendue)"
    _etendue_add_on = ""
        
    file_name_out_merit = results_folder_main + f"/convergence_merit_fun_{etendue_add_on}.png"
    create_convergence_plot_res("Convergence of Merit Function"+_etendue_add_on,"fun_vals","$m$", file_name_out_merit)

    file_name_out_error = results_folder_main + f"/convergence_error_{etendue_add_on}.png"
    create_convergence_plot_default("Convergence of Error "+_etendue_add_on,"convergence","Error", file_name_out_error)

    file_name_out = results_folder_main + f"/power_irr_smooth_{etendue_add_on}.png"
    create_convergence_plot_default("Integrated Power of Smoothed Irradiance"+_etendue_add_on,"power_irr_smooth_list","Power [W]", file_name_out)

    file_name_out = results_folder_main + f"/power_desired_irr_smooth_{etendue_add_on}.png"
    create_convergence_plot_default("Integrated Power of (Smoothed) Desired Irradiance"+_etendue_add_on,"power_desired_irr_list","Power [W]", file_name_out)
#%%
create_2d4x4_plots(True)
#create_2d4x4_plots(False)
#%%

etendue = True
results_classical = load_results(0,etendue)
results_desired_irr_smoothing = load_results(1,etendue)
#%%
settings = results_desired_irr_smoothing["settings"]

aperture_radius_detector = settings['aperture_radius_detector']
    
desired_none_smooth_irradiance_opti = results_desired_irr_smoothing["results_minimize"][-1]["desired_none_smooth_irradiance_opti"]
desired_smooth_irradiance = results_classical["results_minimize"][-1]["desired_smooth_irradiance"]
dit.plotting.quantity2D.plot(desired_smooth_irradiance,"Smoothed Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
plt.savefig(results_folder_main + "/CCdesired_smooth_irradiance0.png")


dit.plotting.quantity2D.plot(desired_none_smooth_irradiance_opti,"Desired Irradiance [W/mm²]",[-aperture_radius_detector,aperture_radius_detector],cmap="gray",show=False)
plt.savefig(results_folder_main + "/CCdesired_none_smooth_irradiance0.png")
#%%
font_size_PIL=30
cbar_labelsize=16
cbar_title_fontsize=16
column_title_ratio=0.12

from image_grid_maker import make_row,concatenate_images_tempfile_vertical,create_image_with_text_orientation,concatenate_images_tempfile_row

vmax = max(desired_none_smooth_irradiance_opti.max(),desired_smooth_irradiance.max())
vmin = min(desired_none_smooth_irradiance_opti.min(),desired_smooth_irradiance.min())

#%%
image_desired = make_row([desired_none_smooth_irradiance_opti,desired_smooth_irradiance],[-aperture_radius_detector,aperture_radius_detector,-aperture_radius_detector,aperture_radius_detector],"gray",cbar_labelsize,"[W/mm²]",cbar_title_fontsize,show_x_axis_first=True,vmin=vmin,vmax=vmax)
#tmp = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=False,vmin=rows_vmin[i],vmax=rows_vmax[i])
bottom = concatenate_images_tempfile_row(image_desired)
Image.open(bottom).save(results_folder_main + "/CCtmp0.png")

#%%

text1 = create_image_with_text_orientation(image_desired[0],"Desired Irradiance", column_title_ratio,font_size_PIL,vertical=False)
text2 = create_image_with_text_orientation(image_desired[0],"Smoothed Desired Irr.", column_title_ratio,font_size_PIL,vertical=False)
top = concatenate_images_tempfile_row([text1,text2])
#create_white_image_with_dimensions2(wdith_plus_img,width_minus_img, image2_path)
out = concatenate_images_tempfile_vertical([top,bottom])
Image.open(out).save(results_folder_main + "/CC_smooth_desired_both0.png")


#create_white_image_with_dimensions2(wdith_plus_img,width_minus_img, image2_path)
#create_image_with_text_orientation(row_files[1], rows_title[i], column_title_ratio,font_size_PIL,vertical=True)


#%%
Image.open(surface_img).save(results_folder_main + "/CCsurface_img1.png")


#%%

#run_data_grid_final_results()
#run_data_grid_final_results(False)
#%%
etendue = True
for results_index in range(2):
    create_all_plots_from_run(etendue,results_index)
#%%
#etendue = False
#for results_index in range(2):
#    create_all_plots_from_run(etendue,results_index)

#%%
results_desired_irr_smoothing = load_results(1,True)
results_desired_irr_smoothing["settings"]

# %%
etendue = True
results_desired_irr_smoothing = load_results(1,etendue)
lens_saved = results_desired_irr_smoothing["final_lens"]
settings = results_desired_irr_smoothing["settings"]
aperture_radius_source = settings["aperture_radius_source"]
aperture_radius_source = settings["aperture_radius_source"]
image_padding = settings["image_padding"]
detector_distance = settings["detector_distance"]
lens_distance = settings["lens_distance"]
lens_thickness = settings["lens_thickness"]
aperture_radius_lens = settings["aperture_radius_lens"]
air_material=dit.materials["NONE"]
detector_distance
#%%       
total_power = 1.0
light_transform = dit.transforms.Offset(torch.tensor([0.0,0.0,0.0]))
light_transform.pos.requires_grad = False

light_source = None
if etendue:
    light_source = dit.source.VisibleSunlightSimpleMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power)
else:
    light_source = dit.source.CollimatedMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power,is_square=True)


aperture_radius_detector = aperture_radius_source*(1+image_padding)
aperture_radius_detector
#%%

"""lens_transform = dit.transforms.Distance(lens_distance,parent_transform=light_transform)

bspline_surface1 = dit.Bspline(aperture_radius_lens,lens_saved.surface2.surface.orders,lens_saved.surface2.surface.ns)
bspline_surface1.coeff = lens_saved.surface2.surface.coeff
surface1 = dit.Plane()
lens1 = dit.Lens(lens_transform,lens_thickness,surface1,bspline_surface1,dit.materials["PMMA"],aperture_radius_lens,is_square=True)
lens_transform.distance.requires_grad = False
lens1.lens_thickness.requires_grad = False
            
detector_transform = dit.transforms.Distance(detector_distance,parent_transform=lens1)#25.0+0.5
detector_transform.distance.requires_grad = False
plane_surface = dit.Plane()
detector = dit.Detector(detector_transform,plane_surface,aperture_radius_detector)
system = dit.SequentialOpticalSystem({"source":light_source,"lens":lens1,"detector":detector},n_func_enviroment=air_material)
sequence = ["source","lens","detector"]
lens1.n_func = dit.materials["PMMA"]
resolution = 64
dit.plotting.system3D.plot(lens1,resolution=resolution,html_file_name=results_folder_main+"/ilustrations/something.html")

binned_irradiance = results_desired_irr_smoothing["final_irr_results"]["binned_irradiance"]

data = []

resolution = 32
data += dit.plotting.system3D._plot_surface_recursively(lens1,"",resolution)
data += dit.plotting.system3D._plot_surface_recursively(light_source,"",resolution)

xr,_ = light_source.sample(15)
O,D,wave_len,_,RayPaths = system(xr,sequence)

if not RayPaths is None:
    if isinstance(RayPaths,dict):
        rays = RayPaths["ray_paths"]
        
show_grid=True
xlabel="x [mm]"
ylabel="y [mm]"
zlabel="z [mm]"
xticks=None
yticks=None
zticks=None
axislabel_font_size=10
tick_font_size=10
ray_color="#9673A6"
ray_linewidth=3.
         
layout = dit.plotting.system3D.get_optical_system_layout(False,xlabel,ylabel,zlabel,xticks,yticks,zticks,axislabel_font_size,tick_font_size)

data += dit.plotting.system3D.ray_paths(rays,ray_color,ray_linewidth)

import plotly.graph_objects as go


num_bins = binned_irradiance.shape[0]
aperture_half = aperture_radius_detector
x = np.linspace(-aperture_half, aperture_half, num_bins)  # Width
y = np.linspace(-aperture_half, aperture_half, num_bins)  # Height
z = torch.ones((num_bins, num_bins))*detector.get_transformation_matrix()[2,3]        # Flat surface

gosurface2 = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=binned_irradiance,
        #cmin=0,
        #cmax=1.,
        colorscale="gray",
        showscale=False,
        showlegend=False
)

data += [gosurface2]
fig = go.Figure(data=data,layout=layout)
fig.write_html(results_folder_main+"/ilustrations/magical_lens.html")
fig.show()

binned_irradiance.max()

etendue = True
results_desired_irr_smoothing = load_results(1,etendue)
lens_saved = results_desired_irr_smoothing["final_lens"]
settings = results_desired_irr_smoothing["settings"]
aperture_radius_source = settings["aperture_radius_source"]
aperture_radius_source = settings["aperture_radius_source"]
image_padding = settings["image_padding"]
detector_distance = settings["detector_distance"]
lens_distance = settings["lens_distance"]
lens_thickness = settings["lens_thickness"]
aperture_radius_lens = settings["aperture_radius_lens"]
air_material=dit.materials["NONE"]

total_power = 1.0
light_transform = dit.transforms.Offset(torch.tensor([0.0,0.0,0.0]))
light_transform.pos.requires_grad = False

light_source = None
if etendue:
    light_source = dit.source.VisibleSunlightSimpleMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power)
else:
    light_source = dit.source.CollimatedMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power,is_square=True)


aperture_radius_detector = aperture_radius_source*(1+image_padding)
lens_transform = dit.transforms.Distance(lens_distance,parent_transform=light_transform)

bspline_surface1 = dit.Bspline(aperture_radius_lens,lens_saved.surface2.surface.orders,lens_saved.surface2.surface.ns)
surface1 = dit.Plane()
lens1 = dit.Lens(lens_transform,lens_thickness,surface1,bspline_surface1,dit.materials["PMMA"],aperture_radius_lens,is_square=True)
            
detector_transform = dit.transforms.Distance(detector_distance,parent_transform=lens1)#25.0+0.5
detector_transform.distance.requires_grad = False
plane_surface = dit.Plane()
detector = dit.Detector(detector_transform,plane_surface,aperture_radius_detector)
system = dit.SequentialOpticalSystem({"source":light_source,"lens":lens1,"detector":detector},n_func_enviroment=air_material)
sequence = ["source","lens","detector"]

xr,_ = light_source.sample(15)
O,D,wave_len,_,RayPaths = system(xr,sequence)

dit.plotting.system2D.plot(system,RayPaths)

import tempfile
import os
import cv2
import matplotlib.pyplot as plt
import torch

temp_dir= tempfile.TemporaryDirectory()


def get_surface_data(lens,resolution):
        aperture_radius = lens.aperture_radius
        surface = lens.surface2.surface

        x_range = (-aperture_radius,aperture_radius)
        y_range = (-aperture_radius,aperture_radius)
        _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
        _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
        O = torch.zeros((x.shape[0],2))        
            
        O[:,0] = x
        O[:,1] = y
        z = None
            
        with torch.no_grad():
            z = surface.explicit(O)
            
        if not lens.is_square:
            z[O[:,[0,1]].norm(dim=-1)>aperture_radius] = float("nan")

        z = z.detach().reshape(resolution,resolution).numpy()
        z = z.T
        return z

result = load_results(1,False)
lens_resolution = 512
results_minimize = result["results_minimize"]

settings = result["settings"]
aperture_radius_source = settings["aperture_radius_source"]
aperture_radius_source = settings["aperture_radius_source"]
image_padding = settings["image_padding"]
detector_distance = settings["detector_distance"]
lens_distance = settings["lens_distance"]
lens_thickness = settings["lens_thickness"]
aperture_radius_lens = settings["aperture_radius_lens"]

aperture_radius_detector = aperture_radius_source*(1+image_padding)

lens_heights = [get_surface_data(elem["lens"],lens_resolution) for elem in results_minimize]    
lens_heights.append(get_surface_data(result["final_lens"],lens_resolution))

irrs = [elem["binned_irradiance"] for elem in results_minimize]    
irrs.append(result["final_irr_results"]["binned_irradiance"])


titles = [f"Desired Irradiance Smoothing $(n={elem["coeff_shape"][0]})$" for elem in results_minimize]
titles.append("Final Lens after postprocessing")


import matplotlib.pyplot as plt
import numpy as np

maxval_irrs = 0.
for k in range(len(irrs)):
    binned_irradiance = irrs[k]
    maxval_irrs = max(np.max(binned_irradiance),maxval_irrs)


maxval_s = -np.inf
minval_s = np.inf
for k in range(len(lens_heights)):
    elem = lens_heights[k]
    maxval_s = max(np.max(elem),maxval_s)
    minval_s = min(np.min(elem),minval_s)

def make_fig(title,k):
    # Generate some example data for the images
    data1 = lens_heights[k]
    data2 = irrs[k]
    # Create the figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # Plot the first image
    im1 = axs[0].imshow(data1, cmap='coolwarm', origin='lower', extent=[-aperture_radius_lens,aperture_radius_lens, -aperture_radius_lens, aperture_radius_lens],vmax=maxval_s,vmin=minval_s)
    axs[0].set_title("Relative Surface Profile", fontsize=12)
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    cbar1 = plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    cbar_title = "$[mm]$"
    cbar1.ax.set_title(cbar_title)  # Set label above
    
    # Plot the second image
    im2 = axs[1].imshow(data2, cmap='gray', origin='lower', extent=[-aperture_radius_detector, aperture_radius_detector, -aperture_radius_detector, aperture_radius_detector],vmin=0,vmax=maxval_irrs)
    axs[1].set_title("Irr. MC", fontsize=12)
    axs[1].set_xlabel("x [mm]")
    axs[1].set_ylabel("y [mm]")
    cbar2 = plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    cbar_title = "$[W/mm^2]$"
    cbar2.ax.set_title(cbar_title)  # Set label above
        
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    plt.savefig(temp_dir.name+f"/irr_{k}.png")



for k,title in enumerate(titles):
    make_fig(title,k)

def make_video():
    frames = sorted(
        [os.path.join(temp_dir.name, f) for f in os.listdir(temp_dir.name) if f.endswith(".png")]
    )

    if not frames:
        print("No frames found to create video.")
        return

    video_name = results_folder_main+"/output_video.avi"
    frame_size = cv2.imread(frames[0]).shape[1::-1]  # Get frame size (width, height)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 1, frame_size)

    for frame in frames:
        img = cv2.imread(frame)
        out.write(img)

    out.release()
    print(f"Video saved as {video_name}")

# Generate video after optimization
make_video()

temp_dir.cleanup()"""
# %%
