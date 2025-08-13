import diffinytrace as dit
from diffinytrace.nonimaging.scripts.sunlight_picture import create_lens
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
from PIL import Image

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
image_file_name = "image_vertical.jpg"
#results_folder_main = "Results/results_v3"#.2"
#create_folder(results_folder_main)


def save_matrix_plot_to_temp(matrix,extent,cmap,vmin,vmax,cbar_labelsize,cbar_title,cbar_title_fontsize,plot_colorbar,show_x_axis):
    # Create the plot
    fig, ax = plt.subplots()
    matrix = matrix[::-1]
    cax = ax.imshow(matrix, cmap=cmap,vmin=vmin,vmax=vmax,extent=extent)
    #ax.axis('off')  # Remove the axes
    if plot_colorbar:
        cbar = plt.colorbar(cax,shrink=0.65,aspect=9)  # Add a colorbar for reference
        cbar.ax.tick_params(labelsize=cbar_labelsize)
        cbar.ax.set_title(cbar_title, fontsize=cbar_title_fontsize, pad=10,loc="left")  # Set label above
        offset_text = cbar.ax.yaxis.offsetText
        offset_text.set_size(cbar_labelsize)  # Set the font size
        offset_text.set_ha('left')  # Align the text to the left

    #ax.set_ylabel("ylabel", fontsize=12, labelpad=10)  # Add ylabel
    ax.set_aspect('equal')

    range_xaxis = [extent[0],extent[1]]
    len_xaxis = range_xaxis[1]-range_xaxis[0]
    if show_x_axis:
        ax.set_xticks([range_xaxis[0]+len_xaxis*0.25, range_xaxis[0]+len_xaxis*0.5,range_xaxis[0]+len_xaxis*0.75])
        ax.tick_params(axis='x', labelsize=cbar_labelsize)  # Set size for x-axis tick labels

        ax.set_xlabel("x [mm]", fontsize=cbar_title_fontsize, labelpad=10)  # Add ylabel
        
        ax.xaxis.set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        ax.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    ax.yaxis.set_visible(False)
    #plt.tight_layout(pad=2.0)
    
    # Align the plot edges with the matrix
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Save the plot to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        plt.savefig(temp_path, bbox_inches='tight', pad_inches=0.025, format='png')
        plt.close(fig)  # Close the plot to free resources
    
    return temp_path

def get_row_vmin_vmax(matrices):
    _vmin = np.inf
    _vmax = -np.inf
    matrices = [np.array(matrix) for matrix in matrices]
    for matrix in matrices:
        _vmin = min(_vmin,(matrix).min())
        _vmax = max(_vmax,(matrix).max())
    return _vmin,_vmax

def make_row(matrices,extent,cmap,cbar_labelsize,cbar_title,cbar_title_fontsize,show_x_axis_first,vmin=None,vmax=None):
    matrices = [np.array(matrix) for matrix in matrices]
    _vmin,_vmax = get_row_vmin_vmax(matrices)

    if vmin is None:
        vmin = _vmin
    if vmax is None:
        vmax = _vmax
        
    file_names = []
    for i,matrix in enumerate(matrices):
        if len(matrices)-1 == i:
            file_name = save_matrix_plot_to_temp(matrix,extent,cmap,vmin,vmax,cbar_labelsize,cbar_title,cbar_title_fontsize,plot_colorbar = True,show_x_axis=False)
            file_names.append(file_name)
        elif show_x_axis_first and i ==0:
            file_name = save_matrix_plot_to_temp(matrix,extent,cmap,vmin,vmax,cbar_labelsize,cbar_title,cbar_title_fontsize,plot_colorbar = False,show_x_axis=True)
            file_names.append(file_name)

        else:
            file_name = save_matrix_plot_to_temp(matrix,extent,cmap,vmin,vmax,cbar_labelsize,cbar_title,cbar_title_fontsize,plot_colorbar = False,show_x_axis=False)
            file_names.append(file_name)
        
    return file_names    

def concatenate_images_tempfile_row(image_files):
    # Open images and find total width and max height
    images = [Image.open(file) for file in image_files]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create a new image with the combined size
    concatenated_image = Image.new('RGB', (total_width, max_height),color='white')
    
    # Paste each image next to each other
    x_offset = 0
    for img in images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    # Save the concatenated image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        concatenated_image.save(temp_file.name)
        return temp_file.name

def concatenate_images_tempfile_vertical(image_files):
    # Open images and find total width and max height
    images = [Image.open(file) for file in image_files]
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)
    
    # Create a new image with the combined size
    concatenated_image = Image.new('RGB', (max_width, total_height),color='white')
    
    # Paste each image below the previous one
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    # Save the concatenated image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        concatenated_image.save(temp_file.name)
        return temp_file.name


def concatenate_images_vertical(image_files,file_name):
    # Open images and find total width and max height
    images = [Image.open(file) for file in image_files]
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)
    
    # Create a new image with the combined size
    concatenated_image = Image.new('RGB', (max_width, total_height),color='white')
    
    # Paste each image below the previous one
    y_offset = 0
    for img in images:
        concatenated_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    # Save the concatenated image to a temporary file
    concatenated_image.save(file_name)
    
def create_image_with_text_orientation(base_image_path, text, ratio,font_size, vertical=True):
    """
    Create an empty image with text. Adjust dimensions based on orientation.
    
    Args:
        base_image_path (str): Path to the base image.
        text (str): Text to display on the created image.
        ratio (float): Ratio to adjust width/height.
        vertical (bool): If True, text is vertical (rotated) with width adjusted to a ratio of height.
                         If False, text is horizontal with height adjusted to a ratio of width.
    
    Returns:
        str: Path to the temporary file containing the created image.
    """
    base_image = Image.open(base_image_path)
    base_width, base_height = base_image.size

    if vertical:
        # Maintain base height, adjust width by ratio
        output_width = int(base_width * ratio)
        output_height = base_height
    else:
        # Maintain base width, adjust height by ratio
        output_width = base_width
        output_height = int(base_height * ratio)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new('RGB', (output_width, output_height), color='white')

    try:
        # Load a default font; adjust size as needed
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        # Fallback to a basic PIL font if default is unavailable
        font = ImageFont.load_default()

    if vertical:
        # Create an image for rotated text
        text_image = Image.new('RGBA', (output_height, output_width), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_image)
        text_bbox = text_draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = (output_height - text_width) // 2
        text_y = (output_width - text_height) // 2

        # Draw and rotate the text
        text_draw.text((text_x, text_y), text, fill="black", font=font)
        rotated_text = text_image.rotate(90, expand=True)
        new_image.paste(rotated_text, (0, 0), rotated_text)
    else:
        # Draw horizontal text directly on the image
        draw = ImageDraw.Draw(new_image)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = (output_width - text_width) // 2
        text_y = (output_height - text_height) // 2
        draw.text((text_x, text_y), text, fill="black", font=font)

    # Save the new image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        new_image.save(temp_file.name)
        return temp_file.name
    
def create_white_image_with_dimensions2(wdith_plus_img,width_minus_img, image2_path):
    # Open the two images to get their dimensions
    wdith_plus_img = Image.open(wdith_plus_img)
    width_minus_img= Image.open(width_minus_img)
    image2 = Image.open(image2_path)
        
    width = wdith_plus_img.size[0]-width_minus_img.size[0]  # Width of the first image
    height = image2.size[1]  # Height of the second image
        
    # Create a new blank white image with the specified dimensions
    white_image = Image.new('RGB', (width, height), color='white')
        
    # Save the new image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        white_image.save(temp_file.name)
        return temp_file.name



def create_white_image_with_dimensions(image1_path, image2_path):
    """
    Create a white image with the width of the first image and the height of the second image.
    
    Args:
        image1_path (str): Path to the first image (for width).
        image2_path (str): Path to the second image (for height).
    
    Returns:
        str: Path to the temporary file containing the created image.
    """
    # Open the two images to get their dimensions
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
        
    width = image1.size[0]  # Width of the first image
    height = image2.size[1]  # Height of the second image
        
    # Create a new blank white image with the specified dimensions
    white_image = Image.new('RGB', (width, height), color='white')
        
    # Save the new image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        white_image.save(temp_file.name)
        return temp_file.name
    
def _image_from_grid(image_grid,rows_extent,rows_cmap,rows_title,columns_title,cbar_titles,font_size_PIL,cbar_labelsize,cbar_title_fontsize,rows_vmin=None,rows_vmax=None):
    if rows_vmin is None:
        rows_vmin = [None for elem in image_grid]
    if rows_vmax is None:
        rows_vmax = [None for elem in image_grid]
    
    out = []
    last_extents = None
    show_x_axis_first_all = [False for elem in image_grid]
    print("_image_from_grid",len(image_grid),len(rows_extent))
    for i,matrices_row in enumerate(image_grid):
        if not last_extents is None:
            all_same = True
            for k in range(len(last_extents)):
                if last_extents[k]!=rows_extent[i][k]:
                    all_same = False
            if not all_same:
                show_x_axis_first_all[i-1] = True
        last_extents = rows_extent[i]
    show_x_axis_first_all[len(image_grid)-1] = True
    
    for i,matrices_row in enumerate(image_grid):
        # Example usage (example files would need valid paths)
        show_x_axis_first = show_x_axis_first_all[i]
        #if not last_extents is None:
        #    all_same = True
        #    for k in range():

        row_files = make_row(matrices_row,rows_extent[i],rows_cmap[i],cbar_labelsize,cbar_titles[i],cbar_title_fontsize,show_x_axis_first=show_x_axis_first,vmin=rows_vmin[i],vmax=rows_vmax[i])
        image_column_text = create_image_with_text_orientation(row_files[1], rows_title[i], 0.2,font_size_PIL,vertical=True)
        
        if i==0:
            images_row_text = []
            for k,elem in enumerate(columns_title):

                tmp1 = create_image_with_text_orientation(row_files[0], elem, 0.2,font_size_PIL,vertical=False)
                if k == 0:
                    tmp2 = create_white_image_with_dimensions(image_column_text,tmp1)
                    images_row_text.append(tmp2)
                
                images_row_text.append(tmp1)

                if k==(len(columns_title)-1):
                    tmp2 = create_white_image_with_dimensions2(row_files[k],tmp1, tmp1)
                    images_row_text.append(tmp2)
                
            image_combined2 = concatenate_images_tempfile_row(images_row_text)
            out.append(image_combined2)

        row_files = [image_column_text]+row_files
        image_combined1 = concatenate_images_tempfile_row(row_files)
        out.append(image_combined1)
    
    
    return concatenate_images_tempfile_vertical(out)


def image_from_grid(image_grid,rows_extent,rows_vidx,rows_cmap,rows_title,columns_title,cbar_titles,max_num_column,font_size_PIL,cbar_labelsize,cbar_title_fontsize):
    num_x = len(image_grid[0])
    num_splits = int(np.ceil(num_x/float(max_num_column)))
    
    rows_vmin_idx_dict = {}
    rows_vmax_idx_dict = {}


    
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
            rows_vmax_idx_dict[idx] = min(vmax,rows_vmax_idx_dict[idx]) 
           

    rows_vmin = []
    rows_vmax = []

    for i,row in enumerate(image_grid):
        idx = rows_vidx[i]
        vmin = rows_vmin_idx_dict[idx]
        vmax = rows_vmax_idx_dict[idx]


        rows_vmin.append(vmin)
        rows_vmax.append(vmax)
        
    num_x_perbrow = int(np.ceil(num_x/float(num_splits)))
    image_grids = []
    for i in range(num_splits):
        split = []
        for row in image_grid:
            split.append(row[num_x_perbrow*i:num_x_perbrow*(i+1)])

        image_grids.append(split)

    out = []
    for i,split in enumerate(image_grids):
        
        tmp = _image_from_grid(split,rows_extent,rows_cmap,rows_title,columns_title[num_x_perbrow*i:num_x_perbrow*(i+1)],cbar_titles,font_size_PIL,cbar_labelsize,cbar_title_fontsize,rows_vmin=rows_vmin,rows_vmax=rows_vmax)
        out.append(tmp)
    return out

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
