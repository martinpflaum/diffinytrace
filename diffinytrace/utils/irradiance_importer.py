# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

import torch
from PIL import Image
import numpy as np
from ..target_grid import GridSquare
import gc

"""
def from_image_square(file_name,padding_ratio,grey_ratio,aperture_radius):
    #TODO maybe generalize to rectangle  - change apreture_radius_detector to target_grid-nöö
    image = load_image(file_name,padding_ratio=padding_ratio,grey_ratio=grey_ratio)
    image = torch.tensor(image).to(torch.get_default_dtype())
    image = image.T
    image_flat = image.reshape(-1)
    grid = dit.target_grid.GridSquare(aperture_radius,grid_size=image.shape[0])
    area = grid.get_area()
    def desired_irradiance_func(x):    
        k = grid.get_k(x)
        tmp = image_flat[k]    
        return tmp/area
    return desired_irradiance_func


"""
def create_irradiance_from_image_square(file_name,padding_ratio,grey_ratio,aperture_radius,
                                        dtype=torch.get_default_dtype(),shape=None):
    #TODO maybe generalize to rectangle  - change apreture_radius_detector to target_grid-nöö
    image = load_image(file_name,padding_ratio=padding_ratio,grey_ratio=grey_ratio,shape=shape)
    #image = image.T
    image = np.array(image[::-1])
    image = np.array(image)
    image = torch.tensor(image).to(dtype=dtype)
    shape = image.shape
    
    image = image.reshape(-1)
    grid = GridSquare(aperture_radius,grid_size=shape[0])
    area = grid.get_area()
    def desired_irradiance_func(x):
        device = x.device
        dtype = x.dtype
        
        x = torch.clamp(x, min=-aperture_radius, max=aperture_radius)
        k = grid.get_k(x,round_to_bounds=True)
        k = k.cpu()
        
        tmp = image[k] 
        out = tmp/area
        out = out.to(device=device,dtype=dtype)
        return out
    
    return desired_irradiance_func


def pil_center_crop(image):
    width, height = image.size
    crop_box = None
    if width < height:
        crop_start = (height-width)//2
        crop_box = (0,crop_start , width, crop_start+width)  # (left, upper, right, lower)
    else:
        crop_start = (width-height)//2
        crop_box = (crop_start,0 ,crop_start+height,height)  # (left, upper, right, lower)

    cropped_image = image.crop(crop_box)
    return cropped_image


def load_image(name,padding_ratio,grey_ratio,shape=None):

    # Open and process the image
    image = Image.open(name)
    image = pil_center_crop(image)  # Assuming this function crops the image to its center
    image = image.convert("L")  # Convert the image to grayscale
    if shape is not None:
        image = image.resize(shape)

    # Convert the image to a numpy array
    image_array = np.array(image)/255.0
    image_array = np.ones_like(image_array)*grey_ratio+(1.-grey_ratio)*image_array
    
    # Calculate the padding size as 20% of the original dimensions
    pad_height = int(image_array.shape[0] * padding_ratio*0.5)  # 20% of the height
    pad_width = int(image_array.shape[1] * padding_ratio*0.5)   # 20% of the width

    # Pad the image with constant value 0 (or any value you prefer)
    padded_image = np.pad(image_array, 
                        pad_width=((pad_height, pad_height), (pad_width, pad_width)), 
                        mode='constant', 
                        constant_values=0)
    padded_image = padded_image / np.sum(padded_image)
    return padded_image