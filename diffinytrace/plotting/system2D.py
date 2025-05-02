# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from copy import deepcopy

def annotate_position_simple(nz,ny,name):
    """
    Annotate the position of a point in 2D space.
    
    Args:
        nz (torch.Tensor): z-coordinates of the point.
        ny (torch.Tensor): y-coordinates of the point.
        name (str): Name to annotate.
    """
    zdiff = (torch.max(nz)-torch.min(nz))
    ydiff = (torch.max(ny)-torch.min(ny))
    offset = max(ydiff*0.05,zdiff*0.025)
    argmax = torch.argmax(ny)
    zpos = nz[argmax]#torch.min(nz)+zdiff*0.5
    ypos = ny[argmax]
    fontsize = 10
    #-len(name)*fontsize/4.0
    plt.annotate(name,xy=(zpos, ypos),fontsize=fontsize,xytext=(0.0,fontsize*0.5), textcoords='offset points')


def annotate_position(position,offset,name,color="black"):
    """
    Annotate the position of a point in 2D space with an arrow.
    
    Args:
        position (tuple): Position of the point (z, y).
        offset (tuple): Offset for the annotation.
        name (str): Name to annotate.
        color (str): Color of the annotation.
        
    """
    plt.annotate(name,color=color,xy=position,xytext=offset, textcoords='offset points',arrowprops=dict(arrowstyle="->",color=color,linewidth=1.5, mutation_scale=10))


def annotated_arrow(start,end,offset,name,arrowstyle,color="black"):
    """
    Annotate an arrow between two points in 2D space.
    
    Args:
        start (tuple): Start position of the arrow (z, y).
        end (tuple): End position of the arrow (z, y).
        offset (tuple): Offset for the annotation.
        name (str): Name to annotate.
        arrowstyle (str): Style of the arrow.
        color (str): Color of the arrow and annotation.
        
    """
    
    arrow_patch = patches.FancyArrowPatch(start, end, arrowstyle=arrowstyle,linewidth=1.5, mutation_scale=10,color=color)
    plt.gca().add_patch(arrow_patch)
    middle = (start[0]+ (end[0]-start[0])*0.5,start[1]+ (end[1]-start[1])*0.5)

    plt.annotate(name,xy=middle,xytext=offset, textcoords='offset points',color=color)

def layout():
    """
    Set up the layout for the plot.
    """
    #plt.grid(True)
    plt.margins(x=0.1,y=0.1)
    plt.gca().set_aspect('equal')
    plt.ylabel("y [mm]")
    plt.xlabel("z [mm]")

def ray_paths(rays,ray_color="#85549c",ray_linewidth=1.25):
    """
    Plot the ray paths in 2D space.
    
    Args:
        rays (list[torch.Tensor]): List of ray paths to plot.
        ray_color (str): Color of the rays.
        ray_linewidth (float): Line width of the rays.
    """
    ray_color = mcolors.to_hex(ray_color)
    print("WARNING: ray_paths will project the ray position onto the y-z plane!")
    pathsA = rays
    if torch.is_tensor(rays[0]):
        pathsA = np.array([elem.numpy() for elem in rays])
    pathsA = np.array(pathsA)

    for iray in range(pathsA.shape[1]):
        plt.plot(pathsA[:,iray,2],pathsA[:,iray,1],color=ray_color,linewidth=ray_linewidth)



def _plot_surface(surface,name,resolution,annotate,fill_color,outline_color,linewidth):
    surface_list = surface.get_plot_points2D(resolution)
    if len(surface_list)==0:
        return
    if fill_color is None:
        fill_color = surface.fill_color
    if outline_color is None:
        outline_color = surface.outline_color

    zs,ys = torch.cat([z for z,y in surface_list]),torch.cat([y for z,y in surface_list])
    if annotate:
        annotate_position_simple(zs,ys,name)
    if surface.is_volume:
        ax = plt.gca()
        ax.fill(zs, ys, facecolor=fill_color, edgecolor=outline_color, linewidth=linewidth)
    else:
        for z,y in surface_list:
            plt.plot(z,y,color=outline_color,label="",linewidth=linewidth)
    

def _plot_surface_recursively(current_elem,name,resolution=200,annotate=False,fill_color=None,outline_color=None,linewidth=None):
    _plot_surface(current_elem,name,resolution,annotate,fill_color,outline_color,linewidth)
    for elem,elem_name in current_elem.get_plotable_childs():
        _plot_surface_recursively(elem,elem_name,resolution,annotate,fill_color,outline_color,linewidth)
        

def plot(element=None,rays=None,resolution=200,annotate=False,ray_color="#85549c",ray_linewidth=1.25,fill_color=None,outline_color=None,linewidth=None,show=True):
    """
    Plot a 2D surface and optionally ray paths.
    Args:
        element (Plotable): The element to plot.
        rays (list[torch.Tensor]): List of ray paths to plot.
        resolution (int): Resolution for the surface plot.
        annotate (bool): Whether to annotate the surface.
        ray_color (str): Color of the rays.
        ray_linewidth (float): Line width of the rays.
        fill_color (str): Fill color for the surface.
        outline_color (str): Outline color for the surface.
        linewidth (float): Line width for the surface.
        show (bool): Whether to show the plot.
    
    Returns:
        None
    """
    
    layout()
    
    if isinstance(element,(list,tuple)):
        for subelem in element:
            subelem = deepcopy(subelem)
            subelem = subelem.to("cpu")
            _plot_surface_recursively(subelem,"",resolution,annotate,fill_color,outline_color,linewidth)

    elif not element is None:    
        element = deepcopy(element)
        element = element.to("cpu")
        _plot_surface_recursively(element,"",resolution,annotate,fill_color,outline_color,linewidth)
    

    if not rays is None:
        if isinstance(rays,dict):
            rays = rays["ray_paths"]

        if torch.is_tensor(rays[0]):
            rays = [elem.cpu() for elem in rays]
        
        ray_paths(rays,ray_color=ray_color,ray_linewidth=ray_linewidth)
    if show:
        plt.show()

