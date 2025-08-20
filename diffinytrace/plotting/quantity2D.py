# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "plot"
]

import matplotlib.pyplot as plt
from torch import linspace,meshgrid,zeros,no_grad,is_tensor
from copy import deepcopy
import numpy as np

def plot(val,title="",x_range=None,y_range=None,cmap="jet",subtitle="",title_fontsize=14,suptitle_fontsize=12,interpolation="none",xlabel="x [mm]",ylabel="y [mm]",colorbar=True,norm=None,show=True,vmin: float | None = None,vmax: float | None = None,resolution=501,**kwargs):
    """
    Plot a 2D quantity using matplotlib.
    This function handles both callable quantities and numpy arrays.
    If a callable is provided, it should accept a 2D array of coordinates and return a 2D array of values.
    The function will create a 2D plot with the specified parameters.
    If a numpy array is provided, it will be plotted directly.
    
    Args:
        val (callable or np.ndarray): The quantity to plot. If callable, it should accept a 2D array of coordinates.
        title (str): Title of the plot.
        x_range (tuple): Range of x-axis.
        y_range (tuple): Range of y-axis.
        cmap (str): Colormap to use.
        subtitle (str): Subtitle of the plot.
        title_fontsize (int): Font size of the title.
        suptitle_fontsize (int): Font size of the subtitle.
        interpolation (str): Interpolation method.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        colorbar (bool): Whether to show colorbar.
        show (bool): Whether to show the plot.
        vmin (float): Minimum value for color normalization.
        vmax (float): Maximum value for color normalization.
        resolution (int): Resolution of the plot.
        **kwargs: Additional arguments for imshow.
    
    Returns:
        None
    """
    
    if is_tensor(val):
        val = val.detach().cpu().numpy()#.T
    
    if not (x_range is None):
        if isinstance(x_range,float):
            x_range = [-x_range,x_range]
        if y_range is None:
            y_range = x_range
    if y_range is None:
        y_range = [0,1.]
    
    if x_range is None:
        x_range = [0,1.]

    if not isinstance(val,np.ndarray):
        _y = linspace(*x_range,resolution)
        _x = linspace(*y_range,resolution)
        mesh = meshgrid(_y,_x)
        y = mesh[0].reshape(-1)
        x = mesh[1].reshape(-1)
        O = zeros((x.shape[0],2))        
        O[:,0] = x
        O[:,1] = y
        val = val(O).reshape(resolution,resolution)
        if is_tensor(val):
            val = val.detach().cpu().numpy()
    
    val = val[::-1]
    
    plt.cla()   # Clear axis
    plt.clf()   # Clear figure
    fig, ax = plt.subplots()
    mappable = ax.imshow(val,cmap=cmap,interpolation=interpolation,extent=list(x_range)+list(y_range),norm=norm,vmin=vmin,vmax=vmax,**kwargs)
    
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if subtitle != "":
        fig.suptitle(title,fontsize=title_fontsize)
        ax.set_title(subtitle, fontsize=suptitle_fontsize)
    else:
        ax.set_title(title,fontsize=title_fontsize)

    if colorbar:
        plt.colorbar(mappable, ax=ax)

    if show:
        plt.show()


"""
TODO: these functions need testing again before integration
def intensity(val,title="",x_range=None,y_range=None,cmap="jet",interpolation="none",xlabel="x [mm]",ylabel="y [mm]",norm=None,show=True,vmin: float | None = None,vmax: float | None = None,**kwargs):
    plot(val,f"{title} [$W/mm^2$]",x_range,y_range,cmap=cmap,interpolation=interpolation,xlabel=xlabel,ylabel=ylabel,norm=norm,show=show,vmin=vmin,vmax=vmax,**kwargs)

def height(val,title="",x_range=None,y_range=None,cmap="cool",interpolation="none",xlabel="x [mm]",ylabel="y [mm]",norm=None,show=True,vmin: float | None = None,vmax: float | None = None,**kwargs):
    plot(val,f"Height z [$mm$] of {title}",x_range,y_range,cmap=cmap,interpolation=interpolation,xlabel=xlabel,ylabel=ylabel,norm=norm,show=show,vmin=vmin,vmax=vmax,**kwargs)
    
def surface(surface,name,aperture_radius,resolution=256,is_square=True,norm=None,show=True,**kwargs):
    surface = deepcopy(surface)
    surface = surface.cpu()
    x_range = (-aperture_radius,aperture_radius)
    y_range = (-aperture_radius,aperture_radius)
    _x = linspace(-aperture_radius,aperture_radius,resolution)
    _y = linspace(-aperture_radius,aperture_radius,resolution)
    mesh = meshgrid(_x,_y)
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    O = zeros((x.shape[0],3))        
    
    O[:,0] = x
    O[:,1] = y
    z = None
    
    with no_grad():
        z = surface.functional(O,*surface.get_functional_param_args())
    
    if not is_square:
        z[O[:,[0,1]].norm(dim=-1)>aperture_radius] = float("nan")

    z = z.detach().reshape(resolution,resolution)
    height(z,name,x_range,y_range,norm=norm,show=show,**kwargs)
"""
    
