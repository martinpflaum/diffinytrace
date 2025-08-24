# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "plot"
]

import matplotlib.pyplot as plt
from torch import linspace,meshgrid,zeros,no_grad,is_tensor
from copy import deepcopy
import numpy as np
from typing import Callable,Tuple,Optional,Union

def plot(val: Union[Callable, np.ndarray], 
         title: str = "",
         x_range: Optional[Tuple[float, float]] = None,
         y_range: Optional[Tuple[float, float]] = None,
         cmap: str = "jet",
         subtitle: str = "",
         title_fontsize: int = 14,
         suptitle_fontsize: int = 12,
         interpolation: str = "none",
         xlabel: str = "x [mm]",
         ylabel: str = "y [mm]",
         colorbar: bool = True,
         norm = None,
         show: bool = True,
         vmin: float | None = None,
         vmax: float | None = None,
         resolution: int = 501,
         **kwargs):
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


def intensity(val,title="",x_range=None,y_range=None,cmap="jet",interpolation="none",xlabel="x [mm]",ylabel="y [mm]",norm=None,show=True,vmin: float | None = None,vmax: float | None = None,**kwargs):
    """
    Plot a 2D intensity distribution using matplotlib.

    Args:
        val (callable, np.ndarray, or torch.Tensor): The intensity data to plot. If callable, it should accept a 2D array of coordinates and return a 2D array of values.
        title (str, optional): Title of the plot.
        x_range (tuple or None, optional): Range of x-axis. If None, defaults to [0, 1].
        y_range (tuple or None, optional): Range of y-axis. If None, defaults to [0, 1].
        cmap (str, optional): Colormap to use. Default is "jet".
        interpolation (str, optional): Interpolation method for imshow. Default is "none".
        xlabel (str, optional): Label for x-axis. Default is "x [mm]".
        ylabel (str, optional): Label for y-axis. Default is "y [mm]".
        norm (matplotlib.colors.Normalize or None, optional): Normalization for color mapping.
        show (bool, optional): Whether to display the plot. Default is True.
        vmin (float or None, optional): Minimum value for color normalization.
        vmax (float or None, optional): Maximum value for color normalization.
        **kwargs: Additional keyword arguments passed to matplotlib's imshow.

    Returns:
        None
    """
    plot(val,f"{title} [$W/mm^2$]",x_range,y_range,cmap=cmap,interpolation=interpolation,xlabel=xlabel,ylabel=ylabel,norm=norm,show=show,vmin=vmin,vmax=vmax,**kwargs)

def height(val,title="",x_range=None,y_range=None,cmap="cool",interpolation="none",xlabel="x [mm]",ylabel="y [mm]",norm=None,show=True,vmin: float | None = None,vmax: float | None = None,**kwargs):
    """
    Plot a 2D height distribution using matplotlib.

    Args:
        val (callable, np.ndarray, or torch.Tensor): The height data to plot. If callable, it should accept a 2D array of coordinates and return a 2D array of values.
        title (str, optional): Title of the plot.
        x_range (tuple or None, optional): Range of x-axis. If None, defaults to [0, 1].
        y_range (tuple or None, optional): Range of y-axis. If None, defaults to [0, 1].
        cmap (str, optional): Colormap to use. Default is "cool".
        interpolation (str, optional): Interpolation method for imshow. Default is "none".
        xlabel (str, optional): Label for x-axis. Default is "x [mm]".
        ylabel (str, optional): Label for y-axis. Default is "y [mm]".
        norm (matplotlib.colors.Normalize or None, optional): Normalization for color mapping.
        show (bool, optional): Whether to display the plot. Default is True.
        vmin (float or None, optional): Minimum value for color normalization.
        vmax (float or None, optional): Maximum value for color normalization.
        **kwargs: Additional keyword arguments passed to matplotlib's imshow.

    Returns:
        None
    """
    plot(val,f"Height z [$mm$] of {title}",x_range,y_range,cmap=cmap,interpolation=interpolation,xlabel=xlabel,ylabel=ylabel,norm=norm,show=show,vmin=vmin,vmax=vmax,**kwargs)

"""
TODO: these functions need testing again before integration

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
    
