"""
Copyright (C) 2024 Martin Pflaum

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>."""

import matplotlib.pyplot as plt
from torch import linspace,meshgrid,zeros,no_grad,is_tensor
from copy import deepcopy
import numpy as np
#TODO implement shorter version for grid class
#TODO yrange defined "reversly" so it fits val

"""
X: ArrayLike | PIL.Image.Image,
    cmap: str | Colormap | None = None,
    norm: str | Normalize | None = None,
    aspect: Literal["equal", "auto"] | float | None = None,
    interpolation: str | None = None,
    alpha: float | ArrayLike | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    origin: Literal["upper", "lower"] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    interpolation_stage: Literal["data", "rgba"] | None = None,
    filternorm: bool = True,
    filterrad: float = 4.0,
    resample: bool | None = None,
    url: str | None = None,
    data=None,
    **kwargs,
"""
def plot(val,title="",x_range=None,y_range=None,cmap="jet",subtitle="",title_fontsize=14,suptitle_fontsize=12,interpolation="none",xlabel="x [mm]",ylabel="y [mm]",colorbar=True,norm=None,show=True,vmin: float | None = None,vmax: float | None = None,resolution=501,**kwargs):
    #val = deepcopy(val)

    
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
    
    #TODO test Flip y axis!!
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

    
