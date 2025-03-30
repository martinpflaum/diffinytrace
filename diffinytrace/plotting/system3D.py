# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
#import matplotlib.pyplot as plt
#from PIL import Image
#import tempfile
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
import plotly.io as pio
import copy

##2013FF
def ray_paths_one_bin(rays,ray_color,ray_linewidth):
    rays = [elem.numpy() for elem in rays]
    rays = np.array(rays)
    rays = torch.tensor(rays)
    x = rays[:,:,0].reshape(-1)
    y = rays[:,:,1].reshape(-1)
    z = rays[:,:,2].reshape(-1)
    ray_id = torch.arange(rays.shape[1]).reshape(-1,1).repeat(1,rays.shape[0]).T.reshape(-1)
    df = pd.DataFrame({"X":x,"Y":y,"Z":z,"ray id":ray_id})

    line_fig = px.line_3d(df, x='X', y='Y', z='Z', line_group="ray id")

    for k in range(len(line_fig.data)):
        line_fig.data[k].line.color = ray_color
        line_fig.data[k].line.width = ray_linewidth
    return line_fig

def ray_paths(rays,ray_color="#9673A6",ray_linewidth=3):#9673A6#D6B656
    ray_color = mcolors.to_hex(ray_color)
    data = []
    if not rays is None:
        ray_path_bins = {}
        for elem in rays:
            if not len(elem) in ray_path_bins.keys():
                ray_path_bins[len(elem)] = []
            ray_path_bins[len(elem)] += [elem]
        for key in ray_path_bins.keys():
            line_fig = ray_paths_one_bin(ray_path_bins[key],ray_color,ray_linewidth)
            data += [*line_fig.data]
    return data


def surface(transformation,surface,name,aperture_radius,resolution,colorscale,is_square=False):
    _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
    _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
    mesh = torch.meshgrid(_x,_y)
    x = mesh[0].reshape(-1)
    y = mesh[1].reshape(-1)
    O = torch.zeros((x.shape[0],3))
    
    if not is_square:
        mul = (torch.sqrt(x*x+y*y)>aperture_radius).float()/torch.sqrt(x*x+y*y)*aperture_radius
        mul += (torch.sqrt(x*x+y*y)<aperture_radius).float()
        x = x*mul
        y = y*mul
            
    
    O[:,0] = x
    O[:,1] = y
    z = None
    
    with torch.no_grad():
        z = surface.explicit(O)
    z = z.detach().reshape(-1)
    x = x.detach().reshape(-1)
    y = y.detach().reshape(-1)
    v = torch.zeros((x.shape[0],4))
    v[:,0] = x
    v[:,1] = y
    v[:,2] = z
    v[:,3] = torch.ones_like(v[:,3])   
    
    Mv = None
    with torch.no_grad():
        M = transformation.get_transformation_matrix().detach()
        Mv = v@M.T

    x = Mv[:,0].reshape(_x.shape[0],_x.shape[0])
    y = Mv[:,1].reshape(_x.shape[0],_x.shape[0])
    z = Mv[:,2].reshape(_x.shape[0],_x.shape[0])

    data = []
    data += [go.Surface(x=x, y=y, z=z,showscale=False,name=name,colorscale=colorscale)]
    return data


def get_optical_system_layout(show_grid,xlabel="x [mm]",ylabel="y [mm]",zlabel="z [mm]",xticks=None,yticks=None,zticks=None,axislabel_font_size=10,tick_font_size=10):
    #TODO write wrapper for plot3D!
    camera = dict(
        up=dict(x=1., y=0., z=0)
    )
    xaxis=dict(
        visible=show_grid,
        title=dict(text=xlabel, font=dict(size=axislabel_font_size)),  # X axis title font size
        tickfont=dict(size=tick_font_size)  # X axis tick labels font size
    )
    yaxis=dict(
        visible=show_grid,
        title=dict(text=ylabel, font=dict(size=axislabel_font_size)),  # Y axis title font size
        tickfont=dict(size=tick_font_size)  # Y axis tick labels font size
    )
    zaxis=dict(
        visible=show_grid,
        title=dict(text=zlabel, font=dict(size=axislabel_font_size)),  # Z axis title font size
        tickfont=dict(size=tick_font_size)  # Z axis tick labels font size
    )

    if xticks is not None:
        xaxis["tickvals"] = xticks
    if yticks is not None:
        yaxis["tickvals"] = yticks

    if zticks is not None:
        zaxis["tickvals"] = zticks

    scene = dict(
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
        aspectmode='data',
        aspectratio = dict(x=1, y=1, z=1),
    )
    """    scene = dict(
        xaxis = dict(visible=show_axis),
        yaxis = dict(visible=show_axis),
        zaxis = dict(visible=show_axis),
        aspectmode='data',
        aspectratio = dict(x=1, y=1, z=1),
        xaxis_title='x [mm]',
        yaxis_title='y [mm]',
        zaxis_title='z [mm]')
    """
    layout = go.Layout(scene_camera=camera,scene=scene)
    return layout


def _plot_surface(surface,name,resolution):
    surface_list = surface.get_plot_points3D(resolution)
    if len(surface_list)==0:
        return []
    colorscale = surface.get_plotly_color_scale()
            
    data = []
    for k,(x,y,z) in enumerate(surface_list):
        try:
            data += [go.Surface(x=x, y=y, z=z,showscale=False,name=name+f"_{k}",colorscale=colorscale[k])]
        except:
            print("Wrong number of colorscales or colorscales is not correct, fallback to first colorscale!")
            data += [go.Surface(x=x, y=y, z=z,showscale=False,name=name+f"_{k}",colorscale=colorscale[0])]
        
    return data


def _plot_surface_recursively(current_elem,name,resolution):
    out = _plot_surface(current_elem,name,resolution)
    for elem,elem_name in current_elem.get_plotable_childs():
        out += _plot_surface_recursively(elem,elem_name,resolution)
    return out

def plot(element=None,
         rays=None,
         resolution=32,
         show_grid=True,
         xlabel="x [mm]",
         ylabel="y [mm]",
         zlabel="z [mm]",
         xticks=None,
         yticks=None,
         zticks=None,
         axislabel_font_size=10,
         tick_font_size=10,
         ray_color="#9673A6",
         ray_linewidth=3.,
         show=True,
         html_file_name=None):
    
    data = []
    if isinstance(element,(list,tuple)):
        for subelem in element:
            subelem = copy.deepcopy(subelem)
            subelem = subelem.to("cpu")
            data += _plot_surface_recursively(subelem,"",resolution)

    elif not element is None:
        element = copy.deepcopy(element)
        element = element.to("cpu")
        data += _plot_surface_recursively(element,"",resolution)
        
    if not rays is None:
        if isinstance(rays,dict):
            rays = rays["ray_paths"]

        rays = [elem.cpu() for elem in rays]
        data += ray_paths(rays,ray_color,ray_linewidth)
    layout = get_optical_system_layout(show_grid,xlabel,ylabel,zlabel,xticks,yticks,zticks,axislabel_font_size,tick_font_size)
    fig = go.Figure(data=data,layout=layout)
    if show:
        fig.show()

    if not html_file_name is None:
        if html_file_name[-5:]!=".html":
            raise RuntimeError("html_file_name should end with .html!")

        pio.write_html(fig, file=html_file_name, auto_open=False)
    
    if not show:
        return fig