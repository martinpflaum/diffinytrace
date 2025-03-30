# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


import matplotlib.colors as mcolors

class Plotable:
    def __init__(self,fill_color="white", outline_color="black",is_volume=False):
        self.fill_color = fill_color
        self.outline_color = outline_color
        self.is_volume = is_volume

    def get_plotly_color_scale(self):
        fill_color_rgb = mcolors.to_rgb(self.fill_color)
        outline_color_rgb = mcolors.to_rgb(self.outline_color)

        fill_color_text = f'rgb({int(fill_color_rgb[0] * 255)}, {int(fill_color_rgb[1] * 255)}, {int(fill_color_rgb[2] * 255)})'
        outline_color_text = f'rgb({int(outline_color_rgb[0] * 255)}, {int(outline_color_rgb[1] * 255)}, {int(outline_color_rgb[2] * 255)})'

        colorscale = [[0,fill_color_text],
                      [1,outline_color_text]]
        return [colorscale]
    
    def get_plotable_childs(self):
        out = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Plotable):
                out.append([attr,attr_name])
        return out
    
    def get_plot_points2D(self,resolution):
        print("get_plot_points2D not implemented")
        return []
    
    def get_plot_points3D(self,resolution):
        print("get_plot_points3D not implemented")
        return []
    
