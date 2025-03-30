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
    
