# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "Plotable"
]

from matplotlib.colors import to_rgb
from typing import List,Tuple,Optional,Union

class Plotable:
    """
    Base class for objects that can be visualized in the optical system.

    This class provides a common interface for objects that support 2D/3D plotting,
    color scale configuration, and hierarchical visualization. Subclasses should implement
    methods for generating plot points and color scales as needed.

    Attributes:
        fill_color (str): Color used to fill the object in plots.
        outline_color (str): Color used for the object's outline in plots.
        is_volume (bool): If True, the object is treated as a volumetric entity.
    """
    def __init__(self, fill_color:str = "white", outline_color:str = "black", is_volume:bool = False):
        """
        Initialize the plotable object with fill and outline colors.
        
        Args:
            fill_color (str): The color used to fill the object.
            outline_color (str): The color used for the outline of the object.
            is_volume (bool): If True, the object is treated as a volume.
        
        """
        self.fill_color = fill_color
        self.outline_color = outline_color
        self.is_volume = is_volume

    def get_plotly_color_scale(self)->List[List[Union[float,str]]]:
        """
        Returns a color scale for Plotly, based on the fill and outline colors.
        """
        
        fill_color_rgb = to_rgb(self.fill_color)
        outline_color_rgb = to_rgb(self.outline_color)

        fill_color_text = f'rgb({int(fill_color_rgb[0] * 255)}, {int(fill_color_rgb[1] * 255)}, {int(fill_color_rgb[2] * 255)})'
        outline_color_text = f'rgb({int(outline_color_rgb[0] * 255)}, {int(outline_color_rgb[1] * 255)}, {int(outline_color_rgb[2] * 255)})'

        colorscale = [[0,fill_color_text],
                      [1,outline_color_text]]
        return [colorscale]
    
    def get_plotable_childs(self)->List:
        """
        Returns a list of all plotable child objects of this object.
        Each child is represented as a list containing the child object and its name.
        """
        out = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Plotable):
                out.append([attr,attr_name])
        return out
    
    def get_plot_points_2D(self, resolution:int)->List:
        """
        Returns a list of 2D plot points for the object.
        
        Args:
            resolution (int): The resolution for the plot points.
        
        
        Returns:
            list: A list of 2D plot points.
        """
        
        print("get_plot_points_2D not implemented")
        return []
    
    def get_plot_points_3D(self, resolution:int)->List:
        """
        Returns a list of 3D plot points for the object.
        
        Args:
            resolution (int): The resolution for the plot points.
        
        Returns:
            list: A list of 3D plot points.
        """
        print("get_plot_points_3D not implemented")
        return []

