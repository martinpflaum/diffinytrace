# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "PlotableWavelength",
    "add_colour_bar",
    "plot"
]

import matplotlib.pyplot as plt
import numpy as np
import torch
import colour

class PlotableWavelength:
    def __init__(self,bounds,ylabel):
        """
        Initialize the PlotableWavelength object with bounds and ylabel.
        
        Args:
            bounds (tuple): The bounds for the wavelength range.
            ylabel (str): The label for the y-axis.
        """
        self.bounds = bounds
        self.ylabel = ylabel

def add_colour_bar(fig, ax, wl):
    """
    Add a color strip below the plot to represent the wavelength spectrum.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The main axis of the plot.
        wl (array-like): Wavelengths in µm.
    """
    left, bottom, width, height = ax.get_position().bounds
    color_ax = fig.add_axes([left, bottom - 0.15, width, 0.03])  # Position further below to avoid overlap
    
    def wavelength_to_rgb(wl):
        wl = wl*1000.
        if 360.0 < wl and wl < 780.0:
            rgb = colour.XYZ_to_sRGB(colour.wavelength_to_XYZ(wl))
            return np.clip(rgb, 0.0, 1.0)  # Ensure RGB values are within [0, 1]
        else:
            return (0.,0.,0.)

    colors = [wavelength_to_rgb(_wl) for _wl in wl]
        
    for i in range(len(wl) - 1):
        color_ax.fill_between([wl[i], wl[i + 1]], 0, 1, color=colors[i])
        
    color_ax.set_xlim(np.min(wl),np.max(wl))
    color_ax.axis('off')  # Hide axis for a clean color strip

#TODO change bmin and bmax to bounds
#refractive_index
def plot(wl,vals=None,title="",xlabel="Wavelength [µm]",ylabel="y",labels=None,colour_bar=True,linewidth=2,legend=True,resolution=500,show=True):
    """
    Plot a spectrum with a color strip below it.

    Args:
        wl (array-like): Wavelengths in nm or µm.
        vals (array-like): Values of the spectrum at the given wavelengths.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        labels (list): Labels for the different curves.
        colour_bar (bool): Whether to show a color bar.
        linewidth (int): Line width of the plot.
        legend (bool): Whether to show a legend.
        resolution (int): Resolution of the plot.
        show (bool): Whether to show the plot.

    Returns:
        None
    """
    if vals is None:
        if not isinstance(wl,PlotableWavelength):
            raise RuntimeError("if vals=None, wl must be a PlotableWavelength!")
        plotable_func = wl
        wl = np.linspace(*plotable_func.bounds,resolution)
        vals = plotable_func(wl)
        if ylabel=="y":
            ylabel = plotable_func.ylabel
    # Create figure and main axis
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)  # Increase space at the bottom
    vals = np.array(vals)
    
    wl = np.array(wl)
    if (wl>100.).any():
        print("wl is µm not nm! Setting wl to µm")
        wl = wl/1000.
    
    
    vmin = np.min(vals)
    vmax = np.max(vals)
    
    if len(vals.shape) == 1:
        val = vals
        vmin = np.min(val)
        ax.plot(wl, val, color='black', linewidth=linewidth)
        ax.fill_between(wl, val, color='gray', alpha=0.2)
    else:
        if vals.shape[1] != wl.shape[0]:
            vals = vals.T
    
        for i in range(len(vals)):
            val = vals[i]
            label = None
            if labels is not None:
                label = labels[i]
            ax.plot(wl, val,label=label, linewidth=linewidth)
            
    
    ax.set_xlim(np.min(wl),np.max(wl))
    ax.set_ylim(vmin,vmax+(vmax-vmin)*0.1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if colour_bar:
        add_colour_bar(fig, ax, wl)
    
    if labels is not None:
        if legend:
            ax.legend(loc='upper right')
    if show:
        plt.show()

    
"""
import matplotlib.pyplot as plt
import numpy as np
import colour


import matplotlib.pyplot as plt
import numpy as np
import colour

# Define the wavelength range (in nm) and the spectrum curve (e.g., Gaussian example)

bmin = 300.
bmax = 3000.
wavelengths = np.linspace(bmin, bmax, 1000)
spectrum = np.exp(-((wavelengths - 550) / 40) ** 2)  # Gaussian curve centered at 550 nm

# Function to map wavelength to RGB color
def wavelength_to_rgb(wavelength):
    if 360.0 < wavelength and wavelength < 780.0:
        rgb = colour.XYZ_to_sRGB(colour.wavelength_to_XYZ(wavelength))
        return np.clip(rgb, 0.0, 1.0)  # Ensure RGB values are within [0, 1]
    else:
        return (0.,0.,0.)
def add_color_strip(ax, bmin, bmax, resolution=1000):
    wavelengths = np.linspace(bmin, bmax, resolution)
    
    
    colors = [wavelength_to_rgb(wl) for wl in wavelengths]
    
    # Create the color strip as a series of filled segments
    for i in range(len(wavelengths) - 1):
        ax.fill_between([wavelengths[i], wavelengths[i + 1]], 0, 1, color=colors[i])
    
    ax.set_xlim(bmin, bmax)
    ax.axis('off')  # Hide axis for a clean color strip

# Create figure and main axis
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.3)  # Increase space at the bottom

# Plot the spectrum curve
ax.plot(wavelengths, spectrum, color='black', linewidth=2)
ax.fill_between(wavelengths, spectrum, color='gray', alpha=0.2)

# Add labels and limits for the main plot
ax.set_xlim(bmin, bmax)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_title("Spectrum with Corresponding Colors")

# Add an extra axis for the color strip below the main plot, with extra spacing
left, bottom, width, height = ax.get_position().bounds
color_ax = fig.add_axes([left, bottom - 0.15, width, 0.03])  # Position further below to avoid overlap
add_color_strip(color_ax, bmin, bmax, resolution=1000)

plt.show()
#%%

"""


"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import colour


import matplotlib.pyplot as plt
import numpy as np
import colour

# Define the wavelength range (in nm) and the spectrum curve (e.g., Gaussian example)
wavelengths = np.linspace(380, 780, 1000)
spectrum = pvlib.spectrum.get_am15g(wavelengths)
spectrum = np.array(spectrum)

# Function to map wavelength to RGB color
def wavelength_to_rgb(wavelength):
    rgb = colour.XYZ_to_sRGB(colour.wavelength_to_XYZ(wavelength))
    return np.clip(rgb, 0.0, 1.0)  # Ensure RGB values are within [0, 1]

colors = [wavelength_to_rgb(wl) for wl in wavelengths]
    
    # Create the color strip as a series of filled segments
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(len(wavelengths) - 1):
    plt.fill_between([wavelengths[i], wavelengths[i + 1]], 0, spectrum[i+1], color=colors[i],interpolate=True)
    
# Plot the spectrum curve
plt.xlim(380, 750)
plt.ylim(0., 2.0)
plt.plot(wavelengths, spectrum, color='black')
plt.show()
"""