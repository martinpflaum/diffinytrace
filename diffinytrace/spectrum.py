# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


import torch
import torch.nn as nn
import numpy as np
import pvlib
from .plotting.wavelength import PlotableWavelength



class Spectrum(nn.Module,PlotableWavelength):
    """
    A class to represent a spectrum as a function of wavelength.
    """
    def __init__(self,func,bounds):
        """
        Initialize the Spectrum class.
        Args:
            func (callable): A function that takes a wavelength and returns the spectrum value.
            bounds (tuple): A tuple containing the minimum and maximum wavelength.
        """
        nn.Module.__init__(self)
        PlotableWavelength.__init__(self,bounds,"Intensity [1]")
        self.func = func
        self.bounds = bounds

    def forward(self,wl):
        """
        Calculate the spectrum for given wavelengths.
        Args:
            wl (torch.Tensor or float): Wavelength in μm.
        Returns:
            torch.Tensor: Spectrum value at the given wavelengths.
        """
        if not torch.is_tensor(wl):
            wl = torch.tensor(wl)
        
        vmin,vmax = self.bounds
        out = self.func(wl)
        
        if isinstance(out,float):
            return out*torch.ones_like(wl)
        if isinstance(out,np.ndarray):
            out = torch.tensor(out,device=wl.device,dtype=wl.dtype)
        
        if (vmin > wl).any():
            out[vmin > wl] = 0.0
        if (wl>vmax).any():
            out[wl>vmax] = 0.0

        if torch.is_tensor(out):
            if len(out.shape) == 0:
                return out*torch.ones_like(wl)
        return out

class VisibleSunlight_am15g(Spectrum):
    """
    A class to represent the AM 1.5 G spectrum.
    This class uses the pvlib library to calculate the spectrum.
    """
    def __init__(self):
        def func(wl):
            device = wl.device
            dtype = wl.dtype
            wl = wl.detach().cpu().numpy()
            out = pvlib.spectrum.get_am15g(wl*1000.)
            out = np.array(out)
            out = torch.tensor(out,device=device,dtype=dtype)
            return out 
        super().__init__(func,[0.360,0.780])