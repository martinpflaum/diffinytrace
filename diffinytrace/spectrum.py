"""
MIT License

Copyright (c) 2025 Martin Pflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import torch
import torch.nn as nn
import numpy as np
import pvlib
from .plotting.wavelength import PlotableWavelength



class Spectrum(nn.Module,PlotableWavelength):
    def __init__(self,func,bounds):
        nn.Module.__init__(self)
        PlotableWavelength.__init__(self,bounds,"Intensity [1]")
        self.func = func
        self.bounds = bounds

    def forward(self,wl):
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