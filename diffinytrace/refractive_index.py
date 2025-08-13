# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


__all__ = [
    "RefractiveIndex",
    "materials"
]

import torch.nn as nn
import torch
import numpy as np
from .plotting.wavelength import PlotableWavelength

class RefractiveIndex(nn.Module,PlotableWavelength):
    r"""This class is used to calculate the refractive index of a material.
    
    At material interfaces, the transmitted direction :math:`\mathbf{D'}` is computed based on the surface normal 
    :math:`\mathbf{N} = \nabla s / \|\nabla s\|` and the incident direction :math:`\mathbf{D}`, using Snell's law (see :cite:`do`):

    .. math::

        \mathbf{D'} = \mathbf{N} \sqrt{1 - (1 - \cos^2 \psi_i) \eta^2} + \eta (\mathbf{D} - \mathbf{N} \cos \psi_i),

    where :math:`\cos \psi_i = \mathbf{D} \cdot \mathbf{N}` and :math:`\eta = n / n'` is the ratio of the refractive indices 
    of the two materials.

    We have implemented the refractive indices as a class that is initialized by a refractive index function and the start 
    and end wavelengths for which the function is valid. This makes it very convenient to use with the RefractiveIndex.info 
    database of optical constants (see :cite:`polyanskiy2024refractiveindex`), since this database often provides Python 
    functions for wavelength-dependent refractive indices.

    Example:
        Below is an example of how to set up an optical material in our ray tracing library:

        >>> import diffinytrace as dit
        >>> BaSF = dit.RefractiveIndex(
        >>>     lambda x: (1 + 1.65554268 / (1 - 0.0104485644 / x**2) +
        >>>                   0.17131977 / (1 - 0.0499394756 / x**2) +
        >>>                   1.33664448 / (1 - 118.961472 / x**2))**0.5,
        >>>     [0.365, 2.5]
        >>> )
        >>> dit.plotting.wavelength.plot(
        >>>     BaSF, title="Refractive index of BaSF (Barium dense flint)"
        >>> )

    Args:
        func (callable): A function that takes a wavelength in μm and returns the refractive index.
        bounds (tuple): A tuple containing the minimum and maximum wavelength in μm.
    """
    def __init__(self,func,bounds):
        nn.Module.__init__(self)
        PlotableWavelength.__init__(self,bounds,"n [1]")
        self.func = func
        self.bounds = bounds

    def forward(self,wl):
        """Calculates the refractive index for given wavelengths.
        Args:
            wl (torch.Tensor or float): Wavelength in μm.
        Returns:
            torch.Tensor: Refractive index at the given wavelengths.
        """
        if not torch.is_tensor(wl):
            wl = torch.tensor(wl)
        vmin,vmax = self.bounds
        if not (((vmin <=wl).float()*(wl<=vmax).float())==1.0).all():
            print(f"The wavelength should be given in μm and between {vmin} and {vmax}. Fallback to constant val.")


        out = self.func(wl)
        if isinstance(out,float):
            return out*torch.ones_like(wl)
        if isinstance(out,np.ndarray):
            out = torch.tensor(out,device=wl.device,dtype=wl.dtype)

        out[vmin > wl] = self.func(vmin)
        out[wl>vmax] = self.func(vmax)
         
        if torch.is_tensor(out):
            if len(out.shape) == 0:
                return out*torch.ones_like(wl)
        return out

"""
All material data is from https://refractiveindex.info/. Please verify the equation and ranges by ur self and the references.
"""
materials = {
    "NONE": RefractiveIndex(lambda x: 1.0,(0.0,torch.inf)),
    "AIR": RefractiveIndex(lambda x: 1+0.05792105/(238.0185-x**-2)+0.00167917/(57.362-x**-2),(0.23,1.69)),#P. E. Ciddor. Refractive index of air: new equations for the visible and near infrared, Appl. Optics 35, 1566-1573 (1996)
    "HELIUM": RefractiveIndex(lambda x: (1+4977.77e-8/(1-28.54e-6/x**2)+1856.94e-8/(1-7.76e-3/x**2))**.5,(0.48,2.06)),#C. R. Mansfield and E. R. Peck. Dispersion of helium, J. Opt. Soc. Am. 59, 199-203 (1969)
    "PMMA": RefractiveIndex(lambda x: (1+0.99654/(1-0.00787/x**2)+0.18964/(1-0.02191/x**2)+0.00411/(1-3.85727/x**2))**.5,(0.405,1.08)),#Marcin Szczurowski
    "NBK7": RefractiveIndex(lambda x: (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**.5,(0.3,2.5)),#SCHOTT
    "BAF10": RefractiveIndex(lambda x: (1+1.5851495/(1-0.00926681282/x**2)+0.143559385/(1-0.0424489805/x**2)+1.08521269/(1-105.613573/x**2))**.5,(0.35,2.5)),#SCHOTT
    "BAK1": RefractiveIndex(lambda x: (1+1.12365662/(1-0.00644742752/x**2)+0.309276848/(1-0.0222284402/x**2)+0.881511957/(1-107.297751/x**2))**.5,(0.3,2.5)),#SCHOTT
    "FK51A": RefractiveIndex(lambda x: (1+0.971247817/(1-0.00472301995/x**2)+0.216901417/(1-0.0153575612/x**2)+0.904651666/(1-168.68133/x**2))**.5,(0.29,2.5)),#SCHOTT
    "LASF9": RefractiveIndex(lambda x: (1+2.00029547/(1-0.0121426017/x**2)+0.298926886/(1-0.0538736236/x**2)+1.80691843/(1-156.530829/x**2))**.5,(0.365,2.5)),#SCHOTT
    "SF5": RefractiveIndex(lambda x: (1+1.52481889/(1-0.011254756/x**2)+0.187085527/(1-0.0588995392/x**2)+1.42729015/(1-129.141675/x**2))**.5,(0.37,2.5)),#SCHOTT
    "SF10": RefractiveIndex(lambda x: (1+1.62153902/(1-0.0122241457/x**2)+0.256287842/(1-0.0595736775/x**2)+1.64447552/(1-147.468793/x**2))**.5,(0.38,2.5)),#SCHOTT
    "SF11": RefractiveIndex(lambda x: (1+1.73759695/(1-0.013188707/x**2)+0.313747346/(1-0.0623068142/x**2)+1.89878101/(1-155.23629/x**2))**.5,(0.37,2.5)),#SCHOTT
}
