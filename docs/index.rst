Diffinytrace Documentation
==========================

**DiffinyTrace** is a Python library for differentiable ray tracing and optical system optimization using PyTorch. It enables automatic differentiation through optical systems, making it possible to optimize lens designs, mirror configurations, and other optical components using gradient-based methods.

Key Features
------------


* **Differentiable Ray Tracing**: Full automatic differentiation support through optical systems
* **Freeform Surfaces**: Advanced B-spline surface modeling for complex optical geometries
* **CAD Export**: Export lenses and mirrors to popular CAD file formats
* **Constraint Optimization**: Advanced optimization with PyTorch and SciPy integration
* **Illumination Design**: Algorithms for computing lens surfaces to achieve desired illumination profiles
* **GPU Acceleration**: CUDA support for high-performance computations


Quick Start
-----------

Install DiffinyTrace:

.. code-block:: bash

   pip install diffinytrace

Basic usage example:

.. code-block:: python

   import diffinytrace as dit
   import torch
   NBK7 = dit.materials["NBK7"]

   wave_len = 1.024
   light_transform = dit.transforms.Offset(torch.tensor([0.0,0.0,0.0]))
   source = dit.source.CollimatedMonochromatic(light_transform,8.0,wave_len)

   plane_surface = dit.Plane()
   surface2 = dit.Aspheric(-1/50.)
   transf1 = dit.transforms.Distance(10.0,parent_transform=source)
   lens1 = dit.Lens(transf1,5.,plane_surface,surface2,NBK7,13.0)
   transf2 = dit.transforms.Distance(15.0,parent_transform=lens1)
   detector = dit.Detector(transf2,plane_surface,8.0)
   system = dit.SequentialOpticalSystem({"source":source, "lens":lens1, "detector":detector})
   #dit.plotting.system3D.plot(system,resolution=10)
   x,weights = source.sample(10)
   O,D,wave_len,_,meta_data = system(x,["source","lens","detector"])
   dit.plotting.system2D.plot(system,meta_data)

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents
   
   index_base
   index_utilities
   examples
   references

Getting Help
------------

* **GitHub Repository**: `https://github.com/yourusername/diffinytrace`
* **Issues & Bug Reports**: Use the GitHub issue tracker
* **Examples**: See the :doc:`examples` section for complete tutorials

License
-------

DiffinyTrace is licensed under the MIT License. See the repository for full license details.
