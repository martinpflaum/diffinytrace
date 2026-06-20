Diffinytrace Documentation
==========================

**DiffinyTrace** is a Python library for differentiable ray tracing and optical system optimization using PyTorch. It enables automatic differentiation through optical systems, making it possible to optimize lens designs, mirror configurations, and other optical components using gradient-based methods.

The source code is available at the `GitHub repository <https://github.com/martinpflaum/diffinytrace>`_.


Key Features
------------

.. figure:: _static/system_3D_plot.png
   :width: 60%
   :align: center
   :alt: Transformation example

   **Flexible Transformations** — apply general transformations such as rotations and translations to optical components, with full control over the parameters and their role in the transformation.

.. figure:: _static/cad_export.png
   :width: 40%
   :align: center
   :alt: CAD export example

   **Seamless CAD Export** — generate lenses and mirrors that can be exported to
   standard CAD file formats.

.. figure:: _static/bspline_plot1.png
   :width: 80%
   :align: center
   :alt: B-spline surface example

   **Freeform Surfaces** — design complex optical elements with advanced
   B-spline representations for maximum flexibility.

* **Differentiable Ray Tracing**: Full automatic differentiation support through optical systems
* **Constraint Optimization**: Advanced optimization with PyTorch and SciPy integration
* **Illumination Design**: Algorithms for computing lens surfaces to achieve desired illumination profiles
* **GPU Acceleration**: CUDA support for high-performance computations


Installation
------------

1. **Create a new Environment** via conda:

   .. code-block:: bash

      conda create -n dit python==3.12

   Activate the environment via:

   .. code-block:: bash

      conda activate dit

   Install pip:

   .. code-block:: bash

      conda install pip

2. **Install PyTorch**

   Check your CUDA version with:

   .. code-block:: bash

      nvcc --version

   DiffinyTrace has only been tested with 2.10.0+cu130. Make sure to install the appropriate version of PyTorch for your system. You can find the installation instructions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_. DiffinyTrace should work for both CPU and CUDA versions.

3. **Install DiffinyTrace**

   Install all other dependencies and the library itself via:

   .. code-block:: bash

      pip install diffinytrace

   Or directly in the folder via:

   .. code-block:: bash

      pip install -r requirements.txt

Basic Usage Example
-------------------

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
   index_quick_start
   examples
   references

License
-------

DiffinyTrace is licensed under the MIT License. See the repository for full license details.
