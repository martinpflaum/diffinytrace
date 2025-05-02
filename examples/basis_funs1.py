#%%
import sys
import os

sys.path.insert(0, os.path.abspath(".."))

import diffinytrace as dit
from diffinytrace.basis_functions.bspline import basis_2d
import torch


U1 = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
Us = [U1, U1]
ps = [3, 3]
ns = [3, 3]

side_points = 100
_x = torch.linspace(0, 1, side_points)
_y = torch.linspace(0, 1, side_points)
grid_y, grid_x = torch.meshgrid(_y, _x, indexing='ij')
points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], dim=-1)

N2D = basis_2d(points,Us, ps, ns,torch.tensor([0,1]),torch.tensor([0,1]))

xi = 0
yi = 2
dit.plotting.quantity2D.plot(
    N2D[:, yi, xi].reshape(side_points, side_points),
    "basis fun",
    [0, 1],
    [0, 1],
    xlabel="x",
    ylabel="y"
)
# %%
