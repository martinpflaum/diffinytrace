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

"""import diffinytrace as dit


U1 = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
Us = [U1,U1]
ps = [3,3]
ns = [3,3]

side_points = 100
_x = torch.linspace(0,1,side_points)
_y = torch.linspace(0,1,side_points)
grid_y,grid_x = torch.meshgrid(_y, _x, indexing='ij')
points = torch.cat([grid_x.reshape(-1,1),grid_y.reshape(-1,1)],dim=-1)
N2D = bspline_basis_funs2D(Us,ps,ns,points)
N2D.shape

xi = 0
yi = 2 
dit.plotting.quantity2D.plot(N2D[:,yi,xi].reshape(side_points,side_points),"basis fun",[0,1],[0,1],xlabel="x",ylabel="y")

U = torch.tensor([0., 0.2, 0.4, 0.6, 0.8, 1])
n = 3
p=3#this is order 3
xis = torch.linspace(0,1,100)
xN = bspline_basis_funs1D(U,p,n,xis)

num_points = xN.shape[0]
tmp = xN.reshape(num_points,-1,1)*xN.reshape(num_points,1,-1)

for yin in xN.T:
    plt.plot(xis,yin)

plt.gca().set_aspect('equal')

"""


"""
import numpy as np
n = 4

control_points = torch.randn((n,2))#torch.tensor([[0.0,0.],[1.,1.0]])

#torch.randn((n,2))
p = 4 # Quadratic B-spline
U = torch.tensor([0.0]*(p-1) +list(np.linspace(0,1.0,n+p-2*(p-1)))+ [1.0]*(p-1))  # Knot vector
U = U.float()

#TODO check below thing in bspline
print(U.shape[0]-p==n,n>=p)
for m in range(100):
    U_new,new_control_points = bspline_insert1D(U,p, torch.rand((1)),control_points)
    print("new_control_points",new_control_points)
    print("control_points",control_points)

xis = torch.linspace(0,1,1000)
xN1 = bspline_basis_funs1D(U,p,3,xis)
out1 = xN1@control_points
xN2 = bspline_basis_funs1D(U_new,p,4,xis)
out2 = xN2@new_control_points
plt.plot(out1[:,0],out1[:,1],linewidth=5.0)
plt.plot(out2[:,0],out2[:,1],"--")
torch.mean((out1-out2)**2)
"""