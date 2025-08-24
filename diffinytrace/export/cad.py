"""
This module provides functions for exporting CAD data to different formats.
It includes functions for exporting to STEP, IGES, and STL formats.
"""

# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "lens_to_solid",
    "extract_knots_and_multiplicities",
    "makeNurbsFace",
    "makeBsplineFace",
    "export_lens",
]

import torch
import cadquery as cq
from collections import Counter
from typing import List, Tuple
import copy
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCP.Message import Message, Message_Gravity

for printer in Message.DefaultMessenger_s().Printers():
    printer.SetTraceLevel(Message_Gravity.Message_Fail)

from OCP.Precision import Precision
from OCP.Geom import Geom_BSplineSurface

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
)
import numpy as np


def lens_to_solid(lens,
                  resolution: int,
                  tol:float = 0.001,
                  smoothing = None,
                  minDeg: int = 1,
                  maxDeg: int = 3) -> cq.Solid:
    """
    Convert a lens object to a CAD solid.
    
    Args:
        lens: The lens object to be converted.
        resolution (int): Resolution of the CAD model.
        tol (float): Tolerance for the CAD model.
        smoothing: Smoothing parameter for the CAD model.
        minDeg (int): Minimum degree for the B-spline surface.
        maxDeg (int): Maximum degree for the B-spline surface.
    Returns:
        cq.Solid: The CAD solid representing the lens.
    """
    #direction
    dtype = torch.float
    device = "cpu"
    lens = copy.deepcopy(lens)
    lens.to(dtype)
    lens.to(device)
    
    local_direction = torch.tensor([0.,0.,1.0],dtype=dtype,device=device)
    transform = lens.surface1.get_transform()
    direction = transform.to_global_dir(local_direction).detach().cpu().numpy()

    lens_thickness = lens.lens_thickness.detach().cpu().numpy()
    surface1,surface2 = lens.surface1,lens.surface2

    face1 = surface1.get_CAD_face(resolution,tol=tol,smoothing=smoothing,minDeg=minDeg,maxDeg=maxDeg)
    face2 = surface2.get_CAD_face(resolution,tol=tol,smoothing=smoothing,minDeg=minDeg,maxDeg=maxDeg)

    if lens.is_square:
        outer_face = cq.Face.makeRuledSurface(face1.outerWire(), face2.outerWire())
        shell = cq.Shell.makeShell([outer_face,face1,face2])
        solid = cq.Solid.makeSolid(shell)
        return solid
        
    lens_thickness = lens_thickness*5.
    affine_matrix = lens.surface1.get_transform().get_transformation_matrix().detach().cpu().numpy()
    off_vec = affine_matrix[:3, 3]
    off_vec = off_vec-lens_thickness*direction
    # Extract rotation (and scale)
    cylinder_solid = cq.Solid.makeCylinder(lens.aperture_radius, lens_thickness*2., cq.Vector(off_vec[0],off_vec[1],off_vec[2]), cq.Vector(direction[0],direction[1],direction[2]))
    #if not lens.is_square:
    face1 = face1.intersect(cylinder_solid)
    face2 = face2.intersect(cylinder_solid)
    
    face1_edge = [edge for edge in face1.edges()]
    face2_edge = [edge for edge in face2.edges()]
    outer_faces = []
    for k in range(len(face1_edge)):
        outer_faces += [cq.Face.makeRuledSurface(face1_edge[k],face2_edge[k])]

    shell = cq.Shell.makeShell(outer_faces+[face1,face2])
    solid = cq.Solid.makeSolid(shell)
    return solid


def extract_knots_and_multiplicities(knots: List[float]) -> Tuple[List[float], List[int]]:
    """
    Extract unique knots and their multiplicities from a knot vector.
    
    Args:
        knots (List[float]): The knot vector with implicit multiplicities.

    Returns:
        unique_knots (List[float]): A list of unique knots.
        multiplicities (List[int]): A list of multiplicities corresponding to the unique knots.
    """    
    unique_knots = []
    multiplicities = []
    
    # Count occurrences of each knot
    knot_counts = Counter(knots)
    
    # Sort the knots to ensure increasing order
    sorted_knots = sorted(knot_counts.items())
    
    for knot, count in sorted_knots:
        unique_knots.append(knot)
        multiplicities.append(count)
    
    return unique_knots, multiplicities


def makeNurbsFace(
    control_points,
    weights,
    U1,
    U2,
    u_degree: int,
    v_degree: int,
    u_periodic: bool = False,
    v_periodic: bool = False
)->cq.Face:
    """
    Create a B-spline surface from control points, weights, and implicit knot vectors.

    Args:
        control_points (list): 2D list of control points as Vectors.
        weights (list): 2D list of weights for each control point.
        U1 (list): Knot vector in U direction with implicit multiplicities.
        U2 (list): Knot vector in V direction with implicit multiplicities.
        u_degree (int): Degree of the B-spline in the U direction.
        v_degree (int): Degree of the B-spline in the V direction.
        u_periodic (bool): If True, makes the surface periodic in the U direction.
        v_periodic (bool): If True, makes the surface periodic in the V direction.

    Returns:
        cq.Face: Face instance representing the B-spline surface.
    """
    U1 = [float(elem) for elem in U1]
    U2 = [float(elem) for elem in U2]
    # Extract unique knots and multiplicities for U and V directions
    u_knots, u_mults = extract_knots_and_multiplicities(U1)
    v_knots, v_mults = extract_knots_and_multiplicities(U2)
    # Initialize control points array
    num_u = len(control_points)
    num_v = len(control_points[0])
    poles = TColgp_Array2OfPnt(1, num_u, 1, num_v)
    for i, row in enumerate(control_points):
        for j, _pt in enumerate(row):
            pt = cq.Vector(_pt[0],_pt[1],_pt[2])
            poles.SetValue(i + 1, j + 1, pt.toPnt())
    
    # Initialize weights array
    weights_array = TColStd_Array2OfReal(1, num_u, 1, num_v)
    for i, row in enumerate(weights):
        for j, w in enumerate(row):
            weights_array.SetValue(i + 1, j + 1, float(w))
    
    # Initialize knot arrays for U and V directions
    u_knots_array = TColStd_Array1OfReal(1, len(u_knots))
    v_knots_array = TColStd_Array1OfReal(1, len(v_knots))
    u_mults_array = TColStd_Array1OfInteger(1, len(u_mults))
    v_mults_array = TColStd_Array1OfInteger(1, len(v_mults))
    
    for idx, val in enumerate(u_knots):
        u_knots_array.SetValue(idx + 1, val)
    for idx, val in enumerate(v_knots):
        v_knots_array.SetValue(idx + 1, val)
    for idx, mult in enumerate(u_mults):
        u_mults_array.SetValue(idx + 1, mult)
    for idx, mult in enumerate(v_mults):
        v_mults_array.SetValue(idx + 1, mult)
    
    # Create the B-spline surface
    spline_surface = Geom_BSplineSurface(
        poles,
        weights_array,
        u_knots_array,
        v_knots_array,
        u_mults_array,
        v_mults_array,
        u_degree,
        v_degree,
        u_periodic,
        v_periodic
    )
    
    # Create a face from the B-spline surface
    face = BRepBuilderAPI_MakeFace(spline_surface, Precision.Confusion_s()).Face()

    # Return an instance of Face initialized with the generated face
    return cq.Face(face)

def makeBsplineFace(
    control_points,
    U1,
    U2,
    u_degree: int,
    v_degree: int,
    u_periodic: bool = False,
    v_periodic: bool = False
)->cq.Face:
    """
    Create a non-rational B-spline surface face from control points and knot vectors.

    Args:
        control_points (np.ndarray): 2D array of control points (shape: [num_u, num_v, 3]).
        U1 (list): Knot vector in U direction with implicit multiplicities.
        U2 (list): Knot vector in V direction with implicit multiplicities.
        u_degree (int): Degree of the B-spline in the U direction.
        v_degree (int): Degree of the B-spline in the V direction.
        u_periodic (bool, optional): If True, makes the surface periodic in the U direction. Defaults to False.
        v_periodic (bool, optional): If True, makes the surface periodic in the V direction. Defaults to False.

    Returns:
        cq.Face: Face instance representing the B-spline surface.
    """

    weights = np.ones((control_points.shape[0],control_points.shape[1]),dtype=float)
    return makeNurbsFace(control_points,weights,U1,U2,u_degree,v_degree,u_periodic,v_periodic)

def export_lens(file_path:str, lens, resolution:int, tol:float = 0.001):
    """
    Export a lens to a CAD file.
    
    Args:
        file_path (str): The path to save the CAD file.
        lens: The lens object to be exported.
        resolution (int): Resolution of the CAD model.
        tol (float): Tolerance for the CAD model.
    """
    
    solid = lens_to_solid(lens,resolution,tol=tol)
    cq.exporters.export(solid, file_path)


