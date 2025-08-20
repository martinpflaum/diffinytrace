"""
Configuration module for diffinytrace.

This module provides global configuration options for controlling ray intersection
solver behavior, such as tolerance, maximum iterations, damping factor, and whether
to display iteration counts. These settings can be adjusted at runtime to tune
performance and accuracy.

Example:
    >>> import diffinytrace.config as config
    >>> config.set_tolerance(1e-8)
    >>> config.set_show_iteration_count(True)
    >>> config.restore_default_settings()
"""

# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = [
    "set_show_iteration_count",
    "get_show_iteration_count",
    "set_tolerance",
    "get_tolerance",
    "set_max_iterations",
    "get_max_iterations",
    "set_damping_factor",
    "get_damping_factor",
    "restore_default_settings"
]

# Default settings for ray intersection parameters
tolerance:float = 1e-6  # Default tolerance for ray intersection
max_iterations:int = 100  # Default maximum number of iterations for the solver
damping_factor:float = 1.0  # Default damping factor for the Newton method (1.0 means no damping)
show_iteration_count:bool = False  # Default setting to not show the number of iterations

def set_show_iteration_count(flag):
    """
    Set the option to show the number of iterations for each intersection.

    Args:
        flag (bool): True to show the number of iterations, False otherwise.
    """
    global show_iteration_count
    show_iteration_count = flag

def get_show_iteration_count():
    """
    Check if the number of iterations should be shown.

    Returns:
        bool: True if the number of iterations should be shown, False otherwise.
    """
    return show_iteration_count

def set_tolerance(new_tolerance):
    """
    Set the tolerance for ray intersection calculations.

    Args:
        new_tolerance (float): The new tolerance value (must be > 0).

    Raises:
        ValueError: If `new_tolerance` is not greater than 0.
    """
    if new_tolerance <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    global tolerance
    tolerance = new_tolerance

def get_tolerance():
    """
    Get the current tolerance for ray intersection calculations.

    Returns:
        float: The current tolerance value.
    """
    return tolerance

def set_max_iterations(new_max_iterations):
    """
    Set the maximum number of iterations for the ray intersection solver.

    Args:
        new_max_iterations (int): The new maximum number of iterations (must be > 0).

    Raises:
        ValueError: If `new_max_iterations` is not greater than 0.
    """
    if new_max_iterations <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    global max_iterations
    max_iterations = new_max_iterations

def get_max_iterations():
    """
    Get the current maximum number of iterations for the ray intersection solver.

    Returns:
        int: The current maximum number of iterations.
    """
    global max_iterations
    return max_iterations

def set_damping_factor(new_damping_factor):
    """
    Set the damping factor for the Newton method used in ray intersections.

    Args:
        new_damping_factor (float): The new damping factor (0 < new_damping_factor <= 1).

    Raises:
        ValueError: If `new_damping_factor` is not between 0 and 1 (exclusive of 0, inclusive of 1).
    """
    global damping_factor
    if 0 < new_damping_factor <= 1:
        damping_factor = new_damping_factor
    else:
        raise ValueError("Damping factor must be between 0 and 1.")

def get_damping_factor():
    """
    Get the current damping factor for the Newton method.

    Returns:
        float: The current damping factor.
    """
    return damping_factor

def restore_default_settings():
    """
    Reset to the default configuration settings for the ray tracer.

    This will reset all configuration parameters to their default values.
    """
    global tolerance, max_iterations, damping_factor, show_iteration_count
    tolerance = 1e-6
    max_iterations = 100
    damping_factor = 1.0
    show_iteration_count = False  # Reset to default (not showing)
