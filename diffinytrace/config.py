# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.


# Default settings for ray intersection parameters
tolerance = 1e-6  # Default tolerance for ray intersection
max_iterations = 100  # Default maximum number of iterations for the solver
damping_factor = 1.0  # Default damping factor for the Newton method (1.0 means no damping)
show_iteration_count = False  # Default setting to not show the number of iterations

def set_show_iteration_count(flag):
    """
    Set the option to show the number of iterations for each intersection.
    
    Parameters:
    flag (bool): True to show the number of iterations, False otherwise.
    """
    global show_iteration_count
    show_iteration_count = flag

def get_show_iteration_count():
    """Check if the number of iterations should be shown."""
    return show_iteration_count

def set_tolerance(new_tolerance):
    global tolerance
    tolerance = new_tolerance

def get_tolerance():
    return tolerance

def set_max_iterations(new_max_iterations):
    global max_iterations
    max_iterations = new_max_iterations

def get_max_iterations():
    return max_iterations

def set_damping_factor(new_damping_factor):
    """
    Set the damping factor for the Newton method used in ray intersections.
    
    Parameters:
    new_damping_factor (float): The new damping factor (0 < new_damping_factor <= 1).
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
    """Reset to the default configuration settings for the ray tracer."""
    global tolerance, max_iterations, damping_factor, show_iteration_count
    tolerance = 1e-6
    max_iterations = 100
    damping_factor = 1.0
    show_iteration_count = False  # Reset to default (not showing)
