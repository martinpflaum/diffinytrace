# Copyright (c) 2025 Martin Pflaum
# This file is part of the diffinytrace project, licensed under the MIT License.

__all__ = ["create_lens"]

import torch
import copy
import gc
from typing import List,Tuple,Optional
import tqdm
from diffinytrace import (
    source, transforms, Bspline, Plane, Lens, Detector, SequentialOpticalSystem,
    utils, plotting, target_grid, render, minimize, set_unused_bspline_coeff_to_nearest,
    export, gaussian_smoother
)
import diffinytrace as dit
#output_step_file_name, #this file should end with .step
#html_plot_file_name=""
    
def create_lens(
    image_file_name:str,
    lens_material:dit.RefractiveIndex,
    air_material:dit.RefractiveIndex,
    device,
    sigma:float,
    aperture_radius_source:float=21.,
    aperture_radius_lens:float=25.,
    lens_thickness:float=5.,
    detector_distance:float=150.,
    lens_distance:float=1.0,
    num_refinements:int=5,
    image_padding:float=0.2,
    bspline_orders:List[int]=[3, 3],
    bspline_ns_start:List[int]=[4, 4],
    use_desired_irradiance_smoothing:bool=True,
    num_rays:int=2**16,
    grid_size:int=300,
    minimization_method:str='L-BFGS-B',
    ):

    r"""
    Creates and optimizes a lens system to match a desired irradiance distribution from an input image.

    Args:
        image_file_name (str): Path to the image file representing the desired irradiance.
        lens_material: Material of the lens.
        air_material: Material for air (should be dit.materials["NONE"]).
        device: Device for computation ('cpu' or 'cuda:0').
        aperture_radius_source (float, optional): Half diameter of the source aperture in mm. Defaults to 21.
        aperture_radius_lens (float, optional): Half diameter of the lens aperture in mm. Defaults to 25.
        lens_thickness (float, optional): Thickness of the lens in mm. Defaults to 5.
        detector_distance (float, optional): Distance from lens to detector in mm. Defaults to 150.
        lens_distance (float, optional): Distance from source to lens in mm. Defaults to 1.0.
        num_refinements (int, optional): Number of refinement steps for optimization. Defaults to 5.
        sigma (float, optional): Standard deviation for Gaussian smoothing. Controls expected resolution. Defaults to 1.0.
        image_padding (float, optional): Padding around the image for the detector aperture. Defaults to 0.2.
        bspline_orders (list, optional): B-spline orders for lens surface smoothing [order_x, order_y]. Defaults to [3, 3].
        bspline_ns_start (list, optional): Initial number of B-spline elements in x and y directions. Defaults to [4, 4].
        use_desired_irradiance_smoothing (bool, optional): Whether to use smoothing for desired irradiance. Defaults to True.
        num_rays (int, optional): Number of rays to trace for simulation. Defaults to 2**16.
        grid_size (int, optional): Number of grid points per dimension for Gaussian measurement functions. Defaults to 300.
        minimization_method (str, optional): Optimization method for minimization. Defaults to 'L-BFGS-B'.

    Returns:
        dict: Results containing optimization history, settings, and irradiance maps.
    """

    gc.collect()

    num_rays = copy.deepcopy(num_rays)

    if (image_padding==0.0):
        raise ValueError("Please don't set image_padding to 0.0.")

    ns_start = copy.deepcopy(bspline_ns_start)
    orders = copy.deepcopy(bspline_orders)
    
    light_transform = transforms.Offset(torch.tensor([0.0,0.0,0.0]))
    light_transform.pos.requires_grad = False

    light_source = None
    
    light_source = source.VisibleSunlightSimpleMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=1.0)
    #light_source = source.CollimatedMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power,is_square=True)

    lens_transform = transforms.Distance(lens_distance,parent_transform=light_transform)

    aperture_radius_detector = aperture_radius_source*(1+image_padding)
    
    bspline_surface1 = Bspline(aperture_radius_lens,orders,ns_start)
           
    surface1 = Plane()
    lens1 = Lens(lens_transform,lens_thickness,surface1,bspline_surface1,lens_material,aperture_radius_lens,is_square=True)
    lens_transform.distance.requires_grad = False
    lens1.lens_thickness.requires_grad = False
    detector_transform = transforms.Distance(detector_distance,parent_transform=lens1)#25.0+0.5
    detector_transform.distance.requires_grad = False
    plane_surface = Plane()
    detector = Detector(detector_transform,plane_surface,aperture_radius_detector)
    system = SequentialOpticalSystem({"source":light_source,"lens":lens1,"detector":detector},air_material)
    sequence = ["source","lens","detector"]
    system.to(device)
    irr_func = utils.irradiance_importer.create_irradiance_from_image_square(image_file_name,image_padding,0.,aperture_radius_detector,shape=[grid_size,grid_size])
    #plotting.quantity2D.plot(irr_func,"Desired Irradiance Distribution",cmap="grey",x_range=[-aperture_radius_detector,aperture_radius_detector])
    
    def get_desired_irradiance_raw():
        resolution = grid_size
        x_range = [-aperture_radius_detector,aperture_radius_detector]
        y_range = x_range
        _y = torch.linspace(*x_range,resolution)
        _x = torch.linspace(*y_range,resolution)
        mesh = torch.meshgrid(_y,_x)
        y = mesh[0].reshape(-1)
        x = mesh[1].reshape(-1)
        O = torch.zeros((x.shape[0],2))        
        O[:,0] = x
        O[:,1] = y
        val = irr_func
        val = val(O).reshape(resolution,resolution)
        if torch.is_tensor(val):
            val = val.detach().cpu().numpy()
        return val
    
    desired_irradiance_raw =  get_desired_irradiance_raw()
    
    
    def create_lens_copy():
        lens1_copy = copy.deepcopy(lens1)
        lens1_copy = lens1_copy.cpu()
        lens1_copy.n_func = None
        lens1_copy.surface1.n_func = None
        lens1_copy.surface2.n_func = None
        
        return lens1_copy
    
    """def create_html_plot(prefix):
        from ...plotting.system3D import plot
        x,_ = light_source.sample(15)
        x = x.to("cuda")
        O,D,wave_len,_,meta_data = system(x,sequence)
        print("is valid all==",meta_data["valid"].all())
        _html_plot_file_name = html_plot_file_name[:(len(html_plot_file_name)-len(".html"))]
        plot(system,meta_data,html_file_name=_html_plot_file_name+prefix+"_system.html",show=False)
        lens_copy = create_lens_copy()
        lens_copy._transform1 = transforms.Offset([0.,0.,0.]) 
        lens_copy._transform2.parent_transform = lens_copy._transform1
        plot(lens_copy,html_file_name=_html_plot_file_name+prefix+"_lens.html",show=False)
    """    
    #create_html_plot("initial")
    
    smoother = gaussian_smoother.GaussianSmootherSquare(aperture_radius_detector,
                                    grid_size=grid_size,
                                    sigma=sigma,
                                    desired_irradiance_fun=irr_func,
                                    smoothed_num_integration_points=2**22,
                                    smoothed_num_splits=10,
                                    device=device)
    
    """def run_ray_tracer_smooth(smoother):
        with torch.no_grad():
            out = render.smoothed_irradiance(system,sequence,light_source,detector,smoother,num_rays=num_rays,device=device,method_ray_tracing=method_ray_tracing)
            out = out.detach().cpu().numpy()
            return out

    
    
    def run_ray_tracer_none_smooth_eval(smoother):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            grid = smoother.grid_eval
            out = render.binned_irradiance(system,sequence,light_source,detector,grid,device=device,num_rays=num_rays_save_irradiance,method_ray_tracing=method_ray_tracing)
            out = out.detach().cpu().numpy()
            return out
    """
    """
    initial_smooth_irradiance = None
    initial_binned_irradiance = None
    initial_smoothed_desired_irradiance = None
    initial_discrete_desired_irradiance = None
    initial_desired_none_smooth_irradiance_eval = None
    initial_binned_irradiance_eval = None
    
    if save_irradiance_results:
        initial_smooth_irradiance = run_ray_tracer_smooth(smoother)
        initial_binned_irradiance_eval = run_ray_tracer_none_smooth_eval(smoother)
        initial_binned_irradiance = initial_binned_irradiance_eval
        
        initial_smoothed_desired_irradiance = smoother.smoothed_desired_irradiance.detach().cpu().numpy()
        initial_discrete_desired_irradiance = smoother.discrete_desired_irradiance.detach().cpu().numpy()
        initial_desired_none_smooth_irradiance_eval = smoother.desired_none_smooth_irradiance_eval.detach().cpu().numpy()
    """
    merit_func = gaussian_smoother.make_merit_function(system,
                        sequence,
                        light_source,
                        detector,
                        smoother,
                        num_rays,
                        method_ray_tracing="sobol_pow2",
                        use_desired_irradiance_smoothing=use_desired_irradiance_smoothing,
                        device=device)

    eval_func = gaussian_smoother.make_evaluation_function(system,
                        sequence,
                        light_source,
                        detector,
                        smoother,
                        num_splits=20,
                        num_rays_per_split=500000,
                        #method_ray_tracing="monte_carlo",
                        device=device)


    def minimization_call(k):

        convergence_list = []
        rmse_list = []
        ssim_list = []


        def make_convergence_callback():
            def return_func():
                L2_error,rmse,ssim_error = eval_func()
                convergence_list.append(L2_error.detach().cpu().numpy())
                rmse_list.append(rmse.detach().cpu().numpy())
                ssim_list.append(ssim_error.detach().cpu().numpy())
            
            return return_func 

        out = minimize(merit_func,system.parameters(),callback=make_convergence_callback(),method=minimization_method,save_history=True,call_before_minimize=True)
        
        out["sigma"] = smoother.sigma
        out["coeff_shape"] = bspline_surface1.coeff.shape
        out["loop_number"] = k
        
        out["history"]["convergence"] = convergence_list
        out["history"]["rmse"] = rmse_list
        out["history"]["ssim"] = ssim_list
        out["smoothed_desired_irradiance"] = smoother.smoothed_desired_irradiance.detach().cpu().numpy()
        out["discrete_desired_irradiance"] = smoother.discrete_desired_irradiance.detach().cpu().numpy()

        
        print("last_merit",merit_func(),out["history"]["fun_vals"][-1])
        print("last_error",eval_func(),convergence_list[-1])

        out["smooth_irradiance"] = smoother.last_smoothed_irradiance.detach().cpu().numpy()
        out["binned_irradiance_eval"] = smoother.last_raycounting.detach().cpu().numpy()

        #if save_lens_history:
        #    out["lens"] = create_lens_copy() 
        return out   
        
    def run_minimization_loop_classical():
        results_minimize = [] 
        for k in range(num_refinements):
            print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
            results_minimize += [minimization_call(k)]    
            print("END")
            bspline_surface1.refine()
        print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
        results_minimize += [minimization_call(num_refinements)]    
        print("END")
        return results_minimize

    gc.collect()
    results_minimize = run_minimization_loop_classical()

    """if post_process_lens:
        set_unused_bspline_coeff_to_nearest(system,sequence,light_source,bspline_surface1,num_rays=num_rays,method_ray_tracing=method_ray_tracing)

    if output_step_file_name is not None:
        if output_step_file_name != "":
            export_resolution = 256 #has no influence because we are using bsplines!
            export.cad.export_lens(output_step_file_name,lens1,export_resolution)
    """



    #final_lens = create_lens_copy()
    
    gc.collect()
    # create_html_plot("final")
    gc.collect()
    
    def get_final_irr_results():
        out = {}
        merit_func()
        eval_func()
        out["smooth_irradiance"] = smoother.last_smoothed_irradiance.detach().cpu().numpy()
        out["binned_irradiance"] = smoother.last_raycounting.detach().cpu().numpy()
        out["smoothed_desired_irradiance"] = smoother.smoothed_desired_irradiance.detach().cpu().numpy()
        out["desired_none_smooth_irradiance"] = smoother.discrete_desired_irradiance.detach().cpu().numpy()
        return out

    """def get_initial_irr_results():
        out = {}
        out["smooth_irradiance"] = initial_smooth_irradiance
        out["binned_irradiance"] = initial_binned_irradiance
        out["binned_irradiance_eval"] = initial_binned_irradiance_eval
        out["smoothed_desired_irradiance"] = initial_smoothed_desired_irradiance
        out["discrete_desired_irradiance"] = initial_discrete_desired_irradiance
        out["desired_none_smooth_irradiance_eval"] = initial_desired_none_smooth_irradiance_eval
        return out
    """
    def get_settings():
        out={}
        out["aperture_radius_detector"] = aperture_radius_detector
        out["aperture_radius_lens"] = aperture_radius_lens
        out["aperture_radius_source"] = aperture_radius_source
        out["num_refinements"] = num_refinements
        out["image_padding"] = image_padding
        out["sigma_final"] = sigma
        out["lens_distance"] = lens_distance
        out["lens_thickness"] = lens_thickness
        out["detector_distance"] = detector_distance
        out["desired_irradiance_raw"] = desired_irradiance_raw
        out["minimization_method"] = minimization_method
        out["num_rays_opti"] = num_rays
        out["bspline_orders"] = bspline_orders
        out["bspline_ns_start"] = bspline_ns_start
        out["grid_size"] = grid_size
        out["use_desired_irradiance_smoothing"] = use_desired_irradiance_smoothing
        return out
    

    def get_surface_data(lens,resolution):
        aperture_radius = lens.aperture_radius
        surface = lens.surface2.surface

        _x = torch.linspace(-aperture_radius,aperture_radius,resolution)
        _y = torch.linspace(-aperture_radius,aperture_radius,resolution)
        mesh = torch.meshgrid(_x,_y)
        x = mesh[0].reshape(-1)
        y = mesh[1].reshape(-1)
        O = torch.zeros((x.shape[0],2))        
                
        O[:,0] = x
        O[:,1] = y
        z = None
        O = O.detach().to(device=device)
        O.requires_grad_(True)  # Enable gradients

        z = surface.explicit(O)

        dz_dx, dz_dy = None, None
        if torch.is_tensor(z) and z.requires_grad:
            dz = torch.autograd.grad(
                z, O,
                grad_outputs=torch.ones_like(z)
            )[0]
            dz_dx = dz[:, 0].reshape(resolution, resolution).cpu().detach().numpy()
            dz_dy = dz[:, 1].reshape(resolution, resolution).cpu().detach().numpy()    

        z = z.cpu().detach().reshape(resolution,resolution).numpy()
        z = z.T
        dz_dx = dz_dx.T
        dz_dy = dz_dy.T
        return z,dz_dx,dz_dy

    out = {}
    out["results_minimize"] = results_minimize
    z,dz_dx,dz_dy = get_surface_data(lens1,256)
    out["lens_offset"] = {"z": z, "dz_dx": dz_dx, "dz_dy": dz_dy}
    #if save_lens_history:
    #    out["final_lens"] = final_lens
    
    out["settings"] = get_settings()
    #if save_irradiance_results:
    #    out["initial_irr_results"] = get_initial_irr_results()
    out["final_irr_results"] = get_final_irr_results()

    target_grid_high_res = target_grid.GridSquare(aperture_radius_detector,grid_size=grid_size*4)
    raycounting_list = []
    for k in tqdm.tqdm(range(1000)):
        tmp = render.binned_irradiance(optical_system=system,sequence=sequence,source=light_source,detector=detector,grid=target_grid_high_res,num_rays=1000000,method_ray_tracing="monte_carlo",device=device)
        tmp = tmp.detach().cpu()
        raycounting_list.append(tmp)
    raycounting = torch.mean(torch.stack(raycounting_list),dim=0).detach().cpu()
    out["high_res_irradiance"] = raycounting
    return out
