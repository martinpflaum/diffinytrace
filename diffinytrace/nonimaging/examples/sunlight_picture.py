import torch
import copy
import gc
def get_all_sigmas_refined(_ns_start,_orders,_sigma_final,_num_refinements_plus):
    from ... import bspline_n_after_refinement
    n_current = _ns_start[0]
    order_k = _orders[0]
    all_n = [n_current]
    for k in range(_num_refinements_plus):
        n_current = bspline_n_after_refinement(n_current,order_k)
        all_n+=[n_current]


    sigma_current = _sigma_final

    all_sigma_rev = [sigma_current]
    for _k in range(_num_refinements_plus-1):
        k = _num_refinements_plus-_k
        ratio = ((all_n[k-1])/(all_n[k]))
        sigma_current = sigma_current*(1.0/ratio)
        all_sigma_rev += [sigma_current]

    all_sigma = all_sigma_rev[::-1] 
    return all_sigma
#= 'image_vertical.jpg'

def create_lens(\
    input_file_name, #the image file
    output_step_file_name, #this file should end with .step
    lens_material, 
    air_material, #this should be dit.materials["NONE"], since probaly a lot refractive indices are measurent in reference to air
    device, #device should be cpu or cuda:0 for gpu
    aperture_radius_source = 21., #TODO EDIT THIS this is the half diameter in [mm]
    aperture_radius_lens = 25., #TODO EDIT THIS this is the half diameter in [mm]
    lens_thickness = 5., #thickness of lens in [mm]
    detector_distance = 150., #distance from lens to detector in [mm]
    lens_distance = 1.0,
    num_refinements = 5, #!number of refinements is also very portant but performance critical 
    sigma_final = 1.0, #!sigma final is a very important option since this says what the maximal expected resolution would be sigma final should be in relation to the number of rays traced
    image_padding = 0.2, #image padding is important to have a well defined optical system
    etendue = True,
    bspline_orders = [3,3], #this option determines how smooth the lens surface is and thus also influences how smooth the irradiance is.
    bspline_ns_start = [4,4], #number of elements in x and y direction. bspline_ns_start = [4,4] is probably the best option always!
    use_sigma_refinement = True, #use_sigma_refinement=True will be the best option probably always!
    use_desired_irradiance_smoothing = True, #use_desired_irradiance_smoothing=True will be the best option probably always!
    use_power_correction = False,
    num_rays=2**16, #!this option performance critical. if sigma_final is low this option should be high.
    method_ray_tracing="sobol_pow2",
    num_conv_points=300, #number of gaussians used in one dimension - so 301x301 gaussian measurement functions used in this case
    residual_integration_method="midpoint", #integration method used for calulating the final error- should be sobol or midpoint, doenst really make much of a difference
    num_integration_points_desired=2**21,
    minimization_method='L-BFGS-B',
    post_process_lens = True,
    total_power=1.0, #TODO CHANGE NAMING CONVETION TO TOTAL_FLUX. you really dont need to change this option it's just has influence on the final plots of the irradiance. total_power is the energy per second in Watts!
    save_lens_history = False,
    save_history = False,
    save_irradiance_results=False,
    num_rays_save_irradiance = None,
    html_plot_file_name=""):
    r"""
    This function creates a lens from an image file and optimizes it using ray tracing.
    
    Args:
        input_file_name (str): The path to the image file.
        output_step_file_name (str): The path to save the lens as a STEP file.
        lens_material (str): The material of the lens.
        air_material (str): The material of the air.
        device (torch.device): The device to use for computation (CPU or GPU).
        aperture_radius_source (float): The radius of the source aperture in mm.
        aperture_radius_lens (float): The radius of the lens aperture in mm.
        lens_thickness (float): The thickness of the lens in mm.
        detector_distance (float): The distance from the lens to the detector in mm.
        lens_distance (float): The distance from the light source to the lens in mm.
        num_refinements (int): The number of refinements for the B-spline surface.
        sigma_final (float): The final sigma value for Gaussian smoothing.
        image_padding (float): The padding for the image.
        etendue (bool): Whether to use etendue or not.
        bspline_orders (list): The orders of the B-spline surface.
        bspline_ns_start (list): The initial number of elements in x and y direction for B-spline surface.
        use_sigma_refinement (bool): Whether to use sigma refinement or not.
        use_desired_irradiance_smoothing (bool): Whether to use desired irradiance smoothing or not.
        use_power_correction (bool): Whether to use power correction or not.
        num_rays (int): The number of rays to trace.
        method_ray_tracing (str): The method used for ray tracing ('sobol_pow2' or 'sobol').
        num_conv_points (int): The number of Gaussian measurement functions used in one dimension.
        residual_integration_method (str): The integration method used for calculating the final error ('sobol' or 'midpoint').
        num_integration_points_desired (list): The number of integration points for desired irradiance calculation.
        minimization_method (str): The method used for minimization ('L-BFGS-B').
        post_process_lens (bool): Whether to post-process the lens or not.
        total_power (float): The total power of the light source in Watts.
        save_lens_history (bool): Whether to save lens history or not.
        save_history (bool): Whether to save optimization history or not.
        save_irradiance_results (bool): Whether to save irradiance results or not.
        num_rays_save_irradiance (int): The number of rays to save for irradiance results.
        html_plot_file_name (str): The name of the HTML file to save the plot.
    
    Returns:
        dict: A dictionary containing the results of the optimization and irradiance calculations.
    """
    gc.collect()

    num_integration_points_desired = copy.deepcopy(num_integration_points_desired)
    num_rays = copy.deepcopy(num_rays)
    
    from ... import source
    from ... import transforms
    from ... import Bspline
    from ... import Plane
    from ... import Lens
    from ... import Detector
    from ... import SequentialOpticalSystem
    from ... import utils
    from ... import plotting
    from ... import target_grid
    from ... import render
    from ... import minimize
    from ... import set_unused_bspline_coeff_to_nearest
    from ... import export
    from ..import smoothing


    if (image_padding==0.0):
        raise ValueError("Please don't set image_padding to 0.0.")

    if num_rays_save_irradiance is None:
        num_rays_save_irradiance = num_rays

    if (not use_desired_irradiance_smoothing) and use_sigma_refinement:
        raise ValueError("if use_sigma_refinement==True,use_desired_irradiance_smoothing must be enabled!")

    ns_start = copy.deepcopy(bspline_ns_start)
    orders = copy.deepcopy(bspline_orders)
    
    light_transform = transforms.Offset(torch.tensor([0.0,0.0,0.0]))
    light_transform.pos.requires_grad = False

    light_source = None
    if etendue:
        light_source = source.VisibleSunlightSimpleMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power)
    else:
        light_source = source.CollimatedMonochromatic(light_transform,aperture_radius_source,wl=0.5,total_power=total_power,is_square=True)

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
    irr_func = utils.irradiance_importer.create_irradiance_from_image_square(input_file_name,image_padding,0.,aperture_radius_detector,shape=[num_conv_points,num_conv_points])
    #plotting.quantity2D.plot(irr_func,"Desired Irradiance Distribution",cmap="grey",x_range=[-aperture_radius_detector,aperture_radius_detector])
    
    def get_desired_irradiance_raw():
        resolution = num_conv_points
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

    sigmas = None
    if use_sigma_refinement:
        sigmas = get_all_sigmas_refined(ns_start,orders,sigma_final,num_refinements+1)
    else:
        sigmas = [sigma_final for _ in range(num_refinements)]
    
    print("using sigmas",sigmas)
    
    def create_lens_copy():
        lens1_copy = copy.deepcopy(lens1)
        lens1_copy = lens1_copy.cpu()
        lens1_copy.n_func = None
        lens1_copy.surface1.n_func = None
        lens1_copy.surface2.n_func = None
        
        return lens1_copy
    
    def create_html_plot(prefix):
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
        
    create_html_plot("initial")
    
    def create_smoother(sigma):
        return smoothing.GaussianSmootherSquare(aperture_radius_detector,\
                                                                       num_conv_points=num_conv_points,\
                                                                        sigma=sigma,\
                                                                        device=device,\
                                                                        num_integration_points_desired=num_integration_points_desired,\
                                                                        desired_irradiance_func=irr_func,\
                                                                        residual_integration_method=residual_integration_method,\
                                                                        total_power_desired=total_power,
                                                                        num_eval_points = num_conv_points,use_eval_avg=False)

    final_smoother = create_smoother(sigmas[-1])
    
    def run_ray_tracer_smooth(smoother):
        import matplotlib.pyplot as plt
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

    smoother = create_smoother(sigmas[0])

    initial_smooth_irradiance = None
    initial_binned_irradiance = None
    initial_desired_smooth_irradiance = None
    initial_desired_none_smooth_irradiance_opti = None
    initial_desired_none_smooth_irradiance_eval = None
    initial_binned_irradiance_eval = None
    
    if save_irradiance_results:
        initial_smooth_irradiance = run_ray_tracer_smooth(smoother)
        initial_binned_irradiance_eval = run_ray_tracer_none_smooth_eval(smoother)
        initial_binned_irradiance = initial_binned_irradiance_eval
        
        initial_desired_smooth_irradiance = smoother.desired_smooth_irradiance.detach().cpu().numpy()
        initial_desired_none_smooth_irradiance_opti = smoother.desired_none_smooth_irradiance_opti.detach().cpu().numpy()
        initial_desired_none_smooth_irradiance_eval = smoother.desired_none_smooth_irradiance_eval.detach().cpu().numpy()


    def convergence_callback_func(smoother:smoothing.Smoother,convergence_list:list,power_irr_smooth_list:list,power_desired_irr_list:list):
        def return_func():
            power_irr_smooth_list.append(smoother.last_smoothed_irradiance_power)
            power_desired_irr_list.append(smoother.last_desired_irr_power)
            convergence_list.append(smoother.last_eval_merit_val)
        
        return return_func 

    def minimization_call(smoother:smoothing.GaussianSmootherSquare,k):
        merit_func = lambda smoother:smoothing.create_merit_function(system,
                                                                     sequence,
                                                                     light_source,
                                                                     detector,
                                                                     num_rays,
                                                                     smoother,
                                                                     device,
                                                                     method_ray_tracing=method_ray_tracing,
                                                                     use_desired_irradiance_smoothing=use_desired_irradiance_smoothing,
                                                                     use_power_correction=use_power_correction,
                                                                     save_last_eval=save_history)
        if save_history:
            convergence_list = []
            power_irr_smooth_list = []
            power_desired_irr_list = []
            out = minimize(merit_func(smoother),system.parameters(),callback=convergence_callback_func(smoother,convergence_list,power_irr_smooth_list,power_desired_irr_list),method=minimization_method,save_history=save_history,call_before_minimize=True)
            out["sigma"] = smoother.sigma
            out["coeff_shape"] = bspline_surface1.coeff.shape
            out["loop_number"] = k
        
            out["history"]["convergence"] = convergence_list
            out["history"]["power_irr_smooth_list"] = power_irr_smooth_list
            out["history"]["power_desired_irr_list"] = power_desired_irr_list

            if save_irradiance_results:
                out["desired_smooth_irradiance"] = smoother.desired_smooth_irradiance.detach().cpu().numpy()
                out["desired_none_smooth_irradiance_opti"] = smoother.desired_none_smooth_irradiance_opti.detach().cpu().numpy()
                out["desired_none_smooth_irradiance_eval"] = smoother.desired_none_smooth_irradiance_eval.detach().cpu().numpy()

                out["smooth_irradiance"] = run_ray_tracer_smooth(smoother)
                out["binned_irradiance_eval"] = run_ray_tracer_none_smooth_eval(smoother)
                out["binned_irradiance"] = out["binned_irradiance_eval"]

            if save_lens_history:
                out["lens"] = create_lens_copy() 
            return out   
        else:
            out = minimize(merit_func(smoother),system.parameters(),method=minimization_method)
            out["sigma"] = smoother.sigma
            out["coeff_shape"] = bspline_surface1.coeff.shape
            out["loop_number"] = k
            if save_irradiance_results:
                out["desired_smooth_irradiance"] = smoother.desired_smooth_irradiance.detach().cpu().numpy()
                out["desired_none_smooth_irradiance_opti"] = smoother.desired_none_smooth_irradiance_opti.detach().cpu().numpy()
                out["desired_none_smooth_irradiance_eval"] = smoother.desired_none_smooth_irradiance_eval.detach().cpu().numpy() 

                
                out["desired_irradiance_smooth"] = smoother.desired_smooth_irradiance
    
                out["smooth_irradiance"] = run_ray_tracer_smooth(smoother)
                out["binned_irradiance_eval"] = run_ray_tracer_none_smooth_eval(smoother)
                out["binned_irradiance"] = out["binned_irradiance_eval"]
                
            if save_lens_history:
                out["lens"] = create_lens_copy()
           
            return out

    def run_minimization_loop_sigma_refinement():
        results_minimize = [] 
        print("run_minimization_loop_sigma_refinement")
        
        for k in range(num_refinements):
            print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
            smoother.sigma = sigmas[k]        
            results_minimize += [minimization_call(smoother,k)]    
            print("END")
            smoother.sigma = sigmas[k+1]        
            
            print("BEGIN: opti after sigma finer: coeff shape:",bspline_surface1.coeff.shape)
            results_minimize += [minimization_call(smoother,k)]    
            bspline_surface1.refine()
            print("END")
        print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
        results_minimize += [minimization_call(smoother,num_refinements)]    
        print("END")
        print("FINISHED MINIMIZATION")
        return results_minimize

    def run_minimization_loop_classical():
        results_minimize = [] 
        print("run_minimization_loop_classical")
        for k in range(num_refinements):
            print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
            smoother.sigma = sigmas[k]        
            results_minimize += [minimization_call(smoother,k)]    
            print("END")
            bspline_surface1.refine()
        print("BEGIN: opti after refine: coeff shape:",bspline_surface1.coeff.shape)
        results_minimize += [minimization_call(smoother,num_refinements)]    
        print("END")
        return results_minimize

    gc.collect()
    results_minimize = None
    if use_sigma_refinement:
        results_minimize = run_minimization_loop_sigma_refinement()
    else:
        results_minimize = run_minimization_loop_classical()

    if post_process_lens:
        set_unused_bspline_coeff_to_nearest(system,sequence,light_source,bspline_surface1,num_rays=num_rays,method_ray_tracing=method_ray_tracing)

    if output_step_file_name is not None:
        if output_step_file_name != "":
            export_resolution = 256 #has no influence because we are using bsplines!
            export.cad.export_lens(output_step_file_name,lens1,export_resolution)
    
    


    final_lens = create_lens_copy()
    
    gc.collect()
    create_html_plot("final")
    gc.collect()
    
    def get_final_irr_results():
        out = {}
        out["smooth_irradiance"] = run_ray_tracer_smooth(final_smoother)
        out["binned_irradiance_eval"] = run_ray_tracer_none_smooth_eval(final_smoother)
        out["binned_irradiance"] = out["binned_irradiance_eval"]
        out["desired_smooth_irradiance"] = final_smoother.desired_smooth_irradiance.detach().cpu().numpy()
        out["desired_none_smooth_irradiance_opti"] = final_smoother.desired_none_smooth_irradiance_opti.detach().cpu().numpy()
        out["desired_none_smooth_irradiance_eval"] = final_smoother.desired_none_smooth_irradiance_eval.detach().cpu().numpy()
        return out

    def get_initial_irr_results():
        out = {}
        out["smooth_irradiance"] = initial_smooth_irradiance
        out["binned_irradiance"] = initial_binned_irradiance
        out["binned_irradiance_eval"] = initial_binned_irradiance_eval
        out["desired_smooth_irradiance"] = initial_desired_smooth_irradiance
        out["desired_none_smooth_irradiance_opti"] = initial_desired_none_smooth_irradiance_opti
        out["desired_none_smooth_irradiance_eval"] = initial_desired_none_smooth_irradiance_eval
        return out
    
    def get_settings():
        out={}
        out["aperture_radius_detector"] = aperture_radius_detector
        out["aperture_radius_lens"] = aperture_radius_lens
        out["aperture_radius_source"] = aperture_radius_source
        out["num_refinements"] = num_refinements
        out["image_padding"] = image_padding
        out["sigma_final"] = sigma_final
        out["lens_distance"] = lens_distance
        out["lens_thickness"] = lens_thickness
        out["detector_distance"] = detector_distance
        out["desired_irradiance_raw"] = desired_irradiance_raw
        out["residual_integration_method"] = residual_integration_method 
        out["minimization_method"] = minimization_method
        out["num_rays_opti"] = num_rays
        out["total_power_preset"] = total_power
        out["num_integration_points_desired"] = num_integration_points_desired
        out["bspline_orders"] = bspline_orders
        out["bspline_ns_start"] = bspline_ns_start
        out["num_conv_points"] = num_conv_points
        out["use_sigma_refinement"] = use_sigma_refinement
        out["use_desired_irradiance_smoothing"] = use_desired_irradiance_smoothing
        out["etendue"] = etendue
        out["use_power_correction"] = use_power_correction
        return out
    
    out = {}
    out["results_minimize"] = results_minimize
    
    if save_lens_history:
        out["final_lens"] = final_lens
    
    out["settings"] = get_settings()
    if save_irradiance_results:
        out["initial_irr_results"] = get_initial_irr_results()
        out["final_irr_results"] = get_final_irr_results()
        
    return out
