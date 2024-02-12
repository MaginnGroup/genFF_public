import numpy as np
import scipy.optimize as optimize
import os
import time
import pandas as pd
import pickle
import gpflow
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
import unyt as u

#Create a function for getting gp data from files
def get_gp_data_from_pkl(key_list):
    """
    Get gp data from .pkl files

    Parameters
    ----------
    key_list: list of keys to consider. Must be valid Keys: "R14", "R32", "R50", "R125", "R143a", "R134a", "R170"

    Returns:
    --------
    all_gp_dict: dict, dictionary of dictionary of gps for each property
    """
    valid_keys = ["R14", "R32", "R50", "R125", "R143a", "R134a", "R170"]
    assert isinstance(key_list, list), "at_names must be a list"
    assert all(isinstance(name, str) for name in key_list) == True, "all key in key_list must be string"
    assert all(key in valid_keys for key in key_list) == True, "all key in key_list must be valid keys"
    #Make a dict of the gp dictionaries for each molecule
    all_gp_dict = {}
    #loop over molecules
    for key in key_list:
        #Get dict of vle gps
        #OPTIONAL append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
        file = os.path.join(key +"-vlegp/vle-gps.pkl")
        assert os.path.isfile(file), "key-vlegp/vle-gps.pkl does not exist. Check key list carefully"
        with open(file, 'rb') as pickle_file:
            all_gp_dict[key] = pickle.load(pickle_file)

    return all_gp_dict

def get_exp_data(molec_object, prop_key):
    """
    Helper function for getting experimental data and bounds

    Parameters
    ----------
    molec_object: Instance of RXXXXConstant() class. Class for refrigerant molecule data
    prop_key: str, The property key to get exp_data for. Valid Keys are "sim_vap_density", "sim_liq_density", 
    "sim_Pvap", "sim_Hvap"

    Returns:
    --------
    exp_data: dict, dictionary of Temperature and property data
    property_bounds: array, array of bounds for the property data
    """
    #How to assert that we have a constants class?
    valid_prop_keys = ["sim_vap_density", "sim_liq_density", "sim_Pvap", "sim_Hvap"]

    if prop_key not in valid_prop_keys:
        raise ValueError(
            "Invalid prop_key {}. Supported prop_key names are "
            "{}".format(prop_key, valid_prop_keys))
    
    if "vap_density" in prop_key:
        exp_data = molec_object.expt_vap_density
        property_bounds = molec_object.vap_density_bounds
    elif "liq_density" in prop_key:
        exp_data = molec_object.expt_liq_density
        property_bounds = molec_object.liq_density_bounds
    elif "Pvap" in prop_key: 
        exp_data = molec_object.expt_Pvap
        property_bounds = molec_object.Pvap_bounds
    elif "Hvap" in prop_key:
        exp_data = molec_object.expt_Hvap
        property_bounds = molec_object.Hvap_bounds
    else:
        raise(ValueError, "all_gp_dict must contain a dict with keys sim_vap_density, sim_liq_density, sim_Hvap, or, sim_Pvap")
    return exp_data, property_bounds

#define the scipy function for minimizing
def scipy_min_fxn(theta_guess, molec_data_dict, all_gp_dict, at_class):
    """
    The scipy function for minimizing the data

    Parameters
    ----------
    theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in kJ/mol)
    molec_data_dict: dict, dictionary of Refrigerant constants
    all_gp_dict: dict, dictionary of refrigerant GP models
    at_class: Atom_Types() instance, The class for atom types 

    Returns
    --------
    obj: float, the objective function from the formula defined in the paper
    """
    assert isinstance(theta_guess, np.ndarray), "theta_guess must be an np.ndarray"
    assert isinstance(molec_data_dict, dict), "molec_data_dict must be a dictionary"
    assert isinstance(all_gp_dict, dict), "all_gp_dict must be a dictionary"
    assert list(molec_data_dict.keys()) == list(all_gp_dict.keys()), "molec_data_dict and all_gp_dict must have same keys"
    #Initialize weight and squared error arrays
    sqerr_array  = []
    weight_array = []
    
    #Loop over molecules
    for molec in list(molec_data_dict.keys()):
        #Get constants for molecule
        molec_object = molec_data_dict[molec]
        #Get theta associated with each gp
        param_matrix = at_class.get_transformation_matrix(molec)
        #Transform the guess, and scale to bounds
        gp_theta = theta_guess.reshape(1,-1)@param_matrix
        gp_theta_guess = values_real_to_scaled(gp_theta, molec_object.param_bounds)
        #Get GPs associated with each molecule
        molec_gps_dict = all_gp_dict[molec]
        
        #Loop over gps (1 per property)
        for key in list(molec_gps_dict.keys()):
            #Get GP associated with property
            gp_model = molec_gps_dict[key]
            #Get X and Y data and bounds associated with the GP
            exp_data, y_bounds = get_exp_data(molec_object, key)
            #Get x and y data
            x_exp = np.array(list(exp_data.keys())).reshape(-1,1)
            y_exp = np.array(list(exp_data.values()))
            # #Evaluate GP
            gp_mean, gp_var = eval_gp_new_theta(gp_theta_guess, molec_object, gp_model, x_exp)
            #Scale gp output to real value
            gp_mean = values_scaled_to_real(gp_mean, y_bounds)
            #Scale gp_variances to real values
            gp_var = variances_scaled_to_real(gp_var, y_bounds)
            #Calculate weight from uncertainty
            weight_mpi = (1/(gp_var)).tolist()
            weight_array += weight_mpi
            #Calculate sse
            sq_err = ((y_exp.flatten() - gp_mean.flatten())**2).tolist()
            sqerr_array += sq_err
    
    #List to array
    sqerr_array = np.array(sqerr_array)
    weight_array = np.array(weight_array)
    #Normalize weights to add up to 1
    scaled_weights = weight_array / np.sum(weight_array)
    #Define objective function
    obj = np.sum(scaled_weights*sqerr_array)
    return obj


#Create fxn for analyzing a single gp w/ gpflow
def eval_gp_new_theta(gp_theta_guess, molec_object, gp_object, Xexp):
    """
    Evaluates the gpflow model

    Parameters
    ----------
    gp_theta_guess: np.ndarray, the initial gp parameter set to start optimization at (sigma in A, eps in kJ/mol)
    molec_object: Instance of RXXConstants(), The data associated with a refrigerant molecule
    gp_object: gpflow.models.GPR, GP Model for a specific property
    Xexp: np.ndarray, Experimental state point (Temperature (K)) data for the property estimated by gp_object

    Returns
    -------
    gp_mean: tf.tensor, The (flattened) mean of the gp prediction
    gp_var: tf.tensor, The (flattened) variance of the gp prediction
    """
    assert isinstance(Xexp, np.ndarray), "Xexp must be an np.ndarray"
    assert isinstance(gp_object, gpflow.models.GPR)
    
    #Scale X data
    gp_Xexp = values_real_to_scaled(Xexp, molec_object.temperature_bounds)
    #Repeat theta guess x number of times
    gp_theta = np.repeat(gp_theta_guess, len(Xexp) , axis = 0)
    #Concatenate theta and tem values to get a gp input
    gp_input = np.concatenate((gp_theta, gp_Xexp), axis=1)
    #Get mean and std from gp
    gp_mean, gp_covar = gp_object.predict_f(gp_input, full_cov=True)
    gp_var = np.diag(np.squeeze(gp_covar))
    return np.squeeze(gp_mean), gp_var

#Define fxn to optimize w/ restarts
def optimize_ats(repeats, at_class, molec_data_dict, all_gp_dict, save_res, seed = None):
    """
    Optimizes New atom typing parameters

    Parameters
    ----------
    at_class: Atom_Types() instance, The class for atomy types 
    molec_data_dict: dict, dictionary of Refrigerant constants
    all_gp_dict: dict, dictionary of refrigerant GP models
    save_res: bool, Determines whether to save results
    seed: int, seed for rng. Default None
    """
    assert isinstance(save_res, bool), "save_res must be bool"
    assert isinstance(repeats, int), "repeats must be int"
    assert isinstance(seed, int) or seed is None, "seed must be int or None"
    assert isinstance(molec_data_dict, dict), "molec_data_dict must be a dictionary"
    assert isinstance(all_gp_dict, dict), "all_gp_dict must be a dictionary"
    assert list(molec_data_dict.keys()) == list(all_gp_dict.keys()), "molec_data_dict and all_gp_dict must have same keys"
    
    #set seed here
    if seed is not None:
        np.random.seed(seed)

    #Get initial guesses from bounds (Sigma in nm and Epsilon in kJ/mol)
    lb = at_class.at_bounds_nm_kjmol[:,0].T
    ub = at_class.at_bounds_nm_kjmol[:,1].T
    param_inits = np.random.uniform(low=lb, high=ub, size=(repeats, len(lb)) )

    #Initialize results dataframe
    column_names = ['Param Init', 'Min Obj', 'Param at Min Obj', 'Min Obj Cum.', 'Param at Min Obj Cum.',
                    "func evals", "jac evals", "Termination", "Total Run Time"]
    ls_results = pd.DataFrame(columns=column_names)

    #Optimize w/ retstarts
    for i in range(repeats):
        #Start timer
        time_start = time.time()
        #Get guess and find scipy.optimize solution
        Solution = optimize.minimize(scipy_min_fxn, param_inits[i] , bounds=at_class.at_bounds_nm_kjmol, method='L-BFGS-B', 
                                    args=(molec_data_dict, all_gp_dict, at_class), options = {"disp":False})
        #End timer and calculate total run time
        time_end = time.time()
        time_per_run = time_end-time_start

        #Back out results
        param_min_obj = Solution.x
        min_obj = Solution.fun

        #Scaled to real parameter values
        midpoint = len(param_min_obj) //2
        sigmas = [float((x * u.nm).in_units(u.Angstrom).value) for x in param_min_obj[:midpoint]]
        epsilons = [float(x / (u.K * u.kb).in_units("kJ/mol")) for x in param_min_obj[midpoint:]]
        param_min_obj = np.array(sigmas + epsilons)
        
        #Create df for each least squares run
        iter_df = pd.DataFrame(columns=column_names)

        #On 1st iteration, min obj cum and theta min obj cum are the same as sse and sse min obj
        if i==0 or min_obj < ls_results["Min Obj Cum."].iloc[i-1]:
            obj_cum = min_obj  
            theta_obj_cum = param_min_obj
        else:
            obj_cum = ls_results["Min Obj Cum."].iloc[i-1]
            theta_obj_cum = ls_results['Param at Min Obj Cum.'].iloc[i-1]

        #get list of data for this iteration
        ls_iter_res = [param_inits[i], min_obj, Solution.x, obj_cum, theta_obj_cum,  Solution.nfev, 
                            Solution.njev, Solution.status, time_per_run]

        # Add the new row to the DataFrame
        iter_df.loc[0] = ls_iter_res
        ls_results = pd.concat([ls_results.astype(iter_df.dtypes), iter_df], ignore_index=True)

        if save_res:
            dir_name = "Results/"
            os.makedirs(dir_name, exist_ok=True) 
            save_path = os.path.join(dir_name, "opt_at_results.csv")
            ls_results.to_csv(save_path)

    return ls_results
