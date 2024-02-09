import numpy as np
import scipy.optimize as optimize
import os
import time
import pandas as pd
import pickle
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type

#Create a function for getting gp data from files
def get_gp_data_from_pkl(key_list):
    #Make a dict of the gp dictionaries for each molecule
    all_gp_dict = {}
    #loop over molecules
    for key in key_list:
        #Get dict of vle gps
        #OPTIONAL append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
        file = os.path.join(key +"-vlegp/vle-gps.pkl")
        with open(file, 'rb') as pickle_file:
            all_gp_dict[key] = pickle.load(pickle_file)
    return all_gp_dict


#define the scipy function for minimizing
def scipy_min_fxn(theta_guess, molec_data_dict, all_gp_dict, at_class):
    #Initialize weight and squared error arrays
    sqerr_array  = []
    weight_array = []

    #Loop over molecules
    for molec in list(molec_data_dict.keys()):
        #Get theta associated with each gp
        param_matrix = at_class.get_transformation_matrix(molec)
        #Get GPs associated with each molecule
        molec_gps_dict = all_gp_dict[molec]
        #Loop over gps (1 per property)
        for key in list(molec_gps_dict.keys()):
            #Get GP associated with property
            gp_model = molec_gps_dict[key]
            #Get X and Y data associated with the GP
            if "vap_density" in key:
                exp_data = molec_data_dict[molec].expt_vap_density
            elif "liq_density" in key:
                exp_data = molec_data_dict[molec].expt_liq_density
            elif "Pvap" in key: 
                exp_data = molec_data_dict[molec].expt_Pvap
            elif "Hvap" in key:
                exp_data = molec_data_dict[molec].expt_Hvap
            else:
                raise(ValueError, "all_gp_dict must contain a dict with keys sim_vap_density, sim_liq_density, sim_Hvap, or, sim_Pvap")
            #Get x and y data
            x_exp = np.array(list(exp_data.keys())).reshape(-1,1)
            y_exp = np.array(list(exp_data.values()))

            # #Evaluate GP
            gp_mean, gp_std = eval_gp_new_theta(theta_guess, param_matrix, gp_model, x_exp)
            #Calculate weight from uncertainty
            weight_mpi = (1/(gp_std**2)).tolist()
            weight_array += weight_mpi
            #Calculate sse
            sq_err = ((y_exp.flatten() - gp_mean)**2).tolist()
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
def eval_gp_new_theta(theta_guess, t_matrix, gp_object, Xexp):
    #Get theta into correct form using t_matrix
    theta_guess = theta_guess.reshape(1,-1)
    gp_theta = theta_guess@t_matrix
    #Append x data for consideration
    gp_theta = np.repeat(gp_theta, len(Xexp) , axis = 0)
    gp_input = np.concatenate((gp_theta, Xexp), axis=1)
    #Get mean and std from gp
    gp_mean, gp_covar = gp_object.predict_f(gp_input, full_cov=True)
    gp_std = np.sqrt(np.diag(np.squeeze(gp_covar)))
    return np.squeeze(gp_mean), gp_std

#Define fxn to optimize w/ restarts
def optimize_ats(repeats, at_class, molec_data_dict, all_gp_dict, save_res):
    #Add a seed here
    
    #Get initial guesses
    lb = at_class.at_bounds[:,0].T
    ub = at_class.at_bounds[:,1].T
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
        Solution = optimize.minimize(scipy_min_fxn, param_inits[i] , bounds=at_class.at_bounds, method='L-BFGS-B', 
                                    args=(molec_data_dict, all_gp_dict, at_class), options = {"disp":False})
        #End timer and calculate total run time
        time_end = time.time()
        time_per_run = time_end-time_start

        #Back out results
        param_min_obj = Solution.x
        min_obj = Solution.fun
        
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
