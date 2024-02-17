import numpy as np
import scipy.optimize as optimize
import os
import time
import pandas as pd
import pickle
import gpflow
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance
import unyt as u
import matplotlib
import matplotlib.pyplot as plt

mpl_is_inline = 'inline' in matplotlib.get_backend()

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

def get_test_data(molec_key, prop_keys):
    """
    Get gp data from .pkl files

    Parameters
    ----------
    key_list: list of keys to consider. Must be valid Keys: "R14", "R32", "R50", "R125", "R143a", "R134a", "R170"

    Returns:
    --------
    all_gp_dict: dict, dictionary of dictionary of gps for each property
    """
    #Get dict of testing data
    test_data = {}
    #OPTIONAL append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
    file = os.path.join(molec_key +"-vlegp/x_test.csv")
    assert os.path.isfile(file), "key-vlegp/x_test.csv does not exist. Check key list carefully"
    x_data = np.loadtxt(file, delimiter=",",skiprows=1)
    test_data["x"]=x_data
    for prop_key in prop_keys:
        file = os.path.join(molec_key +"-vlegp/" + prop_key + "_y_test.csv")
        prop_data = np.loadtxt(file, delimiter=",",skiprows=1)
        test_data[prop_key]=prop_data

    return test_data

class Opt_ATs:
    """
    The class for Least Squares regression analysis. Child class of General_Analysis
    """
    #Inherit objects from General_Analysis
    def __init__(self, molec_data_dict, all_gp_dict, at_class, repeats, seed, save_data):
        #Asserts
        
        assert isinstance(molec_data_dict, dict), "molec_data_dict must be a dictionary"
        assert isinstance(all_gp_dict, dict), "all_gp_dict must be a dictionary"
        assert list(molec_data_dict.keys()) == list(all_gp_dict.keys()), "molec_data_dict and all_gp_dict must have same keys"
        assert isinstance(save_data, bool), "save_res must be bool"
        assert isinstance(repeats, int) and repeats > 0, "repeats must be int > 0"
        assert isinstance(seed, int) or seed is None, "seed must be int or None"

        self.col_names = ["Run", "Iter", 'Min Obj', 'Param Min', "Min Obj Cum.", "Param Cum.", "jac evals", "Termination", 
                            "Run Time"]
        self.col_names_iter = ["Run", "Iter", 'Min Obj', 'Param Min', "Min Obj Cum.", 
                                 "Param Cum."]
        self.iter_param_data = []
        self.iter_obj_data = []
        self.iter_count = 0
        self.seed = seed
        #Placeholder that will be overwritten if None
        self.molec_data_dict = molec_data_dict
        self.all_gp_dict = all_gp_dict
        self.at_class = at_class
        self.seed = seed
        self.repeats = repeats
        self.save_data = save_data
    
    #define the scipy function for minimizing
    def __scipy_min_fxn(self, theta_guess):
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
        #Initialize weight and squared error arrays
        sqerr_array  = []
        weight_array = []
        
        #Loop over molecules
        for molec in list(self.molec_data_dict.keys()):
            #Get constants for molecule
            molec_object = self.molec_data_dict[molec]
            #Get theta associated with each gp
            param_matrix = self.at_class.get_transformation_matrix(molec)
            #Transform the guess, and scale to bounds
            gp_theta = theta_guess.reshape(1,-1)@param_matrix
            gp_theta_guess = values_real_to_scaled(gp_theta, molec_object.param_bounds)
            #Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]
            
            #Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                #Get GP associated with property
                gp_model = molec_gps_dict[key]
                #Get X and Y data and bounds associated with the GP
                exp_data, y_bounds = self.get_exp_data(molec_object, key)
                #Get x and y data
                x_exp = np.array(list(exp_data.keys())).reshape(-1,1)
                y_exp = np.array(list(exp_data.values()))
                # #Evaluate GP
                gp_mean, gp_var = self.eval_gp_new_theta(gp_theta_guess, molec_object, gp_model, x_exp)
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

        #Scale theta_guess to real values 
        midpoint = len(theta_guess) //2
        sigmas = [float((x * u.nm).in_units(u.Angstrom).value) for x in theta_guess[:midpoint]]
        epsilons = [float(x / (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
        theta_guess = np.array(sigmas + epsilons)

        #Append intermediate values to list
        self.iter_param_data.append(theta_guess)
        self.iter_obj_data.append(obj)
        self.iter_count += 1

        return obj

    def get_exp_data(self, molec_object, prop_key):
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
    
    #Create fxn for analyzing a single gp w/ gpflow
    def eval_gp_new_theta(self, gp_theta_guess, molec_object, gp_object, Xexp):
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

    #Define function to check GP Accuracy
    def check_GPs(self):
        """
        Makes GPs 
        """
        #Loop over molecules
        for molec in list(self.all_gp_dict.keys()):
            #Get constants for molecule
            molec_object = self.molec_data_dict[molec]
            #Get GPs associated with each molecule
            molec_gps_dict = self.all_gp_dict[molec]
            #Get testing data for that molecule
            test_data = get_test_data(molec, molec_gps_dict.keys())
            #Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                #Set label
                label = molec + "_" + key
                #Get GP associated with property
                gp_model = molec_gps_dict[key]
                #Get X and Y data and bounds associated with the GP
                exp_data, y_bounds = self.get_exp_data(molec_object, key)
                #Plot
                fig = plot_model_performance({label:gp_model}, test_data["x"], test_data[key], y_bounds)
                if mpl_is_inline:
                    plt.show()
                    plt.close()

        return


    def __get_params_and_df(self):
        """
        Gets parameter guesses and sets up bounds for optimization
        """
        #set seed here
        if self.seed is not None:
            np.random.seed(self.seed)

        #Get initial guesses from bounds (Sigma in nm and Epsilon in kJ/mol)
        lb = self.at_class.at_bounds_nm_kjmol[:,0].T
        ub = self.at_class.at_bounds_nm_kjmol[:,1].T
        param_inits = np.random.uniform(low=lb, high=ub, size=(self.repeats, len(lb)) )

        #Initialize results dataframe
        ls_results = pd.DataFrame(columns=self.col_names)

        return param_inits, ls_results
    
    def __get_scipy_soln(self, run, param_inits):

        #Start timer
        time_start = time.time()
        #Get guess and find scipy.optimize solution
        solution = optimize.minimize(self.__scipy_min_fxn, param_inits[run] , bounds=self.at_class.at_bounds_nm_kjmol, 
                                        method='L-BFGS-B', options = {"disp":False})
        #End timer and calculate total run time
        time_end = time.time()
        time_per_run = time_end-time_start

        return solution, time_per_run

    def __get_opt_iter_info(self, run, solution, time_per_run):
        """
        Runs Optimization, times progress, and makes iter_df
        """
        #Get list of iteration, sse, and parameter data
        iter_list = np.array(range(self.iter_count)) + 1
        obj_list = np.array(self.iter_obj_data)
        param_list = self.iter_param_data

        #Create df for each least squares run
        #Create a pd dataframe of all iteration information. Initialize cumulative columns as zero
        ls_iter_res = [run + 1, iter_list, obj_list, param_list, None , None]
        iter_df = pd.DataFrame([ls_iter_res], columns = self.col_names_iter)
        iter_df = iter_df.apply(lambda col: col.explode(), axis=0).reset_index(drop=True).copy(deep =True)

        #Add Theta min obj and theta_sse obj
        #Loop over each iteration to create the min obj columns
        for j in range(len(iter_df)):
            min_sse = iter_df.loc[j, "Min Obj"].copy()
            if j == 0 or min_sse < iter_df["Min Obj"].iloc[j-1]:
                min_param = iter_df["Param Min"].iloc[j].copy()
            else:
                min_sse = iter_df["Min Obj Cum."].iloc[j-1].copy()
                min_param = iter_df["Param Cum."].iloc[j-1].copy()

            iter_df.loc[j, "Min Obj Cum."] = min_sse
            iter_df.at[j, "Param Cum."] = min_param

        iter_df["Run Time"] = time_per_run
        iter_df["jac evals"] = solution.njev
        iter_df["Termination"] = solution.status

        return iter_df
    
    #Define fxn to optimize w/ restarts
    def optimize_ats(self):
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
        param_inits, ls_results = self.__get_params_and_df()

        #Optimize w/ retstarts
        for i in range(self.repeats):
            #Get Iteration results
            solution, time_per_run = self.__get_scipy_soln(i, param_inits)
            iter_df = self.__get_opt_iter_info(i, solution, time_per_run)

            #Append to results_df
            ls_results = pd.concat([ls_results.astype(iter_df.dtypes), iter_df], ignore_index=True)

            #Reset iter lists and change seed
            self.seed += 1
            self.iter_param_data = []
            self.iter_obj_data = []
            self.iter_count = 0

        #Reset the index of the pandas df
        ls_results = ls_results.reset_index(drop=True)

        #Sort by lowest obj first
        sort_ls_res = ls_results.sort_values(['Min Obj Cum.'], ascending= True)

        #Back out best job for each run
        run_best = sort_ls_res.groupby('Run').first().reset_index()
        best_runs = run_best.sort_values(['Min Obj Cum.'], ascending= True)

        if self.save_data:
            dir_name = "Results/"
            os.makedirs(dir_name, exist_ok=True) 
            #Save original results
            save_path1 = os.path.join(dir_name, "opt_at_results.csv")
            ls_results.to_csv(save_path1, index = False)
            #Save sorted results
            save_path2 = os.path.join(dir_name, "sorted_at_res.csv")
            sort_ls_res.to_csv(save_path2, index = False)
            #Save sorted results for each run
            save_path3 = os.path.join(dir_name, "best_per_run.csv")
            best_runs.to_csv(save_path3, index = False)

        return ls_results

