import numpy as np
import scipy.optimize as optimize
import scipy
import os
import time
import pandas as pd
import pickle
import gpflow
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance, plot_model_vs_test, plot_slices_temperature, plot_slices_params, plot_model_vs_exp, plot_obj_contour
import unyt as u
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tensorflow as tf
from itertools import combinations

mpl_is_inline = 'inline' in matplotlib.get_backend()
# print(mpl_is_inline)

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

class Problem_Setup:
    """
    Gets GP/ Experimental Data For experiments

    Parameters:
    molec_data_dict: dict, keys are refrigerant names w/ capital R, values are class objects from r***.py
    all_gp_dict: dict of dict, keys are refrigerant names w/ capital R, values are dictionaries of properties and GP objects
    at_class: Instance of Atom_Types, class for atom typing
    w_calc: int, 0,1, or 2. The calculation to use for weights in objective calculation. 0 = 1/gp_var scaled to 1, 1 = 1/gp var, 2 = 1/y_exp_var
    save_data: bool, whether to save data or not
    """
    #Inherit objects from General_Analysis
    def __init__(self, molec_data_dict, all_gp_dict, at_class, w_calc, obj_choice, save_data):
        assert isinstance(molec_data_dict, dict), "molec_data_dict must be a dictionary"
        assert isinstance(all_gp_dict, dict), "all_gp_dict must be a dictionary"
        assert list(molec_data_dict.keys()) == list(all_gp_dict.keys()), "molec_data_dict and all_gp_dict must have same keys"
        assert isinstance(save_data, bool), "save_res must be bool"
        assert isinstance(w_calc, int) and w_calc in [0,1,2], "w_calc must be 0, 1 or 2"
        #Placeholder that will be overwritten if None
        self.molec_data_dict = molec_data_dict
        self.all_gp_dict = all_gp_dict
        self.at_class = at_class
        self.save_data = save_data
        self.w_calc = w_calc
        self.obj_choice = obj_choice

        if obj_choice is not "SSE":
            assert w_calc == 2, "Only objective choice SSE is valid with w_calc methods != 2"

    def make_results_dir(self, molecules):
        scheme_name = self.at_class.scheme_name
        molecule_str = '-'.join(molecules)
        if self.w_calc == 0:
            scl_w_str = "wt_sum1_gp_var"
        elif self.w_calc == 1:
            scl_w_str = "wt_gp_var"
        else:
            scl_w_str = "wt_y_var"

        dir_name = os.path.join("Results" ,scheme_name, molecule_str, self.obj_choice, scl_w_str)
        os.makedirs(dir_name, exist_ok=True) 

        return dir_name

    
    def values_pref_to_real(self, theta_guess):
        """
        Scales preferred units (Angstrom and eps/kb) to real units (nm, kJ/mol)
        """
        midpoint = len(theta_guess) //2
        sigmas = [float((x * u.Angstrom).in_units(u.nm).value) for x in theta_guess[:midpoint]]
        epsilons = [float(x * (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
        theta_guess = np.array(sigmas + epsilons)
        
        return theta_guess
    
    def values_real_to_pref(self, theta_guess):
        """
        Scales real units (nm, Kj/mol) to preferred units (Angstrom and eps/kb)
        """
        midpoint = len(theta_guess) //2
        sigmas = [float((x * u.nm).in_units(u.Angstrom).value) for x in theta_guess[:midpoint]]
        epsilons = [float(x / (u.K * u.kb).in_units("kJ/mol")) for x in theta_guess[midpoint:]]
        theta_guess = np.array(sigmas + epsilons)

        return theta_guess

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
            property_name = "Vapor Density [kg/m^3]"
        elif "liq_density" in prop_key:
            exp_data = molec_object.expt_liq_density
            property_bounds = molec_object.liq_density_bounds
            property_name = "Liquid Density [kg/m^3]"
        elif "Pvap" in prop_key: 
            exp_data = molec_object.expt_Pvap
            property_bounds = molec_object.Pvap_bounds
            property_name = "Vapor pressure [bar]"
        elif "Hvap" in prop_key:
            exp_data = molec_object.expt_Hvap
            property_bounds = molec_object.Hvap_bounds
            property_name = "Enthalpy of Vaporization [kJ/kg]"
        else:
            raise(ValueError, "all_gp_dict must contain a dict with keys sim_vap_density, sim_liq_density, sim_Hvap, or, sim_Pvap")
        return exp_data, property_bounds, property_name

    def get_train_test_data(self, molec_key, prop_keys):
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
        train_data = {}
        for str in ["train", "test"]:
            #OPTIONAL append the MD density gp to the VLE density gp dictionary w/ key "MD Density"
            file_x = os.path.join(molec_key +"-vlegp/x_" + str +".csv")
            assert os.path.isfile(file_x), "key-vlegp/x_****.csv does not exist. Check key list carefully"
            x = np.loadtxt(file_x, delimiter=",",skiprows=1)
            dict = train_data if str == "train" else test_data
            dict["x"]=x

            for prop_key in prop_keys:
                file_y = os.path.join(molec_key +"-vlegp/" + prop_key + "_y_" +str+ ".csv")
                prop_data = np.loadtxt(file_y, delimiter=",",skiprows=1)
                dict[prop_key]=prop_data

        return train_data, test_data

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
        return np.squeeze(gp_mean),  np.squeeze(gp_covar), gp_var
    
    def calc_wt_res(self, theta_guess, w_calc = None):
        """
        Calculates the sse objective function

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in kJ/mol)
        """
        #Initialize weight and squared error arrays
        res_array  = []
        weight_array = []
        var_ratios = []
        sse_var_pieces = {}
        mean_wt_pieces = {}
        sse_pieces = {}
        key_list = []

        w_calc = self.w_calc if w_calc == None else w_calc
        assert isinstance(w_calc, int) and w_calc in [0,1,2], "w_calc must be 0, 1 or 2"
        
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
                key_list.append(molec + "-" + key)
                #Get GP associated with property
                gp_model = molec_gps_dict[key]
                #Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                #Get x and y data
                x_exp = np.array(list(exp_data.keys())).reshape(-1,1)
                y_exp = np.array(list(exp_data.values()))
                # #Evaluate GP
                gp_mean_scl, gp_covar_scl, gp_var_scl = self.eval_gp_new_theta(gp_theta_guess, molec_object, gp_model, x_exp)
                #Scale gp output to real value
                gp_mean = values_scaled_to_real(gp_mean_scl, y_bounds)
                #Scale gp_variances to real values
                y_bounds_2D = np.asarray(y_bounds).reshape(-1,2)
                gp_covar = gp_covar_scl * (y_bounds_2D[:, 1] - y_bounds_2D[:, 0]) ** 2
                gp_var = variances_scaled_to_real(gp_var_scl, y_bounds)
                #Calculate weight from uncertainty
                if w_calc == 2:
                    #Get y data uncertainties
                    unc = molec_object.uncertainties[key.replace("sim", "expt")]
                    y_var = (y_exp*unc)**2
                    weight_mpi = (1/y_var).tolist()
                else:
                    weight_mpi = (1/(gp_var)).tolist()
                weight_array += weight_mpi
                #Calculate residuals
                res_vals = y_exp.flatten() - gp_mean.flatten()
                residuals = (res_vals).tolist()
                dL_dz = -2*(res_vals*np.array(weight_mpi)).reshape(-1,1)
                # print(dL_dz.T.shape, gp_covar.shape, dL_dz.shape)
                sse_var = dL_dz.T@gp_covar@dL_dz
                # var_ratios.append((gp_var/y_var).tolist())
                mean_wt_pieces[molec + "-" + key + "-wt"] = np.mean(weight_mpi)
                sse_pieces[molec + "-" + key + "-sse"] = np.sum(np.square(np.array(residuals)))
                sse_var_pieces[molec + "-" + key + "-sse_var"] = sse_var
                res_array += residuals
        
        #List to flattened array
        res_array = np.array(res_array).flatten()
        weight_array = np.array(weight_array).flatten()

        #Normalize weights to add up to 1 if scl_w is True
        sum_weights = np.sum(weight_array) if w_calc == 0 else 1
        scaled_weights = weight_array / sum_weights

        mean_wt_pieces = dict(zip(mean_wt_pieces.keys(), np.array(list(mean_wt_pieces.values()))/sum_weights))

        #Residual is (y - gp_mean)*sqrt(weight) for each data point
        res = res_array*np.sqrt(scaled_weights)
        return res, sse_pieces, sse_var_pieces, var_ratios, mean_wt_pieces
    
    def calc_obj(self, theta_guess, w_calc = None):
        """
        Calculates the sse objective function

        Parameters
        ----------
        theta_guess: np.ndarray, the atom type scheme parameter set to start optimization at (sigma in A, epsilon in kJ/mol)
        """
        res, sse_pieces, sse_var_pieces, var_ratios, mean_wt_pieces = self.calc_wt_res(theta_guess, w_calc)
        sse = np.sum(np.square(res))
        sum_var_ratios = np.sum(var_ratios)
        expected_sse_val = sse + sum_var_ratios
        sse_std = np.sqrt(abs(np.array(list(sse_var_pieces.values()))))
        if self.obj_choice == "SSE":
            obj = sse
        elif self.obj_choice == "ExpVal":
            obj = expected_sse_val
        elif self.obj_choice == "UCB":
            obj = expected_sse_val + sse_std
        elif self.obj_choice == "LCB":
            obj = expected_sse_val - sse_std
        else:
            raise ValueError(
                "Invalid obj_choice. Supported obj_choice names are 'SSE', 'UCB', 'ExpVal', and 'LCB'")

        return obj, sse_pieces, mean_wt_pieces

class Opt_ATs(Problem_Setup):
    """
    The class for Least Squares regression analysis. Child class of General_Analysis
    """
    #Inherit objects from General_Analysis
    def __init__(self, molec_data_dict, all_gp_dict, at_class, repeats, seed, w_calc, obj_choice, save_data):
        #Asserts
        super().__init__(molec_data_dict, all_gp_dict, at_class, w_calc, obj_choice, save_data)
        assert isinstance(repeats, int) and repeats > 0, "repeats must be int > 0"
        assert isinstance(seed, int) or seed is None, "seed must be int or None"
        self.repeats = repeats
        self.seed = seed
        self.iter_param_data = []
        self.iter_obj_data = []
        self.iter_sse_pieces = []
        self.iter_mean_wt_pieces = []
        self.iter_count = 0
    
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
        # res, sse_pieces, mean_wt_pieces =self.calc_wt_res(theta_guess)
        obj, sse_pieces, mean_wt_pieces =self.calc_obj(theta_guess)
        
        #Scale theta_guess to preferred units
        theta_guess_pref = self.values_real_to_pref(theta_guess)

        # #Append intermediate values to list
        if len(self.iter_param_data) == 0 or not np.allclose(self.iter_param_data[-1], theta_guess_pref):
            self.iter_param_data.append(theta_guess_pref)
            self.iter_sse_pieces.append(sse_pieces)
            self.iter_mean_wt_pieces.append(mean_wt_pieces)
            # self.iter_obj_data.append(np.sum(np.square(res)))
            self.iter_obj_data.append(obj)
            self.iter_count += 1
            # print(self.iter_count, np.sum(np.square(res)))

        return obj

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
        
        return param_inits
    
    def __get_scipy_soln(self, run, param_inits):

        #Start timer
        time_start = time.time()
        #Get guess and find scipy.optimize solution
        # bounds = (self.at_class.at_bounds_nm_kjmol[:,0], self.at_class.at_bounds_nm_kjmol[:,1])
        # solution = optimize.least_squares(self.__scipy_min_fxn, param_inits[run], bounds=bounds,
        #                                  method='trf', verbose = 0)
        solution = optimize.minimize(self.__scipy_min_fxn, param_inits[run], bounds=self.at_class.at_bounds_nm_kjmol,
                                         method='L-BFGS-B', options = {'disp':False, 'eps' : 1e-10, 'ftol':1e-10})
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
        #Initialize results dataframe
        #Make list of column names
        org_names = ["Run", "Iter", 'Min Obj', "Min Obj Cum."]
        at_names_min = [name + "_min" for name in self.at_class.at_names]
        at_names_cum = [name + "_cum" for name in self.at_class.at_names]
        self.col_names_iter = org_names[0:3] + at_names_min + [org_names[3]] +  at_names_cum

        #Create a pd dataframe of all iteration information. Initialize cumulative columns as zero
        iter_df = pd.DataFrame(columns = self.col_names_iter)
        iter_df["Iter"] = iter_list
        iter_df["Min Obj"] = obj_list
        iter_df[at_names_min] = param_list
        iter_df = iter_df.apply(lambda col: col.explode(), axis=0).reset_index(drop=True).copy(deep =True)
        iter_df["Run"] = run + 1
        
        #Add Theta min obj and theta_sse obj
        #Loop over each iteration to create the min obj columns
        for j in range(len(iter_df)):
            min_sse = iter_df.loc[j, "Min Obj"].copy()
            if j == 0 or min_sse < iter_df["Min Obj Cum."].iloc[j-1]:
                min_param = iter_df.loc[j, at_names_min].copy().to_numpy()
            else:
                min_sse = iter_df["Min Obj Cum."].iloc[j-1].copy()
                min_param = iter_df.loc[j-1, at_names_cum].copy().to_numpy()

            iter_df.loc[j, "Min Obj Cum."] = min_sse
            iter_df.loc[j, at_names_cum] = min_param

            #Add iteration info from sse and mean weight
            sse_names = list(self.iter_sse_pieces[j].keys())
            wt_names = list(self.iter_mean_wt_pieces[j].keys())
            iter_df.loc[j, sse_names] = np.array(list(self.iter_sse_pieces[j].values()))
            iter_df.loc[j, wt_names] = np.array(list(self.iter_mean_wt_pieces[j].values()))

        #Find largest iter for each restart
        idx = iter_df.groupby('Run')['Iter'].idxmax()
        #Add following to only last iter runs    
        iter_df.loc[idx, "Run Time"] = time_per_run
        iter_df.loc[idx, "jac evals"] = solution.njev
        iter_df.loc[idx, "func evals"] = solution.nfev
        iter_df.loc[idx, "Optimality"] = scipy.linalg.norm(solution.jac, ord = np.inf) 
        iter_df.loc[idx, "Term Status"] = solution.status
        iter_df.loc[idx, "Message"] = solution.message
        # iter_df.loc[idx, "Optimality"] = solution.optimality

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
        param_inits = self.__get_params_and_df()
        ls_results = pd.DataFrame()

        #Optimize w/ retstarts
        for i in range(self.repeats):
            #Get Iteration results
            solution, time_per_run = self.__get_scipy_soln(i, param_inits)
            iter_df = self.__get_opt_iter_info(i, solution, time_per_run)

            #Append to results_df
            ls_results = pd.concat([ls_results, iter_df], ignore_index=True)

            #Reset iter lists and change seed
            self.seed += 1
            self.iter_param_data = []
            self.iter_obj_data = []
            self.iter_sse_pieces = []
            self.iter_mean_wt_pieces = []
            self.iter_count = 0

        #Reset the index of the pandas df
        ls_results = ls_results.reset_index(drop=True)

        #Sort by lowest obj first
        sort_ls_res = ls_results.sort_values(['Min Obj Cum.'], ascending= True)

        #Back out best job for each run
        run_best = sort_ls_res.groupby('Run').first().reset_index()
        best_runs = run_best.sort_values(['Min Obj Cum.'], ascending= True)

        if self.save_data:
            dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
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
    
    def approx_jac_hess(self, theta_guess):
        """
        Builds Jacobian Approximation
        """
        jac = optimize.approx_fprime(theta_guess, self.__scipy_min_fxn)
        hess = jac.T@jac
        return jac, hess
    
class Vis_Results(Problem_Setup):
    """
    Class For vizualizing GP and Optimization Results
    """

    #Inherit objects from General_Analysis
    def __init__(self, molec_data_dict, all_gp_dict, at_class, w_calc, obj_choice, save_data):
        #Asserts
        super().__init__(molec_data_dict, all_gp_dict, at_class, w_calc, obj_choice, save_data)
        

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
            train_data, test_data = self.get_train_test_data(molec, molec_gps_dict.keys())
            #Make pdf
            dir_name = self.make_results_dir(list(molec))
            pdf = PdfPages(dir_name + '/gp_val_figs.pdf')
            #Loop over gps (1 per property)
            for key in list(molec_gps_dict.keys()):
                #Set label
                label = molec + "_" + key
                #Get GP associated with property
                gp_model = molec_gps_dict[key]
                #Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                #Plot model performance
                pdf.savefig(plot_model_performance({label:gp_model}, test_data["x"], test_data[key], y_bounds))
                plt.close()

                #Plot temperature slices
                figs = plot_slices_temperature(
                    {label:gp_model},
                    molec_object.n_params,
                    molec_object.temperature_bounds,
                    y_bounds,
                    plot_bounds = molec_object.temperature_bounds,
                    property_name= y_names
                )

                for fig in figs:
                    pdf.savefig(fig)
                del figs

                #Plot Parameter slices
                for param_name in molec_object.param_names:
                    figs = plot_slices_params(
                        {label:gp_model},
                        param_name,
                        molec_object.param_names,
                        list(exp_data.keys())[2], #Use the 3rd temp to plot param slices
                        molec_object.temperature_bounds,
                        y_bounds,
                        property_name=y_names
                    )
                    plt.close()
                    
                    for fig in figs:
                        pdf.savefig(fig)
                    del figs

                #Plot test vs train for each parameter set
                for test_params in test_data["x"][:,:molec_object.n_params]:
                    #Find points in test set with correct param value
                    # Locate rows where parameter set == test parameter set
                    match_test = np.unique(np.where((test_data["x"][:,:molec_object.n_params] == test_params).all(axis=1))[0])
                    test_points = np.concatenate((test_data["x"][match_test,-1].reshape(-1,1), 
                                                  test_data[key][match_test].reshape(-1,1)), axis = 1)
                    #Find points in train set with correct param value
                    match_trn = np.unique(np.where((train_data["x"][:,:molec_object.n_params] == test_params).all(axis=1))[0])
                    train_points = np.concatenate((train_data["x"][match_trn,-1].reshape(-1,1), 
                                                  train_data[key][match_trn].reshape(-1,1)), axis = 1)

                    pdf.savefig(plot_model_vs_test({label:gp_model}, 
                                                test_params, 
                                                train_points, 
                                                test_points, 
                                                molec_object.temperature_bounds,
                                                y_bounds,
                                                plot_bounds = molec_object.temperature_bounds,
                                                property_name =  y_names ))
                    plt.close()
            pdf.close()

        return
    
    def compare_T_prop_best(self, theta_guess):
        """
        Compares T vs Property for a given set
        """
        #Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        pdf = PdfPages(dir_name + '/prop_vs_T.pdf')
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
                #Set label
                label = molec + "_" + key
                #Get GP associated with property
                gp_model = molec_gps_dict[key]
                #Get X and Y data and bounds associated with the GP
                exp_data, y_bounds, y_names = self.get_exp_data(molec_object, key)
                #Plot Exp Data vs Model Prediction
                pdf.savefig(plot_model_vs_exp(
                {label:gp_model},
                gp_theta_guess,
                exp_data,
                molec_object.temperature_bounds,
                y_bounds,
                plot_bounds=molec_object.temperature_bounds,
                property_name=y_names))
                plt.close()
        pdf.close()

    def make_sse_sens_data(self, theta_guess):
        """
        Makes heat map data for obj predictions given a parameter set
        theta_guess (eps/kb and nm)
        """
        n_points = 15
        #Create dict of heat map theta data
        param_dict = {}

        #Create a linspace for the number of dimensions and define number of points
        dim_list = np.linspace(0,len(theta_guess)-1,len(theta_guess)-1)
        #Create a list of all combinations (without repeats e.g no (1,1), (2,2)) of dimensions of theta
        mesh_combos = np.array(list(combinations(dim_list, 2)), dtype = int)

        #Meshgrid set always defined by n_points**2
        theta_set = np.tile(np.array(theta_guess), (n_points**2, 1))

        #Loop over all possible theta combinations of 2
        for i in range(len(mesh_combos)):
            #Create a copy of the true values to change the mehsgrid valus on
            theta_set_copy = np.copy(theta_set)
            #Set the indeces of theta_set for evaluation as each row of mesh_combos
            idcs = mesh_combos[i]
            #define name of parameter set as tuple ("param_1,param_2")
            data_set_name = (self.at_class.at_names[idcs[0]], self.at_class.at_names[idcs[1]])

            #Create a meshgrid of values of the 2 selected values of theta and reshape to the correct shape
            #Assume that theta1 and theta2 have equal number of points on the meshgrid
            # print(self.at_class.at_bounds, self.at_class.at_bounds.shape, idcs)
            theta1 = np.linspace(self.at_class.at_bounds[idcs[0]][0], self.at_class.at_bounds[idcs[0]][1], n_points)
            theta2 = np.linspace(self.at_class.at_bounds[idcs[1]][0], self.at_class.at_bounds[idcs[1]][1], n_points)
            theta12_mesh = np.array(np.meshgrid(theta1, theta2))
            theta12_vals = np.array(theta12_mesh).T.reshape(-1,2)
            
            #Set initial values for evaluation (true values) to meshgrid values
            theta_set_copy[:,idcs] = theta12_vals
            
            #Append data set to dictionary with name
            param_dict[data_set_name] = theta_set_copy

        #Initialize obj dictionary
        obj_dict = {}
        #Loop over each heat map
        for key, value in param_dict.items():
            #Evaluate obj over data
            obj_arr = np.zeros(len(value))
            for i in range(len(value)):
                #Values pref to real
                val_real = self.values_pref_to_real(value[i])
                obj_arr[i] = self.calc_obj(val_real, self.w_calc)[0]
            obj_dict[key] = obj_arr

        return param_dict, obj_dict
    
    def plot_obj_hms(self, theta_guess):
        """
        Plots objective contours given a set of data
        """
        #Get HM Data
        param_dict, obj_dict = self.make_sse_sens_data(theta_guess)
        #Make pdf
        dir_name = self.make_results_dir(list(self.molec_data_dict.keys()))
        pdf = PdfPages(dir_name + '/obj_contours.pdf')
        #Loop over keys
        for key in list(param_dict.keys()):
            #Get parameter and sse data
            theta_vals = param_dict[key]
            #Get index associated with params in key
            indcs = [self.at_class.at_names.index(key[0]), self.at_class.at_names.index(key[1])]
            n_points = int(np.sqrt(len(theta_vals)))
            obj_vals = obj_dict[key].reshape((n_points,n_points)).T
            #Remove unchanging columns
            theta_mesh = theta_vals[:, indcs].reshape((n_points,n_points, -1)).T
            pdf.savefig(plot_obj_contour(
                    theta_mesh,
                    theta_guess[indcs],
                    obj_vals,
                    key
                ))
            plt.close()
        pdf.close()
        