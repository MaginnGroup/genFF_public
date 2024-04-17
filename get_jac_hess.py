#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd

#Set params for saving results, # of repeats, and the seed
save_data = True
w_scheme = 2
obj_choice = "ExpVal"
at_class = atom_type.AT_Scheme_9()

#Load class properies for each molecule
r14_class = r14.R14Constants()
r32_class = r32.R32Constants()
r50_class = r50.R50Constants()
r125_class = r125.R125Constants()
r134a_class = r134a.R134aConstants()
r143a_class = r143a.R143aConstants()
r170_class = r170.R170Constants()

#Get dict of refrigerant classes to consider, gps, and atom typing class
molec_data_dict = {"R14":r14_class, 
                   "R32":r32_class, 
                   "R50":r50_class, 
                   "R170":r170_class, 
                   "R125":r125_class, 
                   "R134a":r134a_class, 
                   "R143a":r143a_class}

all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
driver = opt_atom_types.Problem_Setup(molec_data_dict, all_gp_dict, at_class, w_scheme, obj_choice, save_data)

#Set parameter set of interest (in this case get the best parameter set)
all_molec_dir = driver.make_results_dir(list(molec_data_dict.keys()))
all_df = pd.read_csv(all_molec_dir+"/best_per_run.csv", header = 0)
first_param_name = driver.at_class.at_names[0] + "_min"
last_param_name = driver.at_class.at_names[-1] + "_min"
full_opt_best = all_df.loc[0, first_param_name:last_param_name].values
best_real = driver.values_pref_to_real(full_opt_best)
x_label = "best_set"

#Optimize AT scheme parameters
jac = driver.approx_jac(best_real, save_data, x_label=x_label)
hess = driver.approx_hess(best_real, save_data, x_label=x_label)