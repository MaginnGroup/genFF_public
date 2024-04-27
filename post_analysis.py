#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy

#Set params for saving results and whether obj wts are scaled
save_data = True
opt_choice = "ExpVal"
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
                   "R143a":r143a_class}

all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
visual = opt_atom_types.Vis_Results(molec_data_dict, all_gp_dict, at_class, opt_choice)

#Best set from Experiment
#Set parameter set of interest (in this case get the best parameter set)
all_molec_dir = visual.make_results_dir(list(molec_data_dict.keys()))
path_best_sets = os.path.join(all_molec_dir, "best_per_run.csv")
assert os.path.exists(path_best_sets), "best_per_run.csv not found in directory"
unsorted_df = pd.read_csv(path_best_sets, header = 0)
#Sort df and overwrite it with sorted df
all_df = unsorted_df.sort_values(by = "Min Obj")
all_df.to_csv(path_best_sets, index=False)
first_param_name = visual.at_class.at_names[0] + "_min"
last_param_name = visual.at_class.at_names[-1] + "_min"
best_set = all_df.loc[0, first_param_name:last_param_name].values
best_real = visual.values_pref_to_real(copy.copy(best_set))
x_label = "best_set"

#Get Property Predictions for all training molecules
molec_names = list(molec_data_dict.keys())
# visual.comp_paper_full_ind(molec_names)

#Calculate MAPD for predictions and save results
df = visual.calc_MAPD_best(molec_names, save_data, save_label=x_label)

#Gat Jac and Hess Approximations
jac = visual.approx_jac(best_real, save_data, x_label=x_label)
hess = visual.approx_hess(best_real, save_data, x_label=x_label)

#Plot optimization result heat maps
visual.plot_obj_hms(best_set, x_label)