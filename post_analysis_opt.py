#Imports
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy
import scipy 
import signac

#Set params for saving results and whether obj wts are scaled
save_data = True
obj_choice = "ExpVal"
at_number = 11
seed = 1

#Load class properies for each molecule
r14_class = r14.R14Constants()
r32_class = r32.R32Constants()
r50_class = r50.R50Constants()
r125_class = r125.R125Constants()
r134a_class = r134a.R134aConstants()
r143a_class = r143a.R143aConstants()
r170_class = r170.R170Constants()

#Get dict of refrigerant classes to consider, gps, and atom typing class
molec_names = ["R14", "R32", "R50", "R170", "R125", "R134a", "R143a"]

project = signac.get_project("opt_at_params")
filtered_jobs = project.find_jobs({"obj_choice": obj_choice, "atom_type": at_number})
grouped_jobs = filtered_jobs.groupby(statepoint="training_molecules")
for statepoint_value, group in grouped_jobs.items():
    #Create dataframe of all results in job.fn("best_run.csv")
    unsorted_df = pd.DataFrame()
    for i,job in enumerate(group):
        #On the 1st iteration, save the path to the job directory
        if i == 0:
            save_path = job.document.dir_name
        df_best_run = pd.read_csv(job.fn("best_run.csv"), header = 0, index_col=False)
        all_df = pd.concat([all_df, df_best_run], ignore_index=True)
    all_df = unsorted_df.sort_values(by='Min Obj', ascending = True).reset_index(drop = True)
    #Save all the best sets in appropriate folder for each set of training molecules
    all_df.to_csv(os.path.join(save_path, "best_per_run.csv"), index=False)

#Best set from Experiment
visual = opt_atom_types.Vis_Results(molec_names, at_number, seed, obj_choice)
#Set parameter set of interest (in this case get the best parameter set)
all_molec_dir = visual.use_dir_name
path_best_sets = os.path.join(all_molec_dir, "best_per_run.csv")
assert os.path.exists(path_best_sets), "best_per_run.csv not found in directory"
all_df = pd.read_csv(path_best_sets, header = 0)
first_param_name = visual.at_class.at_names[0] + "_min"
last_param_name = visual.at_class.at_names[-1] + "_min"
best_set = all_df.loc[0, first_param_name:last_param_name].values
best_real = visual.values_pref_to_real(copy.copy(best_set))
x_label = "best_set"

#Get Property Predictions for all training molecules
molec_names_all = list(visual.all_train_molec_data.keys())
visual.comp_paper_full_ind(molec_names_all, save_label=x_label)

#Calculate MAPD for predictions and save results
df = visual.calc_MAPD_best(molec_names_all, save_data, save_label=x_label)

#Gat Jac and Hess Approximations
scale_theta = True
jac = visual.approx_jac(best_real, scale_theta, save_data, x_label=x_label)
hess = visual.approx_hess(best_real, scale_theta, save_data, x_label=x_label)
eigval, eigvec = scipy.linalg.eig(hess)
if save_data == True:
    eig_val_path = os.path.join(all_molec_dir, "Hess_EigVals")
    eig_vec_path = os.path.join(all_molec_dir, "Hess_EigVecs")
    eigval = [np.real(num) for num in eigval]
    np.savetxt(eig_val_path,eigval,delimiter=",")
    np.savetxt(eig_vec_path,eigvec,delimiter=",")

#Plot optimization result heat maps
visual.plot_obj_hms(best_set, x_label)