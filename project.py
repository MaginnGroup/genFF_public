# project.py
import signac
from pathlib import Path
import os
import json
import flow
from flow import FlowProject, directives

#Import dependencies
import numpy as np
import templates.ndcrc
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_obj_contour
import pickle
import gzip

#Ignore warnings caused by "nan" values
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)

#Ignore warning from scikit learn hp tuning
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class Project(FlowProject):
    def __init__(self):
        super().__init__()
        current_path = Path(os.getcwd()).absolute()

@Project.label
def results_computed(job):
    #Write script that checks whether the intermediate job files are there
    return job.isfile("opt_at_results.csv") and job.isfile("sorted_at_res.csv")

@Project.post(results_computed)
@Project.operation(with_job = True)
def run_obj_alg(job):
    #Define method, ep_enum classes, indecies to consider, and kernel
    training_molecules = job.sp.training_molecules
    try:
        training_molecules = json.loads(training_molecules)
    except:
        training_molecules = list([job.sp.training_molecules])
    
    #Set params for saving results, # of repeats, and the seed
    obj_choice = job.sp.obj_choice
    total_repeats  = job.sp.total_repeats
    repeat_num = job.sp.repeat_number
    seed = job.sp.seed
    save_data = job.sp.save_data
    at_class = atom_type.make_atom_type_class(job.sp.atom_type)

    #Load class properies for each trainingmolecule
    r14_class = r14.R14Constants()
    r32_class = r32.R32Constants()
    r50_class = r50.R50Constants()
    r125_class = r125.R125Constants()
    r134a_class = r134a.R134aConstants()
    r143a_class = r143a.R143aConstants()
    r170_class = r170.R170Constants()

    #Get dict of refrigerant classes to consider, gps, and atom typing class
    all_molec_data_dict = {"R14":r14_class, 
                    "R32":r32_class, 
                    "R50":r50_class, 
                    "R170":r170_class, 
                    "R125":r125_class, 
                    "R134a":r134a_class, 
                    "R143a":r143a_class}
    
    #Make molec data dict given molecules:
    molec_data_dict = {}
    for molec in training_molecules:
        if molec in list(all_molec_data_dict.keys()):
            molec_data_dict[molec] = all_molec_data_dict[molec]

    all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
    driver = opt_atom_types.Opt_ATs(molec_data_dict, all_gp_dict, at_class, total_repeats, seed, obj_choice)

    #Create param sets for the AT optimization based on seed and such
    param_inits = driver.get_param_inits()
    #Get the correct parameter set from the param_inits based on which repeat we are evaluating
    param_guess = param_inits[repeat_num-1].reshape(1,-1)
    #Optimize atom types
    ls_results, sort_ls_res, best_runs = driver.optimize_ats(param_guess, repeat_num-1)
    
    if job.sp.save_data == True:
        #Save results for best set for each run and iter to a csv file in Results
        #Ensure directory exists
        dir_name = driver.make_results_dir(training_molecules)
        os.makedirs(dir_name, exist_ok=True) 
        save_path3 = os.path.join(dir_name, "best_per_run.csv")
        #Save results. Append to file if it already exists
        if os.path.exists(save_path3):
            best_runs.to_csv(save_path3, mode = "a", index = False, header = False)
        else:
            best_runs.to_csv(save_path3, index = False)

    #Store intermediate results in job directory
    #Save original results
    save_path1 = job.fn("opt_at_results.csv")
    ls_results.to_csv(save_path1, index = False)
    #Save sorted results
    save_path2 =  job.fn("sorted_at_res.csv")
    sort_ls_res.to_csv(save_path2, index = False)

if __name__ == "__main__":
    Project().main()