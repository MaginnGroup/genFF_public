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
    return job.isfile("best_per_run.csv")

@Project.post(results_computed)
@Project.operation(with_job = True)
def run_obj_alg(job):
    #Define method, ep_enum classes, indecies to consider, and kernel
    training_molecules = job.sp.training_molecules
    try:
        training_molecules = json.loads(training_molecules)
    except:
        training_molecules = job.sp.training_molecules
    
    #Set params for saving results, # of repeats, and the seed
    obj_choice = job.sp.obj_choice
    repeats  = job.sp.repeats
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

    #Create param sets for the AT optimization

    all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
    driver = opt_atom_types.Opt_ATs(molec_data_dict, all_gp_dict, at_class, repeats, seed, obj_choice, save_data)
    #Optimize AT scheme parameters
    ls_results = driver.optimize_ats()
    
    #Save results for best set for each run and iter to a csv file in Results
    #Store intermediate results in job directory
    #Ensure file exists
    dir_name = driver.make_results_dir(training_molecules)
    save_path_job1 = job.fn("test" + ".csv")
    #Save results
    driver.save_results(ls_results, dir_name)

if __name__ == "__main__":
    Project().main()