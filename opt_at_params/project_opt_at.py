#Import dependencies
from flow import FlowProject, directives
import templates.ndcrc
import warnings
from pathlib import Path
import os
import json
import sys

# simulation_length must be consistent with the "units" field in custom args below
# For instance, if the "units" field is "sweeps" and simulation_length = 1000, 
# This will run a total of 1000 sweeps 
# (1 sweep = N steps, where N is the total number of molecules (job.sp.N_vap + job.sp.N_liq)
sys.path.append("..")
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types
sys.path.remove("..")

#Ignore warnings caused by "nan" values
import warnings
from warnings import simplefilter
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=UserWarning)
#Ignore warning from scikit learn hp tuning
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

class ProjectOPT(FlowProject):
    def __init__(self):
        current_path = Path(os.getcwd()).absolute()
        #Set Project Path to be that of the current working directory
        super().__init__(path = current_path)

@ProjectOPT.label
def results_computed(job):
    #Write script that checks whether the intermediate job files are there
    return job.isfile("opt_at_results.csv") and job.isfile("sorted_at_res.csv")

@ProjectOPT.post(results_computed)
@ProjectOPT.operation()
def run_obj_alg(job):
    #Define method, ep_enum classes, indecies to consider, and kernel
    training_molecules = job.sp.training_molecules
    training_molecules = list(json.loads(training_molecules))
    
    #Set params for saving results, # of repeats, and the seed
    obj_choice = job.sp.obj_choice
    total_repeats  = job.sp.total_repeats
    repeat_num = job.sp.repeat_number
    seed = job.sp.seed
    save_data = job.sp.save_data

    #Get all gp data and make driver class
    driver = opt_atom_types.Opt_ATs(training_molecules, job.sp.atom_type, total_repeats, seed, obj_choice)

    if job.sp.obj_choice == "ExpValPrior":
        #Save weight scaler to job document
        job.doc["weight_sclr"] = driver.weight_sclr

    #Create param sets for the AT optimization based on seed. 
    # Save these to the Results folder in the directory above for reuse
    param_inits = driver.get_param_inits()

    #Get data if the repeat number is less than or equal to the number of param_inits
    if repeat_num <= len(param_inits):
        #Get the correct parameter set from the param_inits based on which repeat we are evaluating
        param_guess = param_inits[repeat_num-1].reshape(1,-1)
        #Optimize atom types
        ls_results, sort_ls_res, best_runs = driver.optimize_ats(param_guess, repeat_num-1)
        
        dir_name = driver.use_dir_name
        job.document.dir_name = str(dir_name)

        #Store intermediate results in job directory
        #Save original results
        save_path3 = job.fn("best_run.csv")
        best_runs.to_csv(save_path3, index = False)
        save_path1 = job.fn("opt_at_results.csv")
        ls_results.to_csv(save_path1, index = False)
        #Save sorted results
        save_path2 =  job.fn("sorted_at_res.csv")
        sort_ls_res.to_csv(save_path2, index = False)
    #If the repeat number is greater than the number of param_inits, then we have already done all the repeats. Delete the job
    else:
        print("Repeat number is greater than the number of param_inits. Deleting job.")
        job.remove()

if __name__ == "__main__":
    ProjectOPT().main()