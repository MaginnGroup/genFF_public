import signac
import numpy as np
import unyt as u
import sys
import pandas as pd

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types

n_vap = 160
n_liq = 640

def init_project():

    # Initialize project
    project = signac.init_project("gromacs")

    # Define temps
    temps = [
        210.0 * u.K,
        230.0 * u.K,
        250.0 * u.K,
        270.0 * u.K,
        290.0 * u.K
    ]

    # Run at vapor pressure
    press = {
        210: (2.1852 * u.bar),
        230: (5.0928 * u.bar),
        250: (10.296 * u.bar),
        270: (18.740 * u.bar),
        290: (31.548 * u.bar),
    }

    n_vap = 160 # number of molecules in vapor phase
    n_liq = 640

    #Load class properies for each training and testing molecule
    R14 = r14.R14Constants()
    R32 = r32.R32Constants()
    R50 = r50.R50Constants()
    R125 = r125.R125Constants()
    R134a = r134a.R134aConstants()
    R143a = r143a.R143aConstants()
    R170 = r170.R170Constants()

    R41 = r41.R41Constants()
    r23_class = r23.R23Constants()
    r161_class = r161.R161Constants()
    r152a_class = r152a.R152aConstants()
    r152_class = r152.R152Constants()
    r143_class = r143.R143Constants()
    r134_class = r134.R134Constants()
    r116_class = r116.R116Constants()

    # Load sample from best set
    at_class = atom_type.AT_Scheme_11()
    save_data = False
    obj_choice = "ExpVal"
    molec_data_dict = {"R14":R14, "R32":R32, "R50":R50, "R170":R170, "R125":R125, "R134a":R134a, "R143a":R143a}
    all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
    setup = opt_atom_types.Problem_Setup(molec_data_dict, all_gp_dict, at_class, obj_choice)
    all_molec_dir = setup.make_results_dir(list(molec_data_dict.keys()))
    all_df = pd.read_csv(all_molec_dir+"/best_per_run.csv", header = 0)
    first_param_name = setup.at_class.at_names[0] + "_min"
    last_param_name = setup.at_class.at_names[-1] + "_min"
    full_opt_best = all_df.loc[:, first_param_name:last_param_name]

    # Convert to units of nm and kJ/mol
    param_matrix = at_class.get_transformation_matrix({"R41":R41})
    all_best_real = setup.values_pref_to_real(full_opt_best.values)
    scaled_params = all_best_real.reshape(-1,1).T@param_matrix

    for temp in temps:
        for sample in scaled_params:

            # Unpack the sample (CHECK ME!!!)
            (sigma_C1, sigma_F1, sigma_H1, epsilon_C1, epsilon_F1, epsilon_H1) = sample

            # Define the state point
            state_point = {
                "mol_name": "R41",
                "mol_weight": R41.molecular_weight,
                "smiles": "CF",
                "T": float(temp.in_units(u.K).value),
                "P": float(press[int(temp.in_units(u.K).value)].in_units(u.bar).value),
                "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
                "sigma_F1": float((sigma_F1 * u.Angstrom).in_units(u.nm).value),
                "sigma_H1": float((sigma_H1 * u.Angstrom).in_units(u.nm).value),
                "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_F1": float((epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value),
                "epsilon_H1": float((epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value),
                "N_atoms": 5,
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": R41.expt_liq_density[
                    int(temp.in_units(u.K).value)
                ],
                "nsteps_liqeq": 5000,
                "nsteps_eq": 10000,
                "nsteps_prod": 100000,
            }            

            job = project.open_job(state_point)
            job.init()