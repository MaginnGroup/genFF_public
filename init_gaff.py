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

#Base on parameters.py
def init_project():

    # Initialize project
    project = signac.init_project("gaff")

    smiles = "CC"
    # mol_weight = 30.069 * u.amu
    n_atoms = 8

    # Thermodynamic conditions for simulation

    #Get simulation temperatures from class constants (and multiply all by u.K)


    # Reference data to compare to (i.e. experiments or other simulation studies)
    reference_data = [
    (290.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 35.159 * u.bar),
    (270.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 22.10 * u.bar),
    (250.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 13.008 * u.bar),
    (230.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 7.0018 * u.bar),
    (210.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 3.338 * u.bar),
    ]

    for temp in temps: 

        state_point = {
            "T": float(temp.in_units(u.K).value),
        }

        job = project.open_job(state_point)
        job.init()


    for temp in temps:
        # Define the state point
        state_point = {
            "mol_name": "R41",
            "mol_weight": R41.molecular_weight,
            "smiles": smiles,
            "T": float(temp.in_units(u.K).value),
           
            "sigma_C1": float((sigma_C1 * u.Angstrom).in_units(u.nm).value),
            "sigma_F1": float((sigma_F1 * u.Angstrom).in_units(u.nm).value),
            "sigma_H1": float((sigma_H1 * u.Angstrom).in_units(u.nm).value),
            "epsilon_C1": float((epsilon_C1 * u.K * u.kb).in_units("kJ/mol").value),
            "epsilon_F1": float((epsilon_F1 * u.K * u.kb).in_units("kJ/mol").value),
            "epsilon_H1": float((epsilon_H1 * u.K * u.kb).in_units("kJ/mol").value),
            "N_atoms": n_atoms,
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