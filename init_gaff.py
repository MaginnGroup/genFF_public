import signac
import numpy as np
import unyt as u
import sys
import pandas as pd

sys.path.append("../../analysis/")
from fffit.utils import values_scaled_to_real
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types

at_number = 11
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
R23 = r23.R23Constants()
R161 = r161.R161Constants()
R152a = r152a.R152aConstants()
R152 = r152.R152Constants()
R143 = r143.R143Constants()
R134 = r134.R134Constants()
R116 = r116.R116Constants()

molec_dict = {"R14": R14,
                "R32": R32,
                "R50": R50,
                "R125": R125,
                "R134a": R134a,
                "R143a": R143a,
                "R170": R170,
                "R41": R41,
                "R23": R23,
                "R161":R161,
                "R152a":R152a,
                # "R152": R152,
                # "R143": R143,
                # "R134": R134,
                "R116": R116}


#Base on parameters.py
def init_project():

    # Initialize project
    project = signac.init_project("gaff_ff_ms")

    #Loop over all molecules
    for molec_name, molec_data in molec_dict.items():
        # Define temps (from constants files)
        temps = list(molec_data.expt_Pvap.keys())

        at_class = atom_type.make_atom_type_class(at_number)

        # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in project_gaff.py as needed)
        # reference_data = [
        # (290.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 35.159 * u.bar),
        # (270.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 22.10 * u.bar),
        # (250.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 13.008 * u.bar),
        # (230.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 7.0018 * u.bar),
        # (210.0 * u.K, 206.18 * u.kilogram/(u.meter)**3, 206.18 * u.kilogram/(u.meter)**3, 3.338 * u.bar),
        # ]


        for temp in temps:
            # Define the state point
            state_point = {
                "mol_name": molec_name,
                "mol_weight": molec_data.molecular_weight.in_units(u.amu).value,
                "smiles": molec_data.smiles_str,
                "N_atoms": molec_data.n_atoms,
                "T": float(temp.in_units(u.K).value),
                "N_vap": n_vap,
                "N_liq": n_liq,
                "expt_liq_density": molec_data.expt_liq_density[int(temp.in_units(u.K).value)],
                "nsteps_nvt": 2500000,
                "nsteps_npt": 2000,
                "nsteps_gemc_eq":10000000,
                "nsteps_gemc_prod": 25000000,
            }            

            job = project.open_job(state_point)
            job.init()