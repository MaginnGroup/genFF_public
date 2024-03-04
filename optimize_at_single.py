#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u

#Set params for saving results, # of repeats, and the seed
save_data = True
repeats = 20
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
molec_data_dict = {"R14":r14_class, 
                   "R32":r32_class, 
                   "R50":r50_class, 
                   "R170":r170_class, 
                   "R125":r125_class, 
                   "R134a":r134a_class, 
                   "R143a":r143a_class}

at_class = atom_type.AT_Scheme_9()

#Loop over all molecules seperately
for k, v in molec_data_dict.items():
    molec_data_dict_1_mol = {k: v}
    all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict_1_mol.keys()))
    
    #Loop over all 3 ways to calculate weight and optimize
    for w_scheme in [0,1,2]:
        #Optimize AT scheme parameters
        driver = opt_atom_types.Opt_ATs(molec_data_dict_1_mol, all_gp_dict, at_class, repeats, seed, w_scheme, save_data)
        ls_results = driver.optimize_ats()