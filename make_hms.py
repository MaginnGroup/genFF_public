#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u

#Set params for saving results and whether obj wts are scaled
save_data = True
scl_w = 2
opt_choice = "SSE"

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
all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
#Best set from Experiment
best_set = np.array([3.4651788382627764,3.9218070855238745,3.9999999999999996,1.691292101355337,2.290823592634345,2.948836531411303,3.8698590079226696,50.74628450631082,54.58103944130325,75.00000000000001,10.000000000000002,50.00000000000001,50.00000000000001,50.00000000000001])
visual = opt_atom_types.Vis_Results(molec_data_dict, all_gp_dict, at_class, scl_w, opt_choice, save_data)
visual.plot_obj_hms(best_set)