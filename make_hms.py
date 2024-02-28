#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u

#Set params for saving results and whether obj wts are scaled
save_data = True
scl_w = False

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

at_class = atom_type.AT_Scheme_7()
all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
#Set best_set from scl_w experiment
# best_set = np.array([2.89631543, 4.0, 1.5, 3.1945792, 2.23150945,  3.07915865, 75.0, 75.0, 9.9999988, 29.9003147, 50.0, 50.0])
#Best set from not scl_wt exp
best_set = np.array([3.0019590836418883,3.9999999999999996,1.5,3.2313895507940393,3.3941233247870666,2.927011182373593,38.33283581744562,75.00000000000001,9.999998797276383,50.00000000000001,50.00000000000001,36.59782932335614])
visual = opt_atom_types.Vis_Results(molec_data_dict, all_gp_dict, at_class, scl_w, save_data)
visual.plot_obj_hms(best_set)