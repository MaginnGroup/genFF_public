#Imports
from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types
import numpy as np
import unyt as u

#Set params for saving results and whether obj wts are scaled
save_data = True
scl_w = 2

#Load class properies for each molecule
r14_class = r14.R14Constants()
r32_class = r32.R32Constants()
r50_class = r50.R50Constants()
r125_class = r125.R125Constants()
r134a_class = r134a.R134aConstants()
r143a_class = r143a.R143aConstants()
r170_class = r170.R170Constants()

#Get dict of refrigerant classes to consider, gps, and atom typing class
molec_data_dict = {"R32":r32_class}

at_class = atom_type.AT_Scheme_7()
all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
#Best set from Experiment
best_set = np.array([3.830327452839443,2.9719813341938197,2.156373165481484,2.6941766822509,3.8523628534129073,3.837466871267212,53.22755465997726,72.61206434864629,8.547339075792452,33.29151423562465,19.727770531635596,32.69817579869139])
visual = opt_atom_types.Vis_Results(molec_data_dict, all_gp_dict, at_class, scl_w, save_data)
visual.plot_obj_hms(best_set)