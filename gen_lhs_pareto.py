from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_obj_contour
import os

#Load class properies for each molecule
r14_class = r14.R14Constants()
r32_class = r32.R32Constants()
r50_class = r50.R50Constants()
r125_class = r125.R125Constants()
r134a_class = r134a.R134aConstants()
r143a_class = r143a.R143aConstants()
r170_class = r170.R170Constants()

r41_class = r41.R41Constants()
r23_class = r23.R23Constants()
r161_class = r161.R161Constants()
r152a_class = r152a.R152aConstants()
r152_class = r152.R152Constants()
r143_class = r143.R143Constants()
r134_class = r134.R134Constants()
r116_class = r116.R116Constants()

samples = 10**5
#Get obj from a set of parameters
at_class = atom_type.AT_Scheme_11()
save_data = True
obj_choice = "ExpVal"
molec_data_dict = {"R14":r14_class, "R32":r32_class, "R50":r50_class, "R170":r170_class, "R125":r125_class, "R134a":r134a_class, "R143a":r143a_class}
all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
problem_setup = opt_atom_types.Problem_Setup(molec_data_dict, all_gp_dict, at_class, obj_choice)
pareto_info = problem_setup.gen_pareto_sets(samples, at_class.at_bounds_nm_kjmol, save_data= save_data)