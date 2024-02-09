from utils import r14, r32, r50, r125, r134a, r143a, r170, atom_type, opt_atom_types

#Set params
save_res = True
seed = 1

#Load class properies for each molecule
r14 = r14.R14Constants()
r32 = r32.R32Constants()
r50 = r50.R50Constants()
r125 = r125.R125Constants()
r134a = r134a.R134aConstants()
r143a = r143a.R143aConstants()
r170 = r170.R170Constants()
#Get dict of refrigerant classes to consider, gps, and atom typing class
molec_data_dict = {"R14":r14, "R32":r32, "R50":r50, "R170":r170, "R125":r125, "R134a":r134a, "R143a":r143a}
all_gp_dict = opt_atom_types.get_gp_data_from_pkl(list(molec_data_dict.keys()))
at_class = atom_type.AT_Scheme_7()
#Set repeats and optimize
repeats = 50
ls_results = opt_atom_types.optimize_ats(repeats, at_class, molec_data_dict, all_gp_dict, save_res, seed)