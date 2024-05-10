#Imports
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy
import scipy 
from utils.analyze_ms import prepare_df_vle, prepare_df_vle_errors, plot_vle_envelopes, plot_pvap_hvap

#After jobs are finished
#save signac results for each atom for a given atom typing scheme and number of training parameters
import signac
import sys

from fffit.signac import save_signac_results

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

at_number = 11
for project_path in ["gaff_ff_ms", "opt_ff_ms"]:
    project = signac.get_project(project_path)
    if project_path == "opt_ff_ms":
        project = project.find_jobs({"atom_type": at_number})
        at_num_str = str(at_number)
    else:
        at_num_str = ""
    csv_root = os.path.join("Results_MS", at_num_str, project_path)
    os.makedirs(csv_root, exist_ok=True)

    #Create a large df with just the molecule name and property predictions (param values calculated in Results)
    property_names = [
        "liq_density",
        "vap_density",
        "Hvap",
        "Pvap",
        "liq_enthalpy",
        "vap_enthalpy",
    ]

    #Get data from molecular simulations. Group by molecule name
    df_molec = save_signac_results(project, "mol_name", property_names, csv_name=csv_root)
    
#Load csvs for Opt_FF, GAFF, NW, Trappe, and Potoff
#Check that files all exist and load them if they do
# df_Potoff = open("Results_MS").read() if os.path.exists(file_path) else None

# #For each FF
#Use prepare_df_vle to get the data in the correct format and save the data
df_all = prepare_df_vle(df_molec, molec_dict, csv_name=csv_root)
#Calculate MAPD and MSE for each T point
df_paramsets = prepare_df_vle_errors(df_all, molec_dict, csv_name = csv_root)

#For each molecule
molecules = df_paramsets['molecule'].unique().tolist()
for molec in molecules:
    #Get the data for the molecule from each FF if it exists

    #Use prepare_df_vle to get the data in the correct format
    one_molec_dict = {molec: molec_dict[molec]}
    df_molec = copy.copy(df_all[df_all['molecule'] == molec])
    #Plot Pvap and Hvap vs T and compare the 5 FF predictions
    plot_vle_envelopes(df_molec, df_all, df_lit = None, df_nw = None, df_trappe = None, df_gaff = None, save_name = None)
    plot_pvap_hvap(df_molec, df_all, df_lit = None, df_nw = None, df_trappe = None, df_gaff = None, save_name = None)
    
