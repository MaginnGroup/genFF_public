#Imports
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy 
from utils.analyze_ms import prepare_df_vle, prepare_df_vle_errors, plot_vle_envelopes,plot_pvap_hvap, plot_err_each_prop, plot_err_avg_props

#After jobs are finished
#save signac results for each atom for a given atom typing scheme and number of training parameters
import signac
import sys

from fffit.fffit.signac import save_signac_results

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

obj_choice = "ExpVal"
at_numbers = [1,2,6,8]
param_set = 1
ff_dict = {}
err_path_dict = {}
err_labels = []
#Dictionary of all molecules
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
                "R152": R152,
                "R143": R143,
                "R134": R134,
                "R116": R116}

def get_mse_data(at_num_str, obj_choice_str, param_set_str, molec_dict, project_path):
    csv_root_unproc = os.path.join("Results_MS","unprocessed_csv", at_num_str, obj_choice_str, param_set_str)
    csv_root_final = os.path.join("Results_MS", at_num_str, obj_choice_str, param_set_str)
    os.makedirs(csv_root_unproc, exist_ok=True)
    os.makedirs(csv_root_final, exist_ok=True)

    #Create a large df with just the molecule name and property predictions (param values calculated in Results)
    property_names = [
        "liq_density",
        "vap_density",
        "Hvap",
        "Pvap",
        "liq_enthalpy",
        "vap_enthalpy",
    ]

    #Get data from molecular simulations. Group by molecule name and save
    csv_name_unproc = os.path.join(csv_root_unproc, project_path)
    if os.path.exists(csv_name_unproc + ".csv"):
        df_molec = pd.read_csv(csv_name_unproc + ".csv", index_col=False)
    else:
        df_molec = save_signac_results(project, "mol_name", property_names, csv_name= csv_name_unproc + ".csv")
    #process data and save
    csv_name_final = os.path.join(csv_root_final, project_path)
    df_all = prepare_df_vle(df_molec, molec_dict, csv_name=csv_name_final + ".csv", drop_one = True)
    
    #Calculate MAPD and MSE for each T point
    df_paramsets = prepare_df_vle_errors(df_all, molec_dict, csv_name = csv_name_final + "_err.csv")
    csv_name_final_err = csv_name_final + "_err.csv"

    return df_all, csv_name_final_err, df_paramsets

#Load csvs for Opt_FF
project_path = "opt_ff_ms"
project_all = signac.get_project(project_path)
for at_number in at_numbers:
    project = project_all.find_jobs({"atom_type": at_number, "obj_choice": obj_choice, "param_set": param_set})
    at_class = atom_type.make_atom_type_class(at_number)
    at_num_str = at_class.scheme_name
    obj_choice_str = obj_choice
    param_set_str = "param_set_" + str(param_set)
    df_all, csv_name, df_paramsets = get_mse_data(at_num_str, obj_choice_str, param_set_str, molec_dict, project_path)
    #Add column to df_all for atom type
    df_all['atom_type'] = at_number
    ff_dict[at_class.scheme_plot_name] = df_all
    err_path_dict[project_path + str(at_number)]= csv_name
    err_labels.append(at_class.scheme_plot_name)

#Load csvs for GAFF, NW, Trappe, and Potoff, and BBFF
project_path = "gaff_ff_ms"
project = signac.get_project(project_path)
at_num_str = ""
obj_choice_str = ""
param_set_str = ""
df_all, csv_name, df_paramsets = get_mse_data(at_num_str, obj_choice_str, param_set_str, molec_dict, project_path)
df_all['atom_type'] = "GAFF"
ff_dict["GAFF"] = df_all
err_labels.append("GAFF")
err_path_dict[project_path]= csv_name

#Load csvs for Opt_FF, GAFF, NW, Trappe, and Potoff, and BBFF
ff_names = ["Potoff", "TraPPE", "Wang_FFO", "BBFF"]
ff_labels = ["Potoff et al.", "TraPPE", "Previous Work", ""]
for ff_name, ff_label in zip(ff_names, ff_labels):
    #Check that files all exist and load them if they do
    read_path = os.path.join("Results_MS/unprocessed_csv", ff_name + ".csv")
    df_simple = pd.read_csv(read_path) if os.path.exists(read_path) else None
    #Use prepare_df_vle to get the data in the correct format and save the data
    csv_path_final = os.path.join("Results_MS", ff_name)
    df_ff_final = prepare_df_vle(df_simple, molec_dict, csv_name=csv_path_final + ".csv")
    if ff_name in  ["Wang_FFO", "BBFF"]:
        df_paramsets_w = prepare_df_vle_errors(df_ff_final, molec_dict, csv_name = csv_path_final + "_err.csv")
        err_path_dict[ff_name]= csv_path_final + "_err.csv"
        err_labels.append(ff_label)
    if df_ff_final is not None:
        df_ff_final['atom_type'] = ff_name
    ff_dict[ff_label] = df_ff_final
      
#If one atom type
if len(at_numbers) == 1:
    full_at_dir = os.path.join("Results_MS", at_class.scheme_name, obj_choice, "aparam_set_" + str(param_set))
else:
    full_at_dir = os.path.join("Results_MS", "AT-" + "".join(map(str, at_numbers)), obj_choice)
os.makedirs(full_at_dir, exist_ok=True)
pdf_vle = PdfPages(os.path.join(full_at_dir ,"vle.pdf"))
pdf_hpvap = PdfPages(os.path.join(full_at_dir ,"h_p_vap.pdf"))
#For each molecule
molecules = df_paramsets['molecule'].unique().tolist()
for molec in molecules:
    #Get the data for the molecule from each FF if it exists
    one_molec_dict = {molec: molec_dict[molec]}
    ff_molec_dict = {}
    for df_label, df_ff in ff_dict.items():
        df_molec = copy.copy(df_ff[df_ff['molecule'] == molec])
        if df_molec.empty:
            df_molec = None
        ff_molec_dict[df_label] = df_molec
    #Get the data for the molecule from each FF
    # print(ff_molec_dict)
    #Plot Vle, Hvap, and Pvap and save to different pdfs
    if molec not in ["R134", "R152"]:
        pdf_vle.savefig(plot_vle_envelopes(one_molec_dict, ff_molec_dict), bbox_inches='tight')
        # plt.show()
        plt.close()
        pdf_hpvap.savefig(plot_pvap_hvap(one_molec_dict, ff_molec_dict), bbox_inches='tight')
        plt.close()
#Close figures    
pdf_vle.close()
pdf_hpvap.close()

df_err_dict = {}
molec_names = ["R14", "R32", "R50", "R125", "R134a", "R143a", "R170", "R41", "R23", "R161", "R152a",  "R143",  "R116"]
for label, key in zip(err_labels, list(err_path_dict.keys())):
    df_err = pd.read_csv(err_path_dict[key], header = 0, index_col = "molecule")
    df_err_dict[label] = df_err.reindex(molec_names)

error_objs = ["mapd"]
for error_obj in error_objs:
    #Make error Plots
    if len(at_numbers) == 1:
        full_at_dir = os.path.join("Results_MS", at_class.scheme_name, obj_choice, "param_set_" + str(param_set))
    else:
        full_at_dir = os.path.join("Results_MS", "AT-" + "".join(map(str, at_numbers)), obj_choice)
    os.makedirs(full_at_dir, exist_ok=True)
    pdf_MAPD = PdfPages(os.path.join(full_at_dir , error_obj.upper() + ".pdf"))
    #For each molecule
    # if error_obj == "mae":
    save_name = os.path.join(full_at_dir, error_obj + "_props.png")
    # else:
    #     save_name = None
    pdf_MAPD.savefig(plot_err_each_prop(molec_names, df_err_dict, obj = error_obj, save_name=save_name), bbox_inches='tight')
    plt.close()
    pdf_MAPD.savefig(plot_err_avg_props(molec_names, df_err_dict, obj = error_obj), bbox_inches='tight')
    plt.close()
    # #Close figures 
    pdf_MAPD.close() 
