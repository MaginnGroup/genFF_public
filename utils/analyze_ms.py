import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance, plot_model_vs_test, plot_slices_temperature, plot_slices_params, plot_model_vs_exp, plot_obj_contour
from .molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116

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

def prepare_df_vle(df_csv, molec_dict, convert_Hvap = False, csv_name = None):
    """Prepare a pandas dataframe for fitting a GP model to density data

    Performs the following actions:
       - Renames "liq_density" to "sim_liq_density" (units kg/m3 assumed)
       - Renames "vap_density" to "sim_vap_density" (units kg/m3 assumed)
       - Renames "Pvap" to "sim_Pvap" (units bar assumed)
       - Removes "liq_enthalpy" and "vap_enthalpy" and adds "sim_Hvap" (units kJ/kg returned)
            - Units kJ/mol assumed for Hvap and kJ/kg for Hvap kJ/kg

    Parameters
    ----------
    df_csv : pd.DataFrame
        The dataframe as loaded from a CSV file with the signac results
    molec_dict : {"Rxx": RxxConstants, ...}
        A dictionary mapping molecule names to classes
    n_molecules : int
        The number of molecules in the simulation

    Returns
    -------
    df_all : pd.DataFrame
        The dataframe with scaled parameters and MD/expt. properties
    """

    def rename_col(df, property_name, units):
        prop_units = property_name + " " + units
        if property_name in df.columns:
            df.rename(columns={property_name: "sim_" + property_name}, inplace=True)
        elif prop_units in df.columns:
            df.rename(columns={prop_units: "sim_" + property_name}, inplace=True)
        else:
            raise ValueError(f"df must contain either {property_name} or {prop_units}")
        return df
    
    # Convert Hvap to kJ/kg if in kJ/mol
    if "Hvap kJ/kg" in df_csv.columns:
        #Add Hvap in kJ/mol
        df_csv["Hvap"] = df_csv["Hvap"]/df_csv["molecule"].apply(
            lambda molec: molec_dict[molec].molecular_weight*1000.0)
        #And drop kJ/kg column
        df_csv.drop("Hvap kJ/kg", axis=1)
    
    # Rename properties to MD
    props = ["liq_density", "vap_density", "Pvap", "Hvap"]
    units = ["kg/m3", "kg/m3", "bar", "kJ/kg"]
    for prop, unit in zip(props, units):
        rename_col(df_csv, prop, unit)
        
    #sort by molecule and temperature -- added by Ning Wang
    df_csv.sort_values(by=["molecule", "temperature"], inplace=True)

    #Add Tc and Rhoc predictions
    Tc, rhoc = calc_critical(df_csv)
    df_csv["sim_Tc"] = Tc
    df_csv["sim_rhoc"] = rhoc

    if csv_name != None:
        df_csv.to_csv(csv_name)
           
    return df_csv

def prepare_df_vle_errors(df, molec_dict, csv_name = None):
    """Create a dataframe with mean square error (mse) and mean absolute
    percent error (mape) for each unique parameter set. The critical
    temperature and density are also evaluated.

    Parameters
    ----------
    df : pandas.Dataframe
        per simulation results
    molecule : R143a
        molecule class with bounds/experimental data

    Returns
    -------
    df_new : pandas.Dataframe
        dataframe with one row per parameter set and including
        the MSE and MAPE for liq_density, vap_density, pvap, hvap,
        critical temperature, critical density
    """
    new_data = []

    #sort by molecule and temperature -- added by Ning Wang
    df=df.sort_values(by=["molecule", "temperature"])
    molecules = df['molecule'].unique().tolist()
    for group, values in df.groupby(molecules):
        #The molecule is listed as the first value in the group
        molecule = molec_dict[values["molecule"].values[0]]

        # Temperatures
        temps = values["temperature"].values

        #Add experimental data (if not R134, 143 or R152)
        values["expt_liq_density"] = values["temperature"].apply(
            lambda temp: molecule.expt_liq_density[int(temp)])
        values["expt_vap_density"] = values["temperature"].apply(
            lambda temp: molecule.expt_vap_density[int(temp)] )
        values["expt_Pvap"] = values["temperature"].apply(
            lambda temp: molecule.expt_Pvap[int(temp)])
        values["expt_Hvap"] = values["temperature"].apply(
            lambda temp: molecule.expt_Hvap[int(temp)])
        values["expt_Tc"] = molecule.expt_Tc
        values["expt_rhoc"] = molecule.expt_rhoc
    
        # Liquid density
        mse_liq_density = mean_squared_error(values["expt_liq_density"], values["sim_liq_density"])
        mape_liq_density = mean_absolute_percentage_error(
            values["expt_liq_density"], values["sim_liq_density"]) * 100.0

        # Vapor density
        mse_vap_density = mean_squared_error(values["expt_vap_density"], values["sim_vap_density"])
        mape_vap_density = mean_absolute_percentage_error(
            values["expt_liq_density"], values["sim_liq_density"]) * 100.0
        
        # Vapor pressure
        mse_Pvap = mean_squared_error(values["expt_Pvap"], values["sim_Pvap"])
        mape_Pvap = mean_absolute_percentage_error(
            values["expt_Pvap"], values["sim_Pvap"]) * 100.0
        
        # Enthalpy of vaporization
        mse_Hvap = mean_squared_error(values["expt_Hvap"], values["sim_Hvap"])
        mape_Hvap = mean_absolute_percentage_error(values["expt_Hvap"], values["sim_Hvap"]) * 100.0
        
        # Critical Point (Law of rectilinear diameters)
        mse_Tc = mean_squared_error(molecule.expt_Tc, values["sim_Tc"])
        mape_Tc = mean_absolute_percentage_error(molecule.expt_Tc, values["sim_Tc"]) * 100.0
        mse_rhoc = mean_squared_error(molecule.expt_rhoc, values["sim_rhoc"])
        mape_rhoc = mean_absolute_percentage_error(molecule.expt_rhoc, values["sim_rhoc"]) * 100.0
        new_quantities = {
            "mse_liq_density": mse_liq_density,
            "mse_vap_density": mse_vap_density,
            "mse_Pvap": mse_Pvap,
            "mse_Hvap": mse_Hvap,
            "mse_Tc": mse_Tc,
            "mse_rhoc": mse_rhoc,
            "mape_liq_density": mape_liq_density,
            "mape_vap_density": mape_vap_density,
            "mape_Pvap": mape_Pvap,
            "mape_Hvap": mape_Hvap,
            "mape_Tc": mape_Tc,
            "mape_rhoc": mape_rhoc,
        }
        
        new_data.append(list(group) + list(new_quantities.values()))
    columns = list(molecules) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    if csv_name != None:
        new_df.to_csv(csv_name)

    return new_df

def get_min_max(curr_min, curr_max, new_vals):
    for val in new_vals:
        if min(val) < curr_min:
            curr_min = val
        if max(val) > curr_max:
            curr_max = val
    return curr_min, curr_max

def plot_vle_envelopes(molec_dict, df_opt, df_lit = None, df_nw = None, df_trappe = None, df_gaff = None, save_name = None):
    molec = str(molec_dict.keys())
    mol_data = molec_dict[molec]
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
    temps = mol_data.expt_liq_density

    #Initialize min and max values
    min_temp = min(temps)
    max_temp = mol_data.expt_Tc
    min_rho = min(mol_data.expt_vap_density.values())
    max_rho = max(mol_data.expt_liq_density.values())

    #Plot opt_scheme_ms vle curve
    min_rho, max_rho = get_min_max(min_rho, max_rho, df_opt["sim_liq_density"])
    min_rho, max_rho = get_min_max(min_rho, max_rho, df_opt["sim_vap_density"])
    ax2.scatter(df_opt["sim_liq_density"], df_opt["temperature"],
        c='blue', s=160, alpha=0.7,)
    ax2.scatter(df_opt["sim_vap_density"], df_opt["temperature"],
        c='blue',s=160,alpha=0.7,)
    #Plot critical points
    ax2.scatter(df_opt["sim_rhoc"],df_opt["sim_Tc"],
        c='blue',s=160,alpha=0.7, label = "This Work")

    #Plot GAFF VLE Data if it exists
    if df_gaff is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_gaff["sim_Tc"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_gaff["sim_liq_density"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_gaff["sim_vap_density"])
        ax2.scatter(df_gaff["sim_liq_density"], df_gaff["temperature"],
            c='gray',s=120,alpha=0.7,marker='s',label="GAFF",)
        ax2.scatter(df_gaff["sim_vap_density"],df_gaff["temperature"],
            c='gray',s=120,alpha=0.7, marker='s',)
        ax2.scatter(df_gaff["sim_rhoc"],df_gaff["sim_Tc"],
            c='gray',s=120,alpha=0.7,marker='s',)

    #Plot NW Data if it exists
    if df_nw is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_nw["sim_Tc"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_nw["sim_liq_density"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_nw["sim_vap_density"])
        ax2.scatter(df_nw["sim_liq_density"],df_nw["temperature"],
            c='green',s=160,alpha=0.7,marker='o',label="Wang et al.",)
        ax2.scatter(df_nw["sim_vap_density"], df_nw["temperature"],
            c='green',s=160,alpha=0.7,marker='o',)
        ax2.scatter(df_nw["sim_rhoc"],df_nw["sim_Tc"],
            c='green',s=160,alpha=0.7,marker='o',)
        
    #Plot Potoff Data if it exists
    if df_lit is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_lit["sim_Tc"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_lit["sim_liq_density"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_lit["sim_vap_density"])
        ax2.scatter(df_lit["sim_liq_density"],df_lit["temperature"],
            c='#0989d9',s=160,alpha=0.7,marker='^',label="Potoff et al.",)
        ax2.scatter(df_lit["sim_vap_density"], df_lit["temperature"],
            c='#0989d9',s=160,alpha=0.7,marker='^',)
        ax2.scatter(df_lit["sim_rhoc"],df_lit["sim_Tc"],
            c='#0989d9',s=160,alpha=0.7,marker='^',)

    #Plot TraPPE data if it exists
    if df_trappe is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_trappe["sim_Tc"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_trappe["sim_liq_density"])
        min_rho, max_rho = get_min_max(min_rho, max_rho, df_trappe["sim_vap_density"])
        ax2.scatter(df_trappe["sim_liq_density"],df_trappe["temperature"],
            c='red',s=160,alpha=0.7,marker='*',label="TraPPE",)
        ax2.scatter(df_trappe["sim_vap_density"],df_trappe["temperature"],
            c='red',s=160,alpha=0.7,marker='*',)
        ax2.scatter(df_trappe["sim_rhoc"],df_trappe["sim_Tc"],
            c='red',s=160, alpha=0.7,marker='*',)

    #Plot experimental data
    ax2.scatter(mol_data.expt_liq_density.values(),mol_data.expt_liq_density.keys(),
        color="black",marker="x",linewidths=2,s=200,label="Experiment",)
    ax2.scatter(mol_data.expt_vap_density.values(),mol_data.expt_vap_density.keys(),
        color="black",marker="x",linewidths=2,s=200,)
    ax2.scatter(mol_data.expt_rhoc, mol_data.expt_Tc, color="black", marker="x", linewidths=2, s=200)

    #Set Axes
    ax2.set_xlim(min_rho*0.95,max_rho*1.05)
    ax2.xaxis.set_major_locator(MultipleLocator(500))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.set_ylim(min_temp*0.95, max_temp*1.05)
    ax2.yaxis.set_major_locator(MultipleLocator(40))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(2.0)

    ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax2.text(0.7,  0.82, molec, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    return fig

    # if save_name is not None:
    #     path = os.path.join(save_name, "vle_plt.png")
    #     fig.savefig(path,dpi=300)


def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    temps = df["temperature"].values
    liq_density = df["sim_liq_density"].values
    vap_density = df["sim_vap_density"].values
    # Critical Point (Law of rectilinear diameters)
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        temps,(liq_density + vap_density) / 2.0,)

    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        temps,(liq_density - vap_density) ** (1 / 0.32),)

    Tc = np.abs(intercept2 / slope2)
    rhoc = intercept1 + slope1 * Tc

    return Tc, rhoc

def plot_pvap_hvap(molec_dict, df_opt, df_lit = None, df_nw = None, df_trappe = None, df_gaff = None, save_name = None):
    molec = str(molec_dict.keys())
    mol_data = molec_dict[molec]
    # Plot Pvap and Hvap
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
    temps = mol_data.expt_Hvap.keys()

    #Initialize min and max values
    min_temp = min(temps)
    max_temp = mol_data.expt_Tc
    min_pvap = min(mol_data.expt_Pvap.values())
    max_pvap = max(mol_data.expt_Pvap.values())
    min_hvap = min(mol_data.expt_Hvap.values())
    max_hvap = max(mol_data.expt_Hvap.values())

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
    clrs = seaborn.color_palette('bright', n_colors=len(df_opt))
    np.random.seed(11)
    np.random.shuffle(clrs)

    #Plot opt pvap
    min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, df_opt["sim_Pvap"])
    axs[0].scatter(df_opt["temperature"], df_opt["sim_Pvap"], c='blue',s=70,alpha=0.7, label = "This Work")
    
    #Plot GAFF pvap if it exists
    if df_gaff is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_gaff["temperature"])
        min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, df_gaff["sim_Pvap"])
        axs[0].scatter(df_gaff["temperature"],df_gaff["sim_Pvap"],
            c='gray',s=70,alpha=0.7,marker='s',label="GAFF",)

    #Plot potoff pvap if it exists
    if df_lit is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_lit["temperature"])
        min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, df_lit["sim_Pvap"])
        axs[0].scatter(df_lit["temperature"],df_lit["sim_Pvap"],
            c='#0989d9',s=70,alpha=0.7,marker='^',label="Potoff et al.",)

    #Plot Nw pvap if it exists
    if df_nw is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_nw["temperature"])
        min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, df_nw["sim_Pvap"])
        axs[0].scatter(df_nw["temperature"],df_nw["sim_Pvap"],
            c='green',s=70,alpha=0.7,marker='^',label="Wang et al.",)
        
    #Plot experimental pvap
    axs[0].scatter(mol_data.expt_Pvap.keys(),mol_data.expt_Pvap.values(),
        color="black",marker="x",label="Experiment",s=80,)


    axs[0].set_xlim(min_temp*0.95,max_temp*1.05)
    axs[0].xaxis.set_major_locator(MultipleLocator(40))
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0].set_ylim(min_pvap*0.95,max_pvap*1.05)
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[0].tick_params("both", which="major", length=4)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[0].set_ylabel(r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, labelpad=8)

    # Plot Enthalpy of Vaporization
    min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, df_opt["sim_Hvap"], label = "This Work")
    axs[1].scatter(df_opt["temperature"],df_opt["sim_Hvap"], c='blue',s=70,alpha=0.7,)
        
    #Plot GAFF Hvap if it exists
    if df_gaff is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_gaff["temperature"])
        min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, df_gaff["sim_Hvap"])
        axs[1].scatter(df_gaff["temperature"],df_gaff["sim_Hvap"],
            c='gray',s=70,alpha=0.7,marker='s',label="GAFF",)
        print(df_gaff["temperature"],df_gaff["sim_Hvap"])

    #Plot potoff Hvap if it exists
    if df_lit is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_lit["temperature"])
        min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, df_lit["sim_Hvap"])
        axs[1].scatter(df_lit["temperature"],df_lit["sim_Hvap"],
            c='#0989d9',s=70,alpha=0.7,marker='^',label="Potoff et al.",)

    #Plot Wang Hvap if it exists
    if df_nw is not None:
        min_temp, max_temp = get_min_max(min_temp, max_temp, df_nw["temperature"])
        min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, df_nw["sim_Hvap"])
        axs[1].scatter(df_nw["temperature"],df_nw["sim_Hvap"],
            c='green',s=70,alpha=0.7,marker='^',label="Wang et al.",)
        
    #Plot experimental Hvap
    axs[1].scatter(mol_data.expt_Hvap.keys(),mol_data.expt_Hvap.values(),
        color="black",marker="x",label="Experiment",s=80,)

    axs[1].set_xlim(min_temp*0.95,max_temp*1.05)
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1].set_ylim(min_hvap*0.95, max_hvap*1.05)
    axs[1].yaxis.set_major_locator(MultipleLocator(100))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[1].tick_params("both", which="major", length=4)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)

    axs[0].text(0.08, 0.8, molec, fontsize=20, transform=axs[0].transAxes)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.35, 1.05), ncol=3, fontsize=16, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)

    return fig
    # if save_name is not None:
    #     path = os.path.join(save_name, "h_p_vap_plt.png")
    #     fig.savefig(path,dpi=300)