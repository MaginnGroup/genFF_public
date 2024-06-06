import sys
import os
import numpy as np
import pandas as pd
import math
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
                "R152": R152,
                "R143": R143,
                "R134": R134,
                "R116": R116}

def block_average(data):
    """
    Calculate block averages

    Implements the technique of Flyvbjerg and Peterson
    J. Chem. Phys. 91, 461 (1989). Also described in
    Appendix D of Frenkel and Smit. This function implements
    equation D.3.4 of Frenkel and Smit.

    The data should be provided as a numpy ndarray with
    shape=(npoints,). The function performs N blocking
    operations, where the block size is 2^N and the
    maximum number of blocking operations is determined
    from the number of points in ``data``

    Parameters:
    -----------
    data : numpy.ndarray, shape=(npoints,)
        numpy array with shape (npoints,) where npoints
        is the number of data point in the sample

    Returns:
    -------
    means, vars_est, vars_err

    means : np.ndarray, shape=(n_avg_ops,)
        mean values calculated from different numbers
        of blocking operations
    vars_est : np.ndarray, shape=(n_avg_ops,)
        estimates of the variances of the average from different
        numbers of blocking operations
    vars_err: np.ndarray, shape=(n_avg_ops,)
        estimates of the error in the variances from different
        numbers of blocking operations
    """

    try:
        data = np.asarray(data)
    except:
        raise TypeError("data should be provided as a numpy.ndarray")

    means = []
    vars_est = []
    vars_err = []

    n_samples = data.shape[0]

    max_blocking_ops = 0
    block_length = 1
    while block_length < 1.0 / 4.0 * n_samples:
        max_blocking_ops += 1
        block_length = 2 ** max_blocking_ops

    # Calc stats for mulitple-of-two block lengths
    for m in range(max_blocking_ops):
        block_length = 2 ** m  # Number of datapoints in each block
        n_blocks = int(
            n_samples / block_length
        )  # Number of blocks we can get with given block size
        # Calculate the 'new' dataset by block averaging
        block_data = [
            np.mean(
                data[i * block_length : (i + 1) * block_length],
                dtype=np.float64,
            )
            for i in range(n_blocks)
        ]
        block_data = np.asarray(block_data, dtype=np.float64)
        # Calculate the mean of this new dataset
        mean = np.mean(block_data, dtype=np.float64)
        # Calculate the variance of this new dataset
        var = np.var(block_data, dtype=np.float64)
        var_err = math.sqrt(2.0 * var ** 2.0 / (n_blocks - 1) ** 3.0)

        # Save data for blocking op
        means.append(mean)
        vars_est.append(var / (n_blocks - 1))
        vars_err.append(var_err)

    return np.asarray(means), np.asarray(vars_est), np.asarray(vars_err)

def prepare_df_vle(df_csv, molec_dict, csv_name = None):
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
        if property_name == "temperature":
            sim_str = ""
        else:
            sim_str = "sim_"
        if prop_units in df.columns:
            df.rename(columns={prop_units: sim_str + property_name}, inplace=True)
        elif property_name in df.columns:
            df.rename(columns={property_name: sim_str + property_name}, inplace=True)
        else:
            raise ValueError(f"df must contain either {property_name} or {prop_units}")
        
        return df
    
    # Convert Hvap to kJ/kg if in kJ/mol
    if "Hvap" in df_csv.columns:
        #Add Hvap in kJ/kg
        df_csv["Hvap kJ/kg"] = df_csv["Hvap"]*1000.0/df_csv["molecule"].apply(
            lambda molec: molec_dict[molec].molecular_weight)
        #And drop kJ/mol column
        df_csv.drop("Hvap", axis=1)
    
    # Rename properties to MD
    props = ["liq_density", "vap_density", "Pvap", "Hvap", "temperature"]
    units = ["kg/m3", "kg/m3", "bar", "kJ/kg", "K"]
    for prop, unit in zip(props, units):
        rename_col(df_csv, prop, unit)
        
    #sort by molecule and temperature -- added by Ning Wang
    df_csv.dropna(how = "any", inplace=True)
    df_csv.sort_values(by=["molecule", "temperature"], inplace=True)

    #Add Tc and Rhoc predictions
    Tc, rhoc = calc_critical(df_csv)
    df_csv["sim_Tc"] = Tc
    df_csv["sim_rhoc"] = rhoc

    if csv_name != None:
        df_csv.to_csv(csv_name)
           
    return df_csv

def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    Tc = []
    rhoc = []
    for group, values in df.groupby(['molecule']):    
        #Need to group by molecule and do this for each molecule
        temps = values["temperature"].values
        liq_density = values["sim_liq_density"].values
        vap_density = values["sim_vap_density"].values

        # Critical Point (Law of rectilinear diameters)
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
            temps,(liq_density + vap_density) / 2.0,)

        try:
            slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
                temps,(liq_density - vap_density)**(1/0.32),)
        except:
            slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
                temps,abs((liq_density - vap_density))**(1/0.32),)

        Tc_mol = np.abs(intercept2 / slope2)
        rhoc_mol = intercept1 + slope1 * Tc_mol

        # if len(temps) == 5:
        Tc += list([Tc_mol])*len(temps)
        rhoc += list([rhoc_mol])*len(temps)
        # else:
        #     Tc += [np.nan]*len(temps)
        #     rhoc += [np.nan]*len(temps)
        
    return Tc, rhoc

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
        the MSE and MAPD for liq_density, vap_density, pvap, hvap,
        critical temperature, critical density
    """
    new_data = []

    #sort by molecule and temperature -- added by Ning Wang
    df=df.sort_values(by=["molecule", "temperature"])
    molecules = df['molecule'].unique().tolist()
    for group, values in df.groupby(['molecule']):

        #The molecule is listed as the first value in the group
        molecule = molec_dict[values["molecule"].values[0]]
        if group[0] not in ["R134", "R152"]:
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
            sim_liq_density = values["sim_liq_density"]
            mse_liq_density = mean_squared_error(values["expt_liq_density"], sim_liq_density)
            mapd_liq_density = mean_absolute_percentage_error(
                values["expt_liq_density"], sim_liq_density) * 100.0

            # Vapor density
            sim_vap_density = values["sim_vap_density"]
            mse_vap_density = mean_squared_error(values["expt_vap_density"],sim_vap_density)
            mapd_vap_density = mean_absolute_percentage_error(
                values["expt_vap_density"], sim_vap_density) * 100.0
            
            # Vapor pressure
            sim_pvap = values["sim_Pvap"]
            mse_Pvap = mean_squared_error(values["expt_Pvap"], sim_pvap)
            mapd_Pvap = mean_absolute_percentage_error(
                values["expt_Pvap"], sim_pvap) * 100.0
        
            if group[0] not in ["R143"]:
                # Enthalpy of vaporization
                sim_hvap = values["sim_Hvap"]
                mse_Hvap = mean_squared_error(values["expt_Hvap"], sim_hvap)
                mapd_Hvap = mean_absolute_percentage_error(values["expt_Hvap"], sim_hvap) * 100.0

            else:
                mse_Hvap = np.nan
                mapd_Hvap = np.nan
        
            # Critical Point (Law of rectilinear diameters)
            expt_Tc_arr = np.array([molecule.expt_Tc])
            sim_Tc_arr = np.array([values["sim_Tc"].values[0]])
            expt_rhoc_arr = np.array([molecule.expt_rhoc])
            sim_rhoc_arr = np.array([values["sim_rhoc"].values[0]])
            try:
                mse_Tc = mean_squared_error(expt_Tc_arr, sim_Tc_arr)
                mapd_Tc = mean_absolute_percentage_error(expt_Tc_arr, sim_Tc_arr) * 100.0
            except ValueError as e:
                print(f"Error in calculating Tc for {group[0]}: {e}. Setting MSE and MAPD to NaN")
                mse_Tc = np.nan
                mapd_Tc = np.nan
            try:
                mse_rhoc = mean_squared_error(expt_rhoc_arr, sim_rhoc_arr)
                mapd_rhoc = mean_absolute_percentage_error(expt_rhoc_arr, sim_rhoc_arr) * 100.0
            except ValueError as e:
                print(f"Error in calculating rhoc for {group[0]}: {e}. Setting MSE and MAPD to NaN")
                mse_rhoc = np.nan
                mapd_rhoc = np.nan
            
        else:
            mse_liq_density = np.nan
            mapd_liq_density = np.nan
            mse_vap_density = np.nan
            mapd_vap_density = np.nan
            mse_Pvap = np.nan
            mapd_Pvap = np.nan
            mse_Hvap = np.nan
            mapd_Hvap = np.nan
            mse_Tc = np.nan
            mapd_Tc = np.nan
            mse_rhoc = np.nan
            mapd_rhoc = np.nan
        
        new_quantities = {
                "mse_liq_density": mse_liq_density,
                "mse_vap_density": mse_vap_density,
                "mse_Pvap": mse_Pvap,
                "mse_Hvap": mse_Hvap,
                "mse_Tc": mse_Tc,
                "mse_rhoc": mse_rhoc,
                "mapd_liq_density": mapd_liq_density,
                "mapd_vap_density": mapd_vap_density,
                "mapd_Pvap": mapd_Pvap,
                "mapd_Hvap": mapd_Hvap,
                "mapd_Tc": mapd_Tc,
                "mapd_rhoc": mapd_rhoc,
            }

        data_to_append = list(group) + list(new_quantities.values())
        # print(data_to_append)
        new_data.append(data_to_append)

    columns = list(["molecule"]) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)

    if csv_name != None:
        new_df.to_csv(csv_name)

    return new_df

def get_min_max(curr_min, curr_max, new_vals):
    if isinstance(new_vals, float):
        new_vals = [new_vals]
    min_new_val = min(new_vals)
    max_new_val = max(new_vals)
    if min_new_val < curr_min:
        curr_min = min_new_val
    if max_new_val > curr_max:
        curr_max = max_new_val
    return curr_min, curr_max

def plot_vle_envelopes(molec_dict, df_ff_list, save_name = None):
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))    
    
    df_labels = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    df_markers = ['o', 's', '^', '*', 'p', 'd']
    df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    if molec not in ["R152", "R134"]:
        min_temp = min(mol_data.expt_liq_density.keys())
        max_temp = mol_data.expt_Tc
        min_rho = min(mol_data.expt_vap_density.values())
        max_rho = max(mol_data.expt_liq_density.values())
    else:
        for df in df_ff_list:
            if df is not None:
                min_temp = min(df["temperature"].values)
                max_temp = max(df["temperature"].values)
                min_rho = min(df["sim_vap_density"].values)
                max_rho = max(df["sim_liq_density"].values)
                break

    for i in range(len(df_ff_list)):
        df_ff = df_ff_list[i]
        if df_ff is not None:
            #Set new max and mins
            min_rho, max_rho = get_min_max(min_rho, max_rho, df_ff["sim_liq_density"].values)
            min_rho, max_rho = get_min_max(min_rho, max_rho, df_ff["sim_vap_density"].values)
            min_temp, max_temp = get_min_max(min_temp, max_temp, df_ff["sim_Tc"].values)
            # #Plot opt_scheme_ms vle curve
            ax2.scatter(df_ff["sim_liq_density"], df_ff["temperature"], c=df_colors[i],s=70, 
                        marker = df_markers[i], alpha=0.7, zorder = df_z_order[i],)
            ax2.scatter(df_ff["sim_vap_density"], df_ff["temperature"],c=df_colors[i],s=70, 
                        marker = df_markers[i], alpha=0.7, zorder = df_z_order[i],)
            #Plot critical points
            ax2.scatter(df_ff["sim_rhoc"],df_ff["sim_Tc"], c=df_colors[i],s=70, 
                        marker = df_markers[i], alpha=0.7, zorder = df_z_order[i],
                        label = df_labels[i] )

    #Plot experimental data
    if molec not in ["R152", "R134"]:
        ax2.scatter(mol_data.expt_liq_density.values(),mol_data.expt_liq_density.keys(),
            color="black",marker="x",linewidths=2,s=80,label="Experiment", zorder = 7)
        ax2.scatter(mol_data.expt_vap_density.values(),mol_data.expt_vap_density.keys(),
            color="black",marker="x",linewidths=2,s=80, zorder = 7)
        ax2.scatter(mol_data.expt_rhoc, mol_data.expt_Tc, color="black", marker="x", linewidths=2, 
                    s=200, zorder = 7)

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

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    ax2.text(0.60,  0.82, molec, fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    return fig

    # if save_name is not None:
    #     path = os.path.join(save_name, "vle_plt.png")
    #     fig.savefig(path,dpi=300)

def plot_pvap_hvap(molec_dict, df_ff_list, save_name = None):
    molec = list(molec_dict.keys())[0]
    mol_data = molec_dict[molec]
    # Plot Pvap and Hvap
    
    df_labels = ["This Work", "GAFF", "Potoff et al.", "TraPPE", "Wang et al.", "Befort et al." ]
    df_colors = ['blue', 'gray', '#0989d9', 'red', 'green','purple']
    df_markers = ['o', 's', '^', '*', 'p', 'd']
    df_z_order = [6,3,2,1,5,4]

    #Initialize min and max values
    if molec not in ["R152", "R134"]:
        min_temp = max(np.array(list(mol_data.expt_Pvap.keys())))
        max_temp = mol_data.expt_Tc
        min_pvap = min(np.log(np.array(list(mol_data.expt_Pvap.values()))))
        max_pvap = max(np.log(np.array(list(mol_data.expt_Pvap.values()))))
    else:
        for df in df_ff_list:
            if df is not None:
                min_temp = min(df["temperature"].values)
                max_temp = max(df["temperature"].values)
                min_pvap = min(np.log(df["sim_Pvap"].values))
                max_pvap = max(np.log(df["sim_Pvap"].values))
                break

    if molec not in ["R152", "R134", "R143"]:
        min_hvap = min(mol_data.expt_Hvap.values())
        max_hvap = max(mol_data.expt_Hvap.values())
    else:
        for df in df_ff_list:
            if df is not None:
                min_hvap = min(df["sim_Hvap"].values)
                max_hvap = max(df["sim_Hvap"].values)
                break

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))

    #Loop over dfs of given ff results
    for i in range(len(df_ff_list)):
        df_ff = df_ff_list[i]
        if df_ff is not None:
            #Set new max and mins
            min_temp, max_temp = get_min_max(min_temp, max_temp, df_ff["temperature"].values)
            min_pvap, max_pvap = get_min_max(min_pvap, max_pvap, np.log(df_ff["sim_Pvap"]).values)
            min_hvap, max_hvap = get_min_max(min_hvap, max_hvap, df_ff["sim_Hvap"].values)
            #Plot 1/T vs log(Pvap) 
            axs[0].scatter(1/df_ff["temperature"], np.log(df_ff["sim_Pvap"]), c=df_colors[i], 
                           s=70,alpha=0.7, label = df_labels[i], marker = df_markers[i],
                           zorder = df_z_order[i])
            #Plot T vs Hvap
            axs[1].scatter(df_ff["temperature"],df_ff["sim_Hvap"], c=df_colors[i], 
                           s=70,alpha=0.7, marker = df_markers[i],
                           zorder = df_z_order[i])
        
    #Plot experimental pvap
    if molec not in ["R152", "R134"]:
        axs[0].scatter(1/np.array(list(mol_data.expt_Pvap.keys())),
                       np.log(np.array(list(mol_data.expt_Pvap.values()))),
            color="black",marker="x",label="Experiment",s=80,zorder = 7)
    #Plot experimental Hvap
    if molec not in ["R152", "R134", "R143"]:
        axs[1].scatter(mol_data.expt_Hvap.keys(),mol_data.expt_Hvap.values(),
            color="black",marker="x",label="Experiment",s=80, zorder = 7)

    #Set axes details
    axs[0].set_xlim((1/max_temp)*0.95,(1/min_temp)*1.05)
    # axs[0].xaxis.set_major_locator(MultipleLocator(40))
    # axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0].set_ylim(min_pvap*0.8,max_pvap*1.05)
    # axs[0].yaxis.set_major_locator(MultipleLocator(10))
    # axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[0].tick_params("both", which="major", length=4)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel("1/T " + r"$\mathregular{K^{-1}}$", fontsize=16, labelpad=8)
    axs[0].set_ylabel(r"$\mathregular{log(P_{vap})}$ (bar)", fontsize=16, labelpad=8)

    axs[1].set_xlim(min_temp*0.95,max_temp*1.05)
    # axs[1].xaxis.set_major_locator(MultipleLocator(40))
    # axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1].set_ylim(min_hvap*0.95, max_hvap*1.05)
    # axs[1].yaxis.set_major_locator(MultipleLocator(100))
    # axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[1].tick_params("both", which="major", length=4)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)

    if molec not in ["R14", "R50", "R170", "R116"]:
        #Substitute mole string R w/ HFC
        molec = molec.replace("R","HFC")
    axs[0].text(0.08, 0.3, molec, fontsize=20, transform=axs[0].transAxes)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.35, 1.05), ncol=3, fontsize=16, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)

    return fig
    # if save_name is not None:
    #     path = os.path.join(save_name, "h_p_vap_plt.png")
    #     fig.savefig(path,dpi=300)