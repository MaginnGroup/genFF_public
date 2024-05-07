import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import linregress
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real
from fffit.fffit.plot import plot_model_performance, plot_model_vs_test, plot_slices_temperature, plot_slices_params, plot_model_vs_exp, plot_obj_contour
from .molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116


def prepare_df_vle_errors(df, molecule):
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
    #sort by temperature -- added by Ning Wang
    df=df.sort_values(by=["temperature"])

    for group, values in df.groupby(list(molecule.param_names)):
        # Temperatures
        temps = values_scaled_to_real(
            values["temperature"], molecule.temperature_bounds
        )
        # Liquid density
        sim_liq_density = values_scaled_to_real(
            values["sim_liq_density"], molecule.liq_density_bounds
        )
        expt_liq_density = values_scaled_to_real(
            values["expt_liq_density"], molecule.liq_density_bounds
        )
        mse_liq_density = np.mean((sim_liq_density - expt_liq_density) ** 2)
        mape_liq_density = (
            np.mean(
                np.abs((sim_liq_density - expt_liq_density) / expt_liq_density)
            )
            * 100.0
        )
        properties = {
            f"sim_liq_density_{float(temp):.0f}K": float(liq_density)
            for temp, liq_density in zip(temps, sim_liq_density)
        }
        # Vapor density
        sim_vap_density = values_scaled_to_real(
            values["sim_vap_density"], molecule.vap_density_bounds
        )
        expt_vap_density = values_scaled_to_real(
            values["expt_vap_density"], molecule.vap_density_bounds
        )
        mse_vap_density = np.mean((sim_vap_density - expt_vap_density) ** 2)
        mape_vap_density = (
            np.mean(
                np.abs((sim_vap_density - expt_vap_density) / expt_vap_density)
            )
            * 100.0
        )
        properties.update(
            {
                f"sim_vap_density_{float(temp):.0f}K": float(vap_density)
                for temp, vap_density in zip(temps, sim_vap_density)
            }
        )

        # Vapor pressure
        sim_Pvap = values_scaled_to_real(
            values["sim_Pvap"], molecule.Pvap_bounds
        )
        expt_Pvap = values_scaled_to_real(
            values["expt_Pvap"], molecule.Pvap_bounds
        )
        mse_Pvap = np.mean((sim_Pvap - expt_Pvap) ** 2)
        mape_Pvap = np.mean(np.abs((sim_Pvap - expt_Pvap) / expt_Pvap)) * 100.0
        properties.update(
            {
                f"sim_Pvap_{float(temp):.0f}K": float(Pvap)
                for temp, Pvap in zip(temps, sim_Pvap)
            }
        )
        # Enthalpy of vaporization
        sim_Hvap = values_scaled_to_real(
            values["sim_Hvap"], molecule.Hvap_bounds
        )
        expt_Hvap = values_scaled_to_real(
            values["expt_Hvap"], molecule.Hvap_bounds
        )
        mse_Hvap = np.mean((sim_Hvap - expt_Hvap) ** 2)
        mape_Hvap = np.mean(np.abs((sim_Hvap - expt_Hvap) / expt_Hvap)) * 100.0
        properties.update(
            {
                f"sim_Hvap_{float(temp):.0f}K": float(Hvap)
                for temp, Hvap in zip(temps, sim_Hvap)
            }
        )

        # Critical Point (Law of rectilinear diameters)
        slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
            temps.flatten(),
            ((sim_liq_density + sim_vap_density) / 2.0).flatten(),
        )

        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
            temps.flatten(),
            ((sim_liq_density - sim_vap_density) ** (1 / 0.32)).flatten(),
        )

        Tc = np.abs(intercept2 / slope2)
        mse_Tc = (Tc - molecule.expt_Tc) ** 2
        mape_Tc = np.abs((Tc - molecule.expt_Tc) / molecule.expt_Tc) * 100.0
        properties.update({"sim_Tc": Tc})

        rhoc = intercept1 + slope1 * Tc
        mse_rhoc = (rhoc - molecule.expt_rhoc) ** 2
        mape_rhoc = (
            np.abs((rhoc - molecule.expt_rhoc) / molecule.expt_rhoc) * 100.0
        )
        properties.update({"sim_rhoc": rhoc})
        new_quantities = {
            **properties,
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
    columns = list(molecule.param_names) + list(new_quantities.keys())
    new_df = pd.DataFrame(new_data, columns=columns)
    return new_df

def plot_vle_envelopes(df_opt_ms, df_lit_ms, df_gaff_ms, molec_data_dict):
    # Plot VLE envelopes
    fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
    temps_r14 = R14.expt_liq_density.keys()

    #clrs = seaborn.color_palette('bright', n_colors=len(df_r14))
    #np.random.seed(11)
    #np.random.shuffle(clrs)

    for temp in temps_r14:
        print(df_r14.loc[f"sim_liq_density_{float(temp):.0f}K"])
        ax2.scatter(
            df_r14.loc[f"sim_liq_density_{float(temp):.0f}K"],
            temp,
            c='blue',
            s=160,
            alpha=0.7,
        )
        ax2.scatter(
            df_r14.loc[f"sim_vap_density_{float(temp):.0f}K"],
            temp,
            c='blue',
            s=160,
            alpha=0.7,
        )
    ax2.scatter(
        df_r14.loc["sim_rhoc"],
        df_r14.loc["sim_Tc"],
        c='blue',
        s=160,
        alpha=0.7,
    )

    tc, rhoc = calc_critical(df_r14_gaff)
    print("GAFF: ", tc, " K ",rhoc)
    ax2.scatter(
        df_r14_gaff["liq_density"],
        df_r14_gaff["temperature"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
        label="GAFF",
    )
    ax2.scatter(
        df_r14_gaff["vap_density"],
        df_r14_gaff["temperature"],
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )
    ax2.scatter(
        rhoc,
        tc,
        c='gray',
        s=120,
        alpha=0.7,
        marker='s',
    )

    tc, rhoc = calc_critical(df_r14_lit)
    print(tc,rhoc)
    ax2.scatter(
        df_r14_lit["liq_density"],
        df_r14_lit["temperature"],
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
        label="Potoff et al.",
    )
    ax2.scatter(
        df_r14_lit["vap_density"],
        df_r14_lit["temperature"],
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
    )
    ax2.scatter(
        rhoc,
        tc,
        c='#0989d9',
        s=160,
        alpha=0.7,
        marker='^',
    )

    tc, rhoc = calc_critical(df_r14_trappe)
    ax2.scatter(
        df_r14_trappe["liq_density"],
        df_r14_trappe["temperature"],
        c='red',
        s=160,
        alpha=0.7,
        marker='*',
        label="TraPPE",
    )
    ax2.scatter(
        df_r14_trappe["vap_density"],
        df_r14_trappe["temperature"],
        c='red',
        s=160,
        alpha=0.7,
        marker='*',
    )
    ax2.scatter(
        rhoc,
        tc,
        c='red',
        s=160,
        alpha=0.7,
        marker='*',
    )

    ax2.scatter(
        R14.expt_liq_density.values(),
        R14.expt_liq_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
        label="Experiment",
    )
    ax2.scatter(
        R14.expt_vap_density.values(),
        R14.expt_vap_density.keys(),
        color="black",
        marker="x",
        linewidths=2,
        s=200,
    )
    ax2.scatter(R14.expt_rhoc, R14.expt_Tc, color="black", marker="x", linewidths=2, s=200)

    ax2.set_xlim(-50, 1850)
    ax2.xaxis.set_major_locator(MultipleLocator(500))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.set_ylim(120,280)
    ax2.yaxis.set_major_locator(MultipleLocator(40))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax2.tick_params("both", direction="in", which="both", length=4, labelsize=26, pad=10)
    ax2.tick_params("both", which="major", length=8)
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")

    ax2.set_ylabel("T (K)", fontsize=32, labelpad=10)
    ax2.set_xlabel(r"$\mathregular{\rho}$ (kg/m$\mathregular{^3}$)", fontsize=32, labelpad=15)
    for axis in ['top','bottom','left','right']:
    #    ax1.spines[axis].set_linewidth(2.0)
        ax2.spines[axis].set_linewidth(2.0)

    ax2.legend(loc="lower left", bbox_to_anchor=(-0.16, 1.03), ncol=2, fontsize=22, handletextpad=0.1, markerscale=0.9, edgecolor="dimgrey")
    #ax1.text(0.08, 0.82, "a", fontsize=40, transform=ax1.transAxes)
    #ax1.text(0.5, 0.82, "HFC-32", fontsize=34, transform=ax1.transAxes)
    #ax2.text(0.08, 0.82, "a", fontsize=40, transform=ax2.transAxes)
    ax2.text(0.7,  0.82, "R-14", fontsize=30, transform=ax2.transAxes)
    fig.subplots_adjust(bottom=0.2, top=0.75, left=0.15, right=0.95, wspace=0.55)

    fig.savefig("pdfs/fig3_r14-results-vle.png",dpi=300)


def calc_critical(df):
    """Compute the critical temperature and density

    Accepts a dataframe with "T_K", "rholiq_kgm3" and "rhovap_kgm3"
    Returns the critical temperature (K) and density (kg/m3)

    Computes the critical properties with the law of rectilinear diameters
    """
    temps = df["temperature"].values
    liq_density = df["liq_density"].values
    vap_density = df["vap_density"].values
    # Critical Point (Law of rectilinear diameters)
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(
        temps,
        (liq_density + vap_density) / 2.0,
    )

    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        temps,
        (liq_density - vap_density) ** (1 / 0.32),
    )

    Tc = np.abs(intercept2 / slope2)
    rhoc = intercept1 + slope1 * Tc

    return Tc, rhoc

def plot_pvap_hvap():
        #fig, ax2 = plt.subplots(1, 1, figsize=(6,6))
    temps_r14 = R14.expt_liq_density.keys()

    # Plot Pvap / Hvap
    fig, axs = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
    #fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
    clrs = seaborn.color_palette('bright', n_colors=len(df_r14))
    np.random.seed(11)
    np.random.shuffle(clrs)

    for temp in temps_r14:
        axs[0].scatter(
            temp,
            df_r14.loc[f"sim_Pvap_{float(temp):.0f}K"],
            #np.tile(temp, len(df_r14)),
            #df_r14.filter(regex=(f"Pvap_{float(temp):.0f}K")),
            c='blue',
            s=70,
            alpha=0.7,
        )
    axs[0].scatter(
        df_r14_gaff["temperature"],
        df_r14_gaff["Pvap"],
        c='gray',
        s=70,
        alpha=0.7,
        label="GAFF",
        marker='s',
    )
    axs[0].scatter(
        df_r14_lit["temperature"],
        df_r14_lit["Pvap"],
        c='#0989d9',
        s=70,
        alpha=0.7,
        label="Potoff et al.",
        marker='^',
    )
    axs[0].scatter(
        R14.expt_Pvap.keys(),
        R14.expt_Pvap.values(),
        color="black",
        marker="x",
        label="Experiment",
        s=80,
    )

    axs[0].set_xlim(120,230)
    axs[0].xaxis.set_major_locator(MultipleLocator(40))
    axs[0].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[0].set_ylim(-2,35)
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[0].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[0].tick_params("both", which="major", length=4)
    axs[0].xaxis.set_ticks_position("both")
    axs[0].yaxis.set_ticks_position("both")

    axs[0].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[0].set_ylabel(r"$\mathregular{P_{vap}}$ (bar)", fontsize=16, labelpad=8)
    #for axis in ['top','bottom','left','right']:
    #    axs[0,0].spines[axis].set_linewidth(2.0)
    #    axs[0,1].spines[axis].set_linewidth(2.0)
    #    axs[1,0].spines[axis].set_linewidth(2.0)
    #    axs[1,1].spines[axis].set_linewidth(2.0)

    # Plot Enthalpy of Vaporization
    for temp in temps_r14:
        axs[1].scatter(
            temp,
            df_r14.loc[f"sim_Hvap_{float(temp):.0f}K"],
            #np.tile(temp, len(df_r14)),
            #df_r14.filter(regex=(f"Hvap_{float(temp):.0f}K")),
            c='blue',
            s=70,
            alpha=0.7,
        )
    axs[1].scatter(
        df_r14_gaff["temperature"],
        df_r14_gaff["Hvap"] / R14.molecular_weight * 1000.0,
        c='gray',
        s=70,
        alpha=0.7,
        marker='s',
    )
    print(df_r14_gaff["temperature"],df_r14_gaff["Hvap"] / R14.molecular_weight * 1000.0)
    axs[1].scatter(
        df_r14_lit["temperature"],
        df_r14_lit["Hvap"] ,#kj/kg
        c='#0989d9',
        s=70,
        alpha=0.7,
        marker='^',
    )
    axs[1].scatter(
        R14.expt_Hvap.keys(),
        R14.expt_Hvap.values(),
        color="black",
        marker="x",
        s=80,
    )

    axs[1].set_xlim(120,230)
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].xaxis.set_minor_locator(AutoMinorLocator(4))

    axs[1].set_ylim(20,210)
    axs[1].yaxis.set_major_locator(MultipleLocator(100))
    axs[1].yaxis.set_minor_locator(AutoMinorLocator(5))

    axs[1].tick_params("both", direction="in", which="both", length=2, labelsize=16, pad=10)
    axs[1].tick_params("both", which="major", length=4)
    axs[1].xaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")

    axs[1].set_xlabel("T (K)", fontsize=16, labelpad=8)
    axs[1].set_ylabel(r"$\mathregular{\Delta H_{vap}}$ (kJ/kg)", fontsize=16, labelpad=8)


    axs[0].text(0.08, 0.8, "R-14", fontsize=20, transform=axs[0].transAxes)
    axs[0].legend(loc="lower left", bbox_to_anchor=(0.35, 1.05), ncol=3, fontsize=16, handletextpad=0.1, markerscale=0.8, edgecolor="dimgrey")

    fig.subplots_adjust(bottom=0.15, top=0.85, left=0.15, right=0.85, wspace=0.55, hspace=0.5)
    fig.savefig("pdfs/fig3-p-h-png",dpi=300)