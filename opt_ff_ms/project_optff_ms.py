from flow import FlowProject, directives
import templates.ndcrc
import warnings
from pathlib import Path
import os
import sys
import unyt as u
import copy
from pymser import pymser
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append("..")
from utils.molec_class_files import (
    r14,
    r32,
    r50,
    r125,
    r134a,
    r143a,
    r170,
    r41,
    r23,
    r161,
    r152a,
    r152,
    r134,
    r143,
    r116,
)

sys.path.remove("..")


class Project(FlowProject):
    def __init__(self):
        # Set Project Path to be that of the current working directory
        current_path = Path(os.getcwd()).absolute()
        super().__init__(path=current_path)


@Project.post.isfile("ff.xml")
@Project.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""

    # Generate content based on job sp molecule name
    molec_xml_function = _get_xml_from_molecule(job.sp.mol_name)
    content = molec_xml_function(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)


def calc_box_helper(job):
    "Calculate the initial box length of the boxes"

    import unyt as u

    # Get reference data from constants file
    # Load class properies for each training and testing molecule
    class_dict = _get_class_from_molecule(job.sp.mol_name)
    class_data = class_dict[job.sp.mol_name]
    # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in ProjectGAFF_gaff.py as needed)
    ref = {}

    # If the gemc simulation failed previously, use the critical values
    if "use_crit" in job.doc and job.doc.use_crit == True:
        rho_liq = class_data.expt_rhoc * u.kilogram / (u.meter) ** 3
        rho_vap = class_data.expt_rhoc * u.kilogram / (u.meter) ** 3
    else:
        # Initialize rho_liq and rho_vap as the experimental values
        rho_liq = job.sp.expt_liq_density * u.kilogram / (u.meter) ** 3
        rho_vap = (
            class_data.expt_vap_density[int(job.sp.T)] * u.kilogram / (u.meter) ** 3
        )

    # Create a tuple containing the values from each dictionary
    ref[int(job.sp.T)] = (rho_liq, rho_vap, job.sp.P)

    vap_density = ref[job.sp.T][1]
    mol_density = vap_density / (job.sp.mol_weight * u.amu)
    vol_vap = job.sp.N_vap / mol_density
    vapboxl = vol_vap ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    vapboxl = round(float(vapboxl.in_units(u.nm).to_value()), 2)

    # If molecule is R41, reduce the vapor box length by 20% to keep it inside the phase envelope
    if job.sp.mol_name == "R41":
        vapboxl = vapboxl * 0.80

    # Save to job document file
    job.doc.vapboxl = vapboxl  # nm, compatible with mbuild

    liq_density = ref[job.sp.T][0]
    mol_density = liq_density / (job.sp.mol_weight * u.amu)
    vol_liq = job.sp.N_liq / mol_density
    liqboxl = vol_liq ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    liqboxl = round(float(liqboxl.in_units(u.nm).to_value()), 2)

    # Save to job document file
    job.doc.liqboxl = liqboxl  # nm, compatible with mbuild

    return job.doc.liqboxl, job.doc.vapboxl

@Project.pre(lambda job: "gemc_failed" not in job.doc)
@Project.post(lambda job: "vapboxl" in job.doc)
@Project.post(lambda job: "liqboxl" in job.doc)
@Project.operation
def calc_boxes(job):
    "Calculate the initial box length of the boxes"
    liqbox, vapbox = calc_box_helper(job)


# @Project.post(lambda job: "vapboxl" in job.doc)
# @Project.operation
# def calc_vapboxl(job):
#     "Calculate the initial box length of the vapor box"

#     import unyt as u

#     pressure = job.sp.P * u.bar
#     temperature = job.sp.T * u.K
#     nmols_vap = job.sp.N_vap

#     vol_vap = nmols_vap * u.kb * temperature / pressure
#     boxl = vol_vap ** (1.0 / 3.0)
#     # Strip unyts and round to 0.1 angstrom
#     boxl = round(float(boxl.in_units(u.nm).value), 2)
#     # Save to job document file
#     job.doc.vapboxl = boxl  # nm, compatible with mbuild

# @Project.post(lambda job: "liqboxl" in job.doc)
# @Project.operation
# def calc_liqboxl(job):
#     "Calculate the initial box length of the liquid box"

#     import unyt as u

#     nmols_liq = job.sp.N_liq
#     liq_density = job.sp.expt_liq_density * u.Unit("kg/m**3")
#     molweight = job.sp.mol_weight * u.amu
#     mol_density = liq_density / molweight
#     vol_liq = nmols_liq / mol_density
#     boxl = vol_liq ** (1.0 / 3.0)
#     # Strip unyts and round to 0.1 angstrom
#     boxl = round(float(boxl.in_units(u.nm).value), 2)
#     # Save to job document file
#     job.doc.liqboxl = boxl  # nm, compatible with mbuild


@Project.label
def nvt_finished(job):
    "Confirm a given nvt simulation is completed or not necessary"
    import numpy as np
    import os

    # If nsteps not in init, then GEMC ran without it earlier
    if "nsteps_nvt" not in job.sp:
        completed = True
    else:
        with job:
            try:
                thermo_data = np.genfromtxt("nvt.eq.out.prp", skip_header=3)
                completed = (
                    int(thermo_data[-1][0]) == job.sp.nsteps_nvt
                )  # job.sp.nsteps_liqeq
            except:
                completed = False
                pass

    return completed


@Project.pre(lambda job: "nsteps_nvt" in job.sp)
@Project.pre(lambda job: "gemc_failed" not in job.doc)
@Project.pre.after(create_forcefield, calc_boxes)
@Project.post(nvt_finished)
@Project.operation(directives={"omp_num_threads": 12})
def NVT_liqbox(job):
    "Equilibrate the liquid box using NVT simulation"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)

    # Create a new moves object and species list
    species_list = [compound_ff]
    moves = mc.MoveSet("nvt", species_list)

    # Property outputs relevant for NPT simulations
    thermo_props = ["energy_total", "pressure"]

    custom_args = {
        "vdw_style": "lj",
        "cutoff_style": "cut_tail",
        "vdw_cutoff": 12.0 * u.angstrom,
        "charge_style": "ewald",
        "charge_cutoff": 12.0 * u.angstrom,
        "ewald_accuracy": 1.0e-5,
        "mixing_rule": "lb",
        "units": "steps",
        "coord_freq": 1000,
        "prop_freq": 1000,
    }
    custom_args["run_name"] = "nvt.eq"
    custom_args["properties"] = thermo_props
    mols_to_add = [[job.sp.N_liq]]

    # Create box list
    boxl = job.doc.liqboxl
    box = mbuild.Box(lengths=[boxl, boxl, boxl])
    box_list = [box]
    system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

    # Move into the job dir and start doing things
    try:
        with job:
            # Run equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_nvt,
                temperature=job.sp.T * u.K,
                **custom_args
            )

            if "use_crit" not in job.doc:
                job.doc.use_crit = False

    except:
        # Note this overwrites liquid and vapor box lengths in job.doc
        liqbox, vapbox = calc_box_helper(job)
        # Create system with box lengths based on critical points
        boxl = job.doc.liqboxl
        box = mbuild.Box(lengths=[boxl, boxl, boxl])
        box_list = [box]
        system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

        try:
            with job:
                # Run equilibration
                mc.run(
                    system=system,
                    moveset=moves,
                    run_type="equilibration",
                    run_length=job.sp.nsteps_nvt,
                    temperature=job.sp.T * u.K,
                    **custom_args
                )
                job.doc.use_crit = True
        except:
            job.doc.nvt_failed == True
            raise Exception(
                "NVT failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
                + " at temperature "
                + str(job.sp.T)
            )


@Project.pre(lambda job: "nsteps_nvt" in job.sp)
@Project.pre.after(NVT_liqbox)
@Project.post(lambda job: job.isfile("nvt.final.xyz"))
@Project.post(lambda job: "nvt_liqbox_final_dim" in job.doc)
@Project.operation
def extract_final_NVT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = job.sp.N_liq * job.sp.N_atoms
    cmd = [
        "tail",
        "-n",
        str(lines + 2),
        job.fn("nvt.eq.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("nvt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("nvt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.nvt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


def npt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os

    with job:
        try:
            if job.isfile("liqbox-equil/equil.out.prp") and "nsteps_nvt" not in job.sp:
                thermo_data = np.genfromtxt("liqbox-equil/equil.out.prp", skip_header=3)
            else:
                thermo_data = np.genfromtxt("npt.eq.out.prp", skip_header=3)
            completed = (
                int(thermo_data[-1][0]) == job.sp.nsteps_liqeq
            ) 
        except:
            completed = False
            pass

    return completed


# @Project.pre.after(extract_final_NVT_config)
@Project.pre.after(create_forcefield, calc_boxes)
@Project.pre(lambda job: "gemc_failed" not in job.doc)
@Project.pre(nvt_finished)
@Project.post(npt_finished)
@Project.operation(directives={"omp_num_threads": 12})
def NPT_liqbox(job):
    "Equilibrate the liquid box"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)

    # Create box list and species list
    # Use nvt initial box length if available, otherwise use original calculated liqboxl
    if "nvt_liqbox_final_dim" in job.doc:
        with job:
            liq_box = mbuild.formats.xyz.read_xyz(job.fn("nvt.final.xyz"))
        boxl = job.doc.nvt_liqbox_final_dim
        liq_box.box = mbuild.Box(lengths=[boxl, boxl, boxl], angles=[90.0, 90.0, 90.0])
        liq_box.periodicity = [True, True, True]
    else:
        boxl = job.doc.liqboxl
        liq_box = mbuild.Box(lengths=[boxl, boxl, boxl])

    box_list = [liq_box]
    species_list = [compound_ff]
    mols_to_add = [[job.sp.N_liq]]

    if "nvt_liqbox_final_dim" in job.doc:
        system = mc.System(box_list, species_list, mols_in_boxes=mols_to_add)
    else:
        system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

    # Create a new moves object
    moves = mc.MoveSet("npt", species_list)

    # Edit the volume move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    new_prob_volume = 1.0 / job.sp.N_liq
    moves.prob_volume = new_prob_volume

    moves.prob_translate = moves.prob_translate + orig_prob_volume - new_prob_volume

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
    ]

    # Define custom args
    custom_args = {
        "run_name": "equil",
        "charge_style": "ewald",
        "rcut_min": 1.0 * u.angstrom,
        "vdw_cutoff": 12.0 * u.angstrom,
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq,
        "coord_freq": 1000,
        "prop_freq": 500,
        "properties": thermo_props,
    }

    custom_args["run_name"] = "npt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
    try:
        with job:
            # Run equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_liqeq,
                temperature=job.sp.T * u.K,
                pressure=job.sp.P * u.bar,
                **custom_args
            )

    except:
        # if GEMC failed with critical conditions as intial conditions, terminate with error
        if "use_crit" in job.doc and job.doc.use_crit == True:
            # If so, terminate with error and log failure in job document
            job.doc.gemc_failed = True
            # If the job fails twice and it doesn't have a obj_choice key delete it
            if "obj_choice" not in job.sp.keys():
                job.remove()
                raise Exception(
                    "NPT failed with critical and experimental starting conditions and the molecule is "
                    + job.sp.mol_name
                    + " at temperature "
                    + str(job.sp.T)
                )
            raise Exception(
                "NPT failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
                + " at temperature "
                + str(job.sp.T)
            )
        else:  # Otherwise, try with critical conditions
            job.doc.use_crit = True
            # Ensure that you will do an nvt simulation before the next gemc simulation
            job.sp.nsteps_nvt = 2500000
            # If GEMC fails, remove files in post conditions of previous operations
            del job.doc["vapboxl"]  # calc_boxes
            del job.doc["liqboxl"]  # calc_boxes
            with job:
                if job.isfile("nvt.eq.out.prp"):
                    os.remove("nvt.eq.out.prp")  # NVT_liqbox
                    os.remove("nvt.final.xyz")  # extract_final_NVT_config
                if "liqbox_final_dim" in job.doc:
                    del job.doc["liqbox_final_dim"]  # extract_final_NPT_config
                    os.remove("liqbox.xyz")  # extract_final_NPT_config


# @Project.label
# def liqbox_equilibrated(job):
#     "Confirm liquid box equilibration completed"

#     import numpy as np

#     try:
#         thermo_data = np.genfromtxt(
#             job.fn("liqbox-equil/equil.out.prp"), skip_header=3
#         )
#         completed = int(thermo_data[-1][0]) == job.sp.nsteps_liqeq
#     except:
#         completed = False
#         pass

#     return completed

# @Project.pre.after(create_forcefield, calc_liqboxl, calc_vapboxl)
# @Project.post(liqbox_equilibrated)
# @Project.operation(directives={"omp_num_threads": 2})
# def equilibrate_liqbox(job):
#     "Equilibrate the liquid box"

#     import os
#     import errno
#     import mbuild
#     import foyer
#     import mosdef_cassandra as mc
#     import unyt as u
#     ff = foyer.Forcefield(job.fn("ff.xml"))

#     # Load the compound and apply the ff
#     compound = mbuild.load(job.sp.smiles, smiles=True)
#     compound_ff = ff.apply(compound)

#     # Create box list and species list
#     boxl = job.doc.liqboxl
#     box = mbuild.Box(lengths=[boxl, boxl, boxl])

#     box_list = [box]
#     species_list = [compound_ff]

#     mols_to_add = [[job.sp.N_liq]]

#     system = mc.System(box_list, species_list, mols_to_add=mols_to_add)

#     # Create a new moves object
#     moves = mc.MoveSet("npt", species_list)

#     # Edit the volume move probability to be more reasonable
#     orig_prob_volume = moves.prob_volume
#     new_prob_volume = 1.0 / job.sp.N_liq
#     moves.prob_volume = new_prob_volume

#     moves.prob_translate = (
#         moves.prob_translate + orig_prob_volume - new_prob_volume
#     )

#     # Define thermo output props
#     thermo_props = [
#         "energy_total",
#         "pressure",
#         "volume",
#         "nmols",
#         "mass_density",
#     ]

#     # Define custom args
#     custom_args = {
#         "run_name": "equil",
#         "charge_style": "ewald",
#         "rcut_min": 1.0 * u.angstrom,
#         "vdw_cutoff": 12.0 * u.angstrom,
#         "units": "sweeps",
#         "steps_per_sweep": job.sp.N_liq,
#         "coord_freq": 500,
#         "prop_freq": 10,
#         "properties": thermo_props,
#     }

#     # Move into the job dir and start doing things
#     with job:
#         liq_dir = "liqbox-equil"
#         try:
#             os.mkdir(liq_dir)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#         os.chdir(liq_dir)
#         # Run equilibration
#         mc.run(
#             system=system,
#             moveset=moves,
#             run_type="equilibration",
#             run_length=job.sp.nsteps_liqeq,
#             temperature=job.sp.T * u.K,
#             pressure=job.sp.P * u.bar,
#             **custom_args
#         )


# @Project.pre.after(equilibrate_liqbox)
# @Project.post.isfile("liqbox.xyz")
# @Project.post(lambda job: "liqbox_final_dim" in job.doc)
# @Project.operation
# def extract_final_liqbox(job):
#     "Extract final coords and box dims from the liquid box simulation"

#     import subprocess

#     n_atoms = job.sp.N_liq * job.sp.N_atoms
#     cmd = [
#         "tail",
#         "-n",
#         str(n_atoms + 2),
#         job.fn("liqbox-equil/equil.out.xyz"),
#     ]

#     # Save final liuqid box xyz file
#     xyz = subprocess.check_output(cmd).decode("utf-8")
#     with open(job.fn("liqbox.xyz"), "w") as xyzfile:
#         xyzfile.write(xyz)

#     # Save final box dims to job.doc
#     box_data = []
#     with open(job.fn("liqbox-equil/equil.out.H")) as f:
#         for line in f:
#             box_data.append(line.strip().split())
#     job.doc.liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@Project.pre.after(NPT_liqbox)
@Project.post(
    lambda job: job.isfile("npt.final.xyz")
    or (job.isfile("liqbox.xyz") and "nsteps_nvt" not in job.sp)
)
@Project.post(lambda job: "npt_liqbox_final_dim" or "liqbox_final_dim" in job.doc)
@Project.operation
def extract_final_NPT_config(job):
    "Extract final coords and box dims from the liquid box simulation"

    import subprocess

    lines = job.sp.N_liq * job.sp.N_atoms
    cmd = [
        "tail",
        "-n",
        str(lines + 2),
        job.fn("npt.eq.out.xyz"),
    ]

    # Save final liuqid box xyz file
    xyz = subprocess.check_output(cmd).decode("utf-8")
    with open(job.fn("npt.final.xyz"), "w") as xyzfile:
        xyzfile.write(xyz)

    # Save final box dims to job.doc
    box_data = []
    with open(job.fn("npt.eq.out.H")) as f:
        for line in f:
            box_data.append(line.strip().split())
    job.doc.npt_liqbox_final_dim = float(box_data[-6][0]) / 10.0  # nm


@Project.label
def gemc_equil_complete(job):
    "Confirm gemc equilibration has completed"
    import numpy as np
    import glob
    import re
    
    # Find all files matching the restart pattern
    restart_pattern = job.fn("gemc.eq.rst.*.out.box1.prp")
    fallback_file = job.fn("gemc.eq.out.box1.prp")
    restart_files = glob.glob(restart_pattern)
    
    # Determine the file to use
    if restart_files:
        # Extract the highest restart value from filenames
        restart_numbers = [
            int(re.search(r"\.rst\.(\d+)\.out\.box1\.prp", f).group(1))
            for f in restart_files
        ]
        max_restart = max(restart_numbers)
        selected_file = job.fn(f"gemc.eq.rst.{max_restart:03d}.out.box1.prp")
    else:
        # Use fallback file if no restart files are found
        selected_file = fallback_file

    try:
        thermo_data = np.genfromtxt(selected_file, skip_header=2)
        #This line will fail until job.doc.nsteps_eq is defined
        if hasattr(job.doc, 'nsteps_eq'):
            completed = int(thermo_data[-1][0]) == job.doc.nsteps_eq
        else:
            completed = False
    except:
        completed = False

    return completed


@Project.label
def gemc_prod_complete(job):
    "Confirm gemc production has completed"
    import numpy as np

    try:
        thermo_data = np.genfromtxt(job.fn("prod.out.box1.prp"), skip_header=3)
        completed = int(thermo_data[-1][0]) == job.sp.nsteps_prod
    except:
        completed = False
        pass

    return completed

def plot_res_pymser(job, eq_col, results, name, box_name):
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    ax1.set_ylabel(name, color="black", fontsize=14, fontweight='bold')
    ax1.set_xlabel("GEMC step", fontsize=14, fontweight='bold')

    ax1.plot(range(len(eq_col)), 
            eq_col, 
            label = 'Raw data', 
            color='blue')

    ax1.plot(range(len(eq_col))[results['t0']:], 
            results['equilibrated'], 
            label = 'Equilibrated data', 
            color='red')

    ax1.plot([0, len(eq_col)], 
            [results['average'], results['average']], 
            color='green', zorder=4, 
            label='Equilibrated average')

    ax1.fill_between(range(len(eq_col)), 
                    results['average'] - results['uncertainty'], 
                    results['average'] + results['uncertainty'], 
                    color='lightgreen', alpha=0.3, zorder=4)

    ax1.set_yticks(np.arange(0, eq_col.max()*1.1, eq_col.max()/10))
    ax1.set_xlim(-len(eq_col)*0.02, len(eq_col)*1.02)
    ax1.tick_params(axis="y", labelcolor="black")

    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.hist(eq_col, 
            orientation=u'horizontal', 
            bins=30, 
            edgecolor='blue', 
            lw=1.5, 
            facecolor='white', 
            zorder=3)

    ax2.hist(results['equilibrated'], 
            orientation=u'horizontal', 
            bins=3, 
            edgecolor='red', 
            lw=1.5, 
            facecolor='white', 
            zorder=3)

    ymax = int(ax2.get_xlim()[-1])

    ax2.plot([0, ymax], 
            [results['average'], results['average']],
            color='green', zorder=4, label='Equilibrated average')

    ax2.fill_between(range(ymax), 
                    results['average'] - results['uncertainty'],
                    results['average'] + results['uncertainty'],
                    color='lightgreen', alpha=0.3, zorder=4)

    ax2.set_xlim(0, ymax)

    ax2.grid(alpha=0.5, zorder=1)

    fig.set_size_inches(9,5)
    fig.set_dpi(100)
    fig.tight_layout()
    save_name = 'MSER_eq_'+ box_name +'.png'
    fig.savefig(job.fn(save_name), dpi=300, facecolor='white')
    plt.close(fig)

def check_equil_converge(job, eq_data_dict, prod_tol):
    equil_matrix = []
    res_matrix = []
    prop_cols = [5]
    prop_names = ["Number of Moles"]
    try:
        # Load data for both boxes
        for key in list(eq_data_dict.keys()):
            eq_col = eq_data_dict[key]["data"]
        # df_box1 = np.genfromtxt(job.fn("gemc.eq.out.box1.prp"))
        # df_box2 = np.genfromtxt(job.fn("gemc.eq.out.box2.prp"))

        # Process both boxes in one loop
        # for box in [df_box1, df_box2]:
            # for prop_index in prop_cols:
            #     eq_col = box[:, prop_index - 1]
            # print(len(eq_col))
            batch_size = max(1, int(len(eq_col) * 0.0005))

            # Try with ADF test enabled, fallback without it if it fails
            try:
                results = pymser.equilibrate(eq_col, LLM=False, batch_size=batch_size, ADF_test=True, uncertainty='uSD', print_results=False)
                adf_test_failed = results["critical_values"]["1%"] <= results["adf"]
            except:
                results = pymser.equilibrate(eq_col, LLM=False, batch_size=batch_size, ADF_test=False, uncertainty='uSD', print_results=False)
                results["adf"], results["critical_values"], adf_test_failed = None, None, False

            equilibrium = len(eq_col) - results['t0'] >= prod_tol
            equil_matrix.append(equilibrium and not adf_test_failed)
            res_matrix.append(results)

        # Log results
        # print("ID", job.id, "AT", job.sp.atom_type, "T", job.sp.T)
        # print(equil_matrix)
        # log_text = '==============================================================================\n'
        
        for i, is_equilibrated in enumerate(equil_matrix):
            # box = df_box1 if i < len(prop_cols) else df_box2
            # box_name = "Liquid" if i < len(prop_cols) else "Vapor"
            # col_vals = box[:, prop_cols[i % len(prop_cols)] - 1]
            key_name = list(eq_data_dict.keys())[i]
            box_name = key_name.rsplit("_", 1)[0]
            col_vals = eq_data_dict[key_name]["data"]
            #plot all

            # if not all(equil_matrix):
            plot_res_pymser(job, col_vals, res_matrix[i], prop_names[i % len(prop_cols)], box_name)

            # Display outcome
            prod_cycles = len(col_vals) - res_matrix[i]['t0']
            if is_equilibrated:
                #Plot successful equilibration
                statement = f"       > Success! Found {prod_cycles} production cycles."
            else:
                #Plot failed equilibration
                statement = f"       > {box_name} Box Failure! "
                if res_matrix[i]["adf"] is None:
                    # Note: ADF test failed to complete
                    statement += f"ADF test failed to complete! "
                elif res_matrix[i]['adf'] > res_matrix[i]['critical_values']['1%']:
                    adf, one_pct = res_matrix[i]['adf'], res_matrix[i]['critical_values']['1%']
                    statement += f"ADF value: {adf}, 99% confidence value: {one_pct}! "
                if len(col_vals) - res_matrix[i]['t0'] < prod_tol:
                   statement += f"Only {prod_cycles} production cycles found."
                
            print(statement)

    except Exception as e:
        #This will cause an error in the GEMC operation which lets us know that the job failed
        raise Exception(f"Error processing job {job.id}: {e}")

    return all(equil_matrix)

@Project.pre(lambda job: "gemc_failed" not in job.doc)
@Project.pre.after(extract_final_NPT_config)
@Project.post(gemc_prod_complete)
@Project.operation(directives={"omp_num_threads": 2})
def run_gemc(job):
    "Run gemc"

    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)

    # Create box list and species list
    if "liqbox_final_dim" in job.doc:
        boxl_liq = job.doc.liqbox_final_dim  # saved in nm
    else:
        boxl_liq = job.doc.npt_liqbox_final_dim  # saved in nm
    # liq_box = mbuild.load(job.fn("liqbox.xyz"))
    with job:
        liq_box = mbuild.formats.xyz.read_xyz(job.fn("npt.final.xyz"))

    liq_box.box = mbuild.Box(
        lengths=[boxl_liq, boxl_liq, boxl_liq], angles=[90.0, 90.0, 90.0]
    )
    liq_box.periodicity = [True, True, True]

    boxl_vap = job.doc.vapboxl  # nm
    vap_box = mbuild.Box(lengths=[boxl_vap, boxl_vap, boxl_vap])

    box_list = [liq_box, vap_box]
    species_list = [compound_ff]

    mols_in_boxes = [[job.sp.N_liq], [0]]
    mols_to_add = [[0], [job.sp.N_vap]]

    system = mc.System(
        box_list,
        species_list,
        mols_in_boxes=mols_in_boxes,
        mols_to_add=mols_to_add,
    )

    # Create a new moves object
    moves = mc.MoveSet("gemc", species_list)

    # Edit the volume and swap move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    orig_prob_swap = moves.prob_swap
    new_prob_volume = 1.0 / (job.sp.N_liq + job.sp.N_vap)
    new_prob_swap = 4.0 / 0.05 / (job.sp.N_liq + job.sp.N_vap)
    moves.prob_volume = new_prob_volume
    moves.prob_swap = new_prob_swap

    moves.prob_translate = moves.prob_translate + orig_prob_volume - new_prob_volume
    moves.prob_translate = moves.prob_translate + orig_prob_swap - new_prob_swap

    # Define thermo output props
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    # Define custom args
    custom_args = {
        "run_name": "gemc.eq",
        "charge_style": "ewald",
        "rcut_min": 1.0 * u.angstrom,
        "charge_cutoff_box2": 0.4
        * (boxl_vap * u.nanometer).to("angstrom"),  # 25.0 * u.angstrom,
        "vdw_cutoff_box1": 12.0 * u.angstrom,
        "vdw_cutoff_box2": 0.4
        * (boxl_vap * u.nanometer).to("angstrom"),  # 25.0 * u.angstrom,
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq + job.sp.N_vap,
        "coord_freq": 500,
        "prop_freq": 10,
        "properties": thermo_props,
    }

    # Move into the job dir and start doing things
    try:
        #Inititalize counter and number of eq_steps
        count = 1
        total_eq_steps = job.sp.nsteps_eq
        eq_extend = int(job.sp.nsteps_eq/4)
        #Originally set the document eq_steps to 1 larger than the max number, it will be overwritten later
        job.doc.nsteps_eq = int(job.sp.nsteps_eq*4+1)
        with job:
            # Run initial equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_eq,
                temperature=job.sp.T * u.K,
                **custom_args
            )

            prop_cols = [5] #Use number of moles to decide equilibrium
            # Load initial eq data from both boxes
            df_box1 = np.genfromtxt(job.fn("gemc.eq.out.box1.prp"))
            df_box2 = np.genfromtxt(job.fn("gemc.eq.out.box2.prp"))

            # Process both boxes in one loop
            eq_data_dict = {}
            for b, box in enumerate([df_box1, df_box2]):
                box_name = "Liquid" if b == 0 else "Vapor"
                for prop_index in prop_cols:
                    eq_col = box[:, prop_index - 1]
                    #Save eq_col as a csv for later analysis
                    key = f"{box_name}_{prop_index}"
                    eq_col_file = job.fn(f"{box_name}_eq_col_{prop_index}.csv")
                    np.savetxt(eq_col_file, eq_col, delimiter=",")
                    #Save the eq_col and file to a dictionary for later use
                    eq_data_dict[key] = {"data": eq_col, "file": eq_col_file}

            #While we are using at most 12 attempts to equilibrate
            #Set production start tolerance as at least 25% of the original number of data points
            prod_tol_eq = int(eq_data_dict[key]["data"].size/4) 
            while count <= 13:
                # Check if equilibration is reached via the pymser algorithms
                is_equil = check_equil_converge(job, eq_data_dict, prod_tol_eq)
                #If equilibration is reached, break the loop and start production
                if is_equil:
                    break
                else:
                    #Increase the total number of eq steps by 25% of the original value and restart the simulation
                    total_eq_steps += int(eq_extend)
                    #If we've exceeded the maximum number of equilibrium steps, raise an exception
                    #This forces a retry with critical conditions or will note complete GEMC failure
                    if count == 13:
                        job.doc.equil_fail = True
                        raise Exception(f"GEMC equilibration failed to converge after {job.sp.nsteps_eq*4} steps")
                    #Otherwise continue equilibration
                    else:
                        mc.restart(
                            restart_from="gemc.eq",
                            run_type="equilibration",
                            total_run_length=total_eq_steps,
                            run_name="gemc.eq", #This will be overwritten by the restart gemc.eq.rst.xxx
                        )
                        #Add restart data to eq_col
                        # After each restart, load the updated properties data for both boxes
                        sim_box1 =  "gemc.eq" + f".rst.{count:03d}" + ".out.box1.prp"
                        sim_box2 =  "gemc.eq" + f".rst.{count:03d}" + ".out.box2.prp"
                        df_box1r = np.genfromtxt(job.fn(sim_box1))
                        df_box2r = np.genfromtxt(job.fn(sim_box2))
                        # df_box1 = np.genfromtxt(job.fn("gemc.eq.out.box1.prp"))
                        # df_box2 = np.genfromtxt(job.fn("gemc.eq.out.box2.prp"))

                        # Process and add the restart data to eq_col for each property in each box
                        for b, box in enumerate([df_box1r, df_box2r]):
                            box_name = "Liquid" if b == 0 else "Vapor"
                            for i, prop_index in enumerate(prop_cols):
                                #Get the key from the property and box name
                                key = f"{box_name}_{prop_index}"
                                # Extract the column data for this restart and append to accumulated data
                                eq_col_restart = box[:, prop_index - 1]
                                all_eq_data = np.concatenate((eq_data_dict[key]["data"], eq_col_restart))
                                #Save the new data to the eq_col file
                                np.savetxt(eq_data_dict[key]["file"], all_eq_data, delimiter=",")
                                #Overwite the current data in the eq_data_dict with restart data
                                eq_data_dict[key]["data"] = all_eq_data
                #Increase the counter
                count += 1

            #Set the step counter to whatever the final number of equilibration steps was
            job.doc.nsteps_eq = total_eq_steps
            job.doc.equil_fail = False

            # Run production
            mc.restart(
                restart_from="gemc.eq",
                run_type="production",
                total_run_length=job.sp.nsteps_prod,
                run_name="prod",
            )
    except:
        # if GEMC failed with critical conditions as intial conditions, terminate with error
        if "use_crit" in job.doc and job.doc.use_crit == True:
            # If so, terminate with error and log failure in job document
            job.doc.gemc_failed = True
            raise Exception(
                "GEMC failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
                + " at temperature "
                + str(job.sp.T)
            )
        else:
            # Otherwise, try with critical conditions
            job.doc.use_crit = True
            # Ensure that you will do an nvt simulation before the next gemc simulation
            job.sp.nsteps_nvt = 2500000
            # If GEMC fails, remove files in post conditions of previous operations
            del job.doc["vapboxl"]  # calc_boxes
            del job.doc["liqboxl"]  # calc_boxes
            with job:
                if job.isfile("nvt.eq.out.prp"):
                    os.remove("nvt.eq.out.prp")  # NVT_liqbox
                    os.remove("nvt.final.xyz")  # extract_final_NVT_config
                if job.isfile("npt.eq.out.prp"):
                    os.remove("npt.eq.out.prp")  # NPT_liqbox
                    os.remove("npt.final.xyz")  # extract_final_NPT_config
                if "liqbox_final_dim" in job.doc:
                    del job.doc["liqbox_final_dim"]  # extract_final_NPT_config
                    os.remove("liqbox.xyz")  # extract_final_NPT_config

#Create operation to delete failed jobs
@Project.label
def gemc_failed(job):
    "Confirm gemc failed"
    return "gemc_failed" in job.doc

@Project.pre(gemc_failed)
@Project.operation
def del_job(job):
    "Delete job if gemc failed"
    job.remove()

@Project.pre.after(run_gemc)
@Project.post(lambda job: "liq_density" in job.doc)
@Project.post(lambda job: "vap_density" in job.doc)
@Project.post(lambda job: "Pvap" in job.doc)
@Project.post(lambda job: "Hvap" in job.doc)
@Project.post(lambda job: "liq_enthalpy" in job.doc)
@Project.post(lambda job: "vap_enthalpy" in job.doc)
@Project.post(lambda job: "nmols_liq" in job.doc)
@Project.post(lambda job: "nmols_vap" in job.doc)
@Project.post(lambda job: "liq_density_unc" in job.doc)
@Project.post(lambda job: "vap_density_unc" in job.doc)
@Project.post(lambda job: "Pvap_unc" in job.doc)
@Project.post(lambda job: "Hvap_unc" in job.doc)
@Project.post(lambda job: "liq_enthalpy_unc" in job.doc)
@Project.post(lambda job: "vap_enthalpy_unc" in job.doc)
@Project.operation
def calculate_props(job):
    """Calculate the density"""

    import numpy as np

    sys.path.append("..")
    from utils.analyze_ms import block_average

    sys.path.remove("..")

    # Load the thermo data
    df_box1 = np.genfromtxt(job.fn("prod.out.box1.prp"))
    df_box2 = np.genfromtxt(job.fn("prod.out.box2.prp"))

    density_col = 6
    pressure_col = 3
    enth_col = 7
    n_mols_col = 5
    # pull density and take average
    liq_density = df_box1[:, density_col - 1]
    liq_density_ave = np.mean(liq_density)
    vap_density = df_box2[:, density_col - 1]
    vap_density_ave = np.mean(vap_density)

    # pull vapor pressure and take average
    Pvap = df_box2[:, pressure_col - 1]
    Pvap_ave = np.mean(Pvap)

    # pull enthalpy and take average
    liq_enthalpy = df_box1[:, enth_col - 1]
    liq_enthalpy_ave = np.mean(liq_enthalpy)
    vap_enthalpy = df_box2[:, enth_col - 1]
    vap_enthalpy_ave = np.mean(vap_enthalpy)

    # pull number of moles and take average
    nmols_liq = df_box1[:, n_mols_col - 1]
    nmols_liq_ave = np.mean(nmols_liq)
    nmols_vap = df_box2[:, n_mols_col - 1]
    # # Find indices where nmols_vap or nmols_liq contain zeros or NaN values
    # problematic_vap = np.where((nmols_vap == 0) | np.isnan(nmols_vap))[0]
    # problematic_liq = np.where((nmols_liq == 0) | np.isnan(nmols_liq))[0]

    # print(f"Indices with problematic values in nmols_vap: {problematic_vap}")
    # print(f"Indices with problematic values in nmols_liq: {problematic_liq}")

    # # Optionally print the actual values as well
    # print(f"Problematic values in nmols_vap: {nmols_vap[problematic_vap]}")
    # print(f"Problematic values in nmols_liq: {nmols_liq[problematic_liq]}")
    nmols_vap_ave = np.mean(nmols_vap)

    # calculate enthalpy of vaporization
    Hvap = (vap_enthalpy / nmols_vap) - (liq_enthalpy / nmols_liq)
    Hvap_ave = np.mean(Hvap)

    # save average density
    job.doc.liq_density = liq_density_ave
    job.doc.vap_density = vap_density_ave
    job.doc.Pvap = Pvap_ave
    job.doc.Hvap = Hvap_ave
    job.doc.liq_enthalpy = liq_enthalpy_ave
    job.doc.vap_enthalpy = vap_enthalpy_ave
    job.doc.nmols_liq = nmols_liq_ave
    job.doc.nmols_vap = nmols_vap_ave

    Props = {
        "liq_density": liq_density,
        "vap_density": vap_density,
        "Pvap": Pvap,
        "Hvap": Hvap,
        "liq_enthalpy": liq_enthalpy,
        "vap_enthalpy": vap_enthalpy,
        "nmols_liq": nmols_liq,
        "nmols_vap": nmols_vap,
    }

    for name, prop in Props.items():
        (means_est, vars_est, vars_err) = block_average(prop)

        with open(job.fn(name + "_blk_avg.txt"), "w") as ferr:
            ferr.write("# nblk_ops, mean, vars, vars_err\n")
            for nblk_ops, (mean_est, var_est, var_err) in enumerate(
                zip(means_est, vars_est, vars_err)
            ):
                ferr.write(
                    "{}\t{}\t{}\t{}\n".format(nblk_ops, mean_est, var_est, var_err)
                )

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))


#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _get_molec_dicts():
    # Load class properies for each training and testing molecule
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

    molec_dict = {
        "R14": R14,
        "R32": R32,
        "R50": R50,
        "R125": R125,
        "R134a": R134a,
        "R143a": R143a,
        "R170": R170,
        "R41": R41,
        "R23": R23,
        "R161": R161,
        "R152a": R152a,
        "R152": R152,
        "R143": R143,
        "R134": R134,
        "R116": R116,
    }
    return molec_dict


def _get_class_from_molecule(molecule_name):
    molec_dict = _get_molec_dicts()
    return {molecule_name: molec_dict[molecule_name]}


def _get_xml_from_molecule(molecule_name):
    if molecule_name == "R41":
        molec_xml_function = _generate_r41_xml
    elif molecule_name == "R116":
        molec_xml_function = _generate_r116_xml
    elif molecule_name == "R23":
        molec_xml_function = _generate_r23_xml
    elif molecule_name == "R152a":
        molec_xml_function = _generate_r152a_xml
    elif molecule_name == "R152":
        molec_xml_function = _generate_r152_xml
    elif molecule_name == "R143":
        molec_xml_function = _generate_r143_xml
    elif molecule_name == "R134":
        molec_xml_function = _generate_r134_xml
    elif molecule_name == "R161":
        molec_xml_function = _generate_r161_xml
    elif molecule_name == "R14":
        molec_xml_function = _generate_r14_xml
    elif molecule_name == "R32":
        molec_xml_function = _generate_r32_xml
    elif molecule_name == "R50":
        molec_xml_function = _generate_r50_xml
    elif molecule_name == "R125":
        molec_xml_function = _generate_r125_xml
    elif molecule_name == "R143a":
        molec_xml_function = _generate_r143a_xml
    elif molecule_name == "R170":
        molec_xml_function = _generate_r170_xml
    elif molecule_name == "R134a":
        molec_xml_function = _generate_r134a_xml
    else:
        raise ValueError("Molecule name not recognized")
    return molec_xml_function


def _generate_r14_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)(F)(F)(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="FC(F)(F)F" desc="F bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029" k="596.30"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.781024"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.195256" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
    )

    return content


def _generate_r50_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(H)(H)(H)(H)" desc="carbon"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="HC(H)(H)H" desc="H bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.68637877"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="-0.512608" sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.128152" sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r170_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(H)(H)(H)" desc="carbon"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H" desc="H bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31000265"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.68637877"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073484" k="388.01928783"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="hc" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.62757555" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="-0.006120"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="H1" charge="0.002040"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r134a_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(F)(F)(F)" desc="carbon bonded to 3 Fs and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="C(C)(H)(H)(F)" desc="carbon bonded to 2 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="FC(C)(F)F" desc="F bonded to C1"/>
  <Type name="F2" class="f" element="F" mass="19.000" def="FC(C)(H)H" desc="F bonded to C2"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956" k="544.13"/>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108" k="387.94"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029" k="596.30"/>
  <Angle class1="f" class2="c3" class3="h1" angle="1.88234" k="431.54"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.91201" k="327.86"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0207707" phase2="3.141592653589793"/>
  <Proper class1="f" class2="c3" class3="c3" class4="h1" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.79494566" phase2="0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.61542"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.020709"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.210427" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="F2" charge="-0.193556" sigma="{sigma_F2:0.6f}" epsilon="{epsilon_F2:0.6f}"/>
  <Atom type="H1" charge="0.115063"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_F2=job.sp.sigma_F2,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_F2=job.sp.epsilon_F2,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r143a_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(F)(F)(F)" desc="carbon bonded to 3 Fs and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="C(C)(H)(H)(H)" desc="carbon bonded to 3 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="hc" element="H" mass="1.008" def="H(C)" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.35"/>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.40"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.73"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956" k="544.13"/>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.920735" k="388.02"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029" k="596.30"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106" k="329.95"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.794946" phase2="0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.78821"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.583262"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.252614" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.184298"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r32_xml(job):

    content = """<ForceField>
<AtomTypes>
 <Type name="C" class="C" element="C" mass="12.011" def="C(F)(F)" desc="central carbon"/>
 <Type name="H" class="H" element="H" mass="1.008" def="H(C)" desc="first H bonded to C1_s1"/>
 <Type name="F" class="F" element="F" mass="18.998" def="F(C)" desc="F bonded to C1_s1"/>
</AtomTypes>
<HarmonicBondForce>
 <Bond class1="C" class2="H" length="0.10961" k="277566.56"/>
 <Bond class1="C" class2="F" length="0.13497" k="298653.92"/>
</HarmonicBondForce>
<HarmonicAngleForce>
 <Angle class1="H" class2="C" class3="H" angle="1.9233528356977512" k="326.352"/>
 <Angle class1="F" class2="C" class3="H" angle="1.898743693244631" k="427.6048"/>
 <Angle class1="F" class2="C" class3="F" angle="1.8737854849411122" k="593.2912"/>
</HarmonicAngleForce>
<NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
 <Atom type="C" charge="0.405467" sigma="{sigma_C}" epsilon="{epsilon_C}"/>
 <Atom type="H" charge="0.0480495" sigma="{sigma_H}" epsilon="{epsilon_H}"/>
 <Atom type="F" charge="-0.250783" sigma="{sigma_F}" epsilon="{epsilon_F}"/>
</NonbondedForce>
</ForceField>
""".format(
        sigma_C=job.sp.sigma_C,
        sigma_F=job.sp.sigma_F,
        sigma_H=job.sp.sigma_H,
        epsilon_C=job.sp.epsilon_C,
        epsilon_F=job.sp.epsilon_F,
        epsilon_H=job.sp.epsilon_H,
    )

    return content


def _generate_r125_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.011" def="C(C)(H)(F)(F)" desc="carbon bonded to 2 Fs, a H, and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.011" def="C(C)(F)(F)(F)" desc="carbon bonded to 3 Fs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="18.998" def="FC(C)(F)H" desc="F bonded to C1"/>
  <Type name="F2" class="f" element="F" mass="18.998" def="FC(C)(F)F" desc="F bonded to C2"/>
  <Type name="H1" class="h2" element="H" mass="1.008" def="H(C)" desc="single H bonded to C1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.15375" k="251793.12"/>
  <Bond class1="c3" class2="f" length="0.13497" k="298653.92"/>
  <Bond class1="c3" class2="h2" length="0.10961" k="277566.56"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.9065976748786053" k="553.1248"/>
  <Angle class1="c3" class2="c3" class3="h2" angle="1.9237019015481498" k="386.6016"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.8737854849411122" k="593.2912"/>
  <Angle class1="f" class2="c3" class3="h2" angle="1.898743693244631" k="427.6048"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0208" phase2="3.141592653589793"/>
  <Proper class1="" class2="c3" class3="c3" class4="" periodicity1="3" k1="0.6508444444444444" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.224067"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="0.500886"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.167131" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="F2" charge="-0.170758" sigma="{sigma_F2:0.6f}" epsilon="{epsilon_F2:0.6f}"/>
  <Atom type="H1" charge="0.121583"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_F2=job.sp.sigma_F2,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_F2=job.sp.epsilon_F2,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r41_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H(C)" desc="H bonded to C1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.119281"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.274252" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.051657"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r116_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(C)(F)(F)(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029483" k="596.29654763"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0207707" phase2="3.14159"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.420069"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.140023" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
    )

    return content


def _generate_r23_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C(F)(F)(F)" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="h3" element="H" mass="1.008" def="H(C)" desc="H bonded to C1"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h3" length="0.1095" k="278988.43"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029483" k="596.29654763"/>
  <Angle class1="f" class2="c3" class3="h3" angle="1.92003671" k="427.18040135"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.605792"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.222094" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.060490"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r152a_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="[C;X4](F)(F)" desc="carbon bonded to 2 Fs and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="[C;X4](C)(H)(H)(H)" desc="carbon bonded to 3 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F(C)" desc="F bonded to C1"/>
  <Type name="H1" class="h2" element="H" mass="1.008" def="H[C;%C1]" desc="H bonded to C1"/>
  <Type name="H2" class="hc" element="H" mass="1.008" def="H[C;%C2]" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.69"/>
  <Bond class1="c3" class2="h2" length="0.1100" k="273131.72"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073484" k="388.01928783"/>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="f" class2="c3" class3="h2" angle="1.89211144" k="429.77451333"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029483" k="596.29654763"/>
  <Angle class1="c3" class2="c3" class3="h2" angle="1.94761291" k="385.0925974"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.79494566" phase2="0"/>
  <Proper class1="h2" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.65268523" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.613473"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.502181"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.293648" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.021315"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="H2" charge="0.151563"  sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
    )

    return content


def _generate_r161_xml(job):

    content = """<ForceField>
 <AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="[C;X4](F)" desc="carbon bonded to 1 F and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="[C;X4](C)(H)(H)(H)" desc="carbon bonded to 3 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F" desc="F bonded to C1"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H[C;%C1]" desc="H bonded to C1"/>
  <Type name="H2" class="hc" element="H" mass="1.008" def="H[C;%C2]" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
  <Bond class1="c3" class2="hc" length="0.1092" k="282252.69"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="hc" angle="1.92073484" k="388.01928783"/>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108391" k="387.93614322"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
  <Angle class1="hc" class2="c3" class3="hc" angle="1.89106424" k="329.95108893"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.79494566" phase2="0"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="hc" periodicity1="3" k1="0.65268523" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.444699"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="-0.304621"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.334359" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="-0.028030"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="H2" charge="0.083447"  sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
    )

    return content


def _generate_r152_xml(job):

    content = """<ForceField name="R152 GAFF" version="1.0">
<AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F" desc="F"/>
  <Type name="H1" class="h1" element="H" mass="1.008" def="H" desc="H"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108391" k="387.93614322"/>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="h1" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.79494566" phase2="0"/>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0207707" phase2="3.14159265"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h1" periodicity1="3" k1="0.65268523" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.154155"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.295997" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.070921"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_F1=job.sp.sigma_F1,
        sigma_H1=job.sp.sigma_H1,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_H1=job.sp.epsilon_H1,
    )

    return content


def _generate_r134_xml(job):

    content = """<ForceField name="R134 GAFF" version="1.0">
<AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="C" desc="carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F" desc="F"/>
  <Type name="H1" class="h2" element="H" mass="1.008" def="H" desc="H"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="h2" length="0.11" k="273131.72"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="h2" angle="1.94761291" k="385.0925974"/>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="f" class2="c3" class3="h2" angle="1.89211144" k="429.77451333"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029483" k="596.29654763"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="h2" periodicity1="3" k1="0.65268523" phase1="0.0"/>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0207707" phase2="3.14159265"/>
  <Proper class1="h2" class2="c3" class3="c3" class4="h2" periodicity1="3" k1="0.65268523" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.293941"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="F1" charge="-0.192377" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="H1" charge="0.090813"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C,
        sigma_F1=job.sp.sigma_F,
        sigma_H1=job.sp.sigma_H,
        epsilon_C1=job.sp.epsilon_C,
        epsilon_F1=job.sp.epsilon_F,
        epsilon_H1=job.sp.epsilon_H,
    )

    return content


def _generate_r143_xml(job):

    content = """<ForceField name="R134 GAFF" version="1.0">
<AtomTypes>
  <Type name="C1" class="c3" element="C" mass="12.010" def="[C;X4](F)(F)" desc="carbon bonded to 2 Fs and another carbon"/>
  <Type name="C2" class="c3" element="C" mass="12.010" def="[C;X4](C)(F)(H)(H)" desc="carbon bonded to 2 Hs and another carbon"/>
  <Type name="F1" class="f" element="F" mass="19.000" def="F[C;%C1]" desc="F bonded to C1"/>
  <Type name="F2" class="f" element="F" mass="19.000" def="F[C;%C2]" desc="F bonded to C2"/>
  <Type name="H1" class="h2" element="H" mass="1.008" def="H[C;%C1]" desc="H bonded to C1"/>
  <Type name="H2" class="h1" element="H" mass="1.008" def="H[C;%C2]" desc="H bonded to C2"/>
 </AtomTypes>
 <HarmonicBondForce>
  <Bond class1="c3" class2="c3" length="0.1535" k="253634.31"/>
  <Bond class1="c3" class2="f" length="0.1344" k="304427.36"/>
  <Bond class1="c3" class2="h2" length="0.11" k="273131.72"/>
  <Bond class1="c3" class2="h1" length="0.1093" k="281080.35"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="c3" class2="c3" class3="f" angle="1.90956473" k="554.12559906"/>
  <Angle class1="c3" class2="c3" class3="h1" angle="1.92108391" k="387.93614322"/>
  <Angle class1="c3" class2="c3" class3="h2" angle="1.94761291" k="385.0925974"/>
  <Angle class1="f" class2="c3" class3="f" angle="1.87029483" k="596.29654763"/>
  <Angle class1="f" class2="c3" class3="h2" angle="1.89211144" k="429.77451333"/>
  <Angle class1="f" class2="c3" class3="h1" angle="1.8823376" k="431.53717916"/>
  <Angle class1="h1" class2="c3" class3="h1" angle="1.9120082" k="327.85584464"/>
 </HarmonicAngleForce>
 <PeriodicTorsionForce>
  <Proper class1="f" class2="c3" class3="c3" class4="f" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="5.0207707" phase2="3.14159265"/>
  <Proper class1="f" class2="c3" class3="c3" class4="h1" periodicity1="3" k1="0.0" phase1="0.0" periodicity2="1" k2="0.79494566" phase2="0"/>
  <Proper class1="f" class2="c3" class3="c3" class4="h2" periodicity1="3" k1="0.65268523" phase1="0.0"/>
  <Proper class1="h1" class2="c3" class3="c3" class4="h2" periodicity1="3" k1="0.65268523" phase1="0.0"/>
 </PeriodicTorsionForce>
 <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
  <Atom type="C1" charge="0.402256"  sigma="{sigma_C1:0.6f}" epsilon="{epsilon_C1:0.6f}"/>
  <Atom type="C2" charge="0.016261"  sigma="{sigma_C2:0.6f}" epsilon="{epsilon_C2:0.6f}"/>
  <Atom type="F1" charge="-0.228238" sigma="{sigma_F1:0.6f}" epsilon="{epsilon_F1:0.6f}"/>
  <Atom type="F2" charge="-0.204996" sigma="{sigma_F2:0.6f}" epsilon="{epsilon_F2:0.6f}"/>
  <Atom type="H1" charge="0.052577"  sigma="{sigma_H1:0.6f}" epsilon="{epsilon_H1:0.6f}"/>
  <Atom type="H2" charge="0.095189"  sigma="{sigma_H2:0.6f}" epsilon="{epsilon_H2:0.6f}"/>
 </NonbondedForce>
</ForceField>
""".format(
        sigma_C1=job.sp.sigma_C1,
        sigma_C2=job.sp.sigma_C2,
        sigma_F1=job.sp.sigma_F1,
        sigma_F2=job.sp.sigma_F2,
        sigma_H1=job.sp.sigma_H1,
        sigma_H2=job.sp.sigma_H2,
        epsilon_C1=job.sp.epsilon_C1,
        epsilon_C2=job.sp.epsilon_C2,
        epsilon_F1=job.sp.epsilon_F1,
        epsilon_F2=job.sp.epsilon_F2,
        epsilon_H1=job.sp.epsilon_H1,
        epsilon_H2=job.sp.epsilon_H2,
    )

    return content


if __name__ == "__main__":
    Project().main()
