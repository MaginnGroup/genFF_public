from flow import FlowProject, directives
import templates.ndcrc
import warnings
from pathlib import Path
import os
import sys
import unyt as u
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning)
sys.path.append("..")
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
sys.path.remove("..")

class Project(FlowProject):
    def __init__(self):
        #Set Project Path to be that of the current working directory
        current_path = Path(os.getcwd()).absolute()
        super().__init__(path = current_path)
        

@Project.post.isfile("ff.xml")
@Project.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""

    #Generate content based on job sp molecule name
    molec_xml_function = _get_xml_from_molecule(job.sp.mol_name)       
    content = molec_xml_function(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)

def calc_box_helper(job):
    "Calculate the initial box length of the boxes"

    import unyt as u
    #Get reference data from constants file
    #Load class properies for each training and testing molecule
    class_dict = _get_class_from_molecule(job.sp.mol_name)
    class_data = class_dict[job.sp.mol_name]
    # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in ProjectGAFF_gaff.py as needed)
    ref = {}
    
    #If the gemc simulation failed previously, use the critical values
    if "use_crit" in job.doc and job.doc.use_crit == True:
        rho_liq = class_data.expt_rhoc * u.kilogram/(u.meter)**3
        rho_vap = class_data.expt_rhoc * u.kilogram/(u.meter)**3
    else:
        #Initialize rho_liq and rho_vap as the experimental values
        rho_liq = job.sp.expt_liq_density * u.kilogram/(u.meter)**3
        rho_vap = class_data.expt_vap_density[int(job.sp.T)] * u.kilogram/(u.meter)**3

    # Create a tuple containing the values from each dictionary
    ref[int(job.sp.T)] = (rho_liq, rho_vap, job.sp.P)

    vap_density = ref[job.sp.T][1]
    mol_density = vap_density / (job.sp.mol_weight * u.amu)
    vol_vap = job.sp.N_vap / mol_density
    vapboxl = vol_vap ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    vapboxl = round(float(vapboxl.in_units(u.nm).to_value()), 2)

    #If molecule is R41, reduce the vapor box length by 20% to keep it inside the phase envelope
    if job.sp.mol_name == "R41":
        vapboxl = vapboxl*0.80

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
    "Confirm a given simulation is completed"
    import numpy as np
    import os 
    #If nsteps not in init, then GEMC ran without it earlier
    if "nsteps_nvt" not in job.sp:
        completed = True
    else:
        with job:
            try:
                thermo_data = np.genfromtxt(
                    "nvt.eq.out.prp", skip_header=3
                )
                completed = int(thermo_data[-1][0]) == job.sp.nsteps_nvt #job.sp.nsteps_liqeq
            except:
                completed = False
                pass

    return completed

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
    thermo_props = [
            "energy_total", 
            "pressure"
            ]

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
        #Note this overwrites liquid and vapor box lengths in job.doc
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
            raise Exception("NVT failed with critical and experimental starting conditions and the molecule is " + job.sp.mol_name)


@Project.pre.after(NVT_liqbox)
@Project.post(lambda job: job.isfile("nvt.final.xyz") or "nsteps_nvt" not in job.sp)
@Project.post(lambda job: "nvt_liqbox_final_dim" in job.doc or "liqbox_final_dim" in job.doc)
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
                thermo_data = np.genfromtxt("liqbox-equil/equil.out.prp", skip_header=3
                )
            else:
                thermo_data = np.genfromtxt(
                    "npt.eq.out.prp", skip_header=3
                )
            completed = int(thermo_data[-1][0]) == job.sp.nsteps_liqeq #job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@Project.pre.after(extract_final_NVT_config)
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

    with job:
        liq_box = mbuild.formats.xyz.read_xyz(job.fn("nvt.final.xyz"))

    boxl = job.doc.nvt_liqbox_final_dim

    liq_box.box = mbuild.Box(lengths=[boxl, boxl, boxl], angles=[90., 90., 90.])

    liq_box.periodicity = [True, True, True]

    box_list = [liq_box]

    species_list = [compound_ff]

    mols_in_boxes = [[job.sp.N_liq]]

    system = mc.System(box_list, species_list, mols_in_boxes=mols_in_boxes)

    # Create a new moves object
    moves = mc.MoveSet("npt", species_list)

    # Edit the volume move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    new_prob_volume = 1.0 / job.sp.N_liq
    moves.prob_volume = new_prob_volume

    moves.prob_translate = (
        moves.prob_translate + orig_prob_volume - new_prob_volume
    )

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
        "coord_freq": 500,
        "prop_freq": 10,
        "properties": thermo_props,
    }

    custom_args["run_name"] = "npt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
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
@Project.post(lambda job: job.isfile("npt.final.xyz") or (job.isfile("liqbox.xyz") and "nsteps_nvt" not in job.sp))
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

    try:
        thermo_data = np.genfromtxt(
            job.fn("equil.out.box1.prp"), skip_header=2
        )
        completed = int(thermo_data[-1][0]) == job.sp.nsteps_eq
    except:
        completed = False
        pass

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

    liq_box.box = mbuild.Box(lengths=[boxl_liq, boxl_liq, boxl_liq], angles=[90., 90., 90.])
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

    moves.prob_translate = (
        moves.prob_translate + orig_prob_volume - new_prob_volume
    )
    moves.prob_translate = (
        moves.prob_translate + orig_prob_swap - new_prob_swap
    )

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
        "charge_cutoff_box2": 0.4 * (boxl_vap * u.nanometer).to("angstrom"), #25.0 * u.angstrom,
        "vdw_cutoff_box1": 12.0 * u.angstrom,
        "vdw_cutoff_box2": 0.4 * (boxl_vap * u.nanometer).to("angstrom"), #25.0 * u.angstrom,
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq + job.sp.N_vap,
        "coord_freq": 500,
        "prop_freq": 10,
        "properties": thermo_props,
    }

    # Move into the job dir and start doing things
    try:
        with job:
            # Run equilibration
            mc.run(
                system=system,
                moveset=moves,
                run_type="equilibration",
                run_length=job.sp.nsteps_eq,
                temperature=job.sp.T * u.K,
                **custom_args
            )

            # Adjust custom args for production
            #custom_args["run_name"] = "prod"
            #custom_args["restart_name"] = "equil"

            # Run production
            mc.restart(
                restart_from="gemc.eq",
                run_type="production",
                total_run_length=job.sp.nsteps_prod,
                run_name="prod",
            )
    except:
        #if GEMC failed with critical conditions as intial conditions, terminate with error
        if "use_crit" in job.doc and job.doc.use_crit == True:
            #If so, terminate with error and log failure in job document
            job.doc.gemc_failed = True
            raise Exception("GEMC failed with critical and experimental starting conditions and the molecule is " + job.sp.mol_name)
        else:
            #Otherwise, try with critical conditions
            job.doc.use_crit = True
            #If GEMC fails, remove files in post conditions of previous operations
            del job.doc["vapboxl"] #calc_boxes
            del job.doc["liqboxl"] #calc_boxes
            with job:
                if job.isfile("nvt.eq.out.prp"): 
                    os.remove("nvt.eq.out.prp") #NVT_liqbox
                    os.remove("npt.eq.out.prp") #NPT_liqbox
                    os.remove("nvt.final.xyz") #extract_final_NVT_config
                    os.remove("npt.final.xyz") #extract_final_NPT_config
                else:
                    del job.doc["liqbox_final_dim"] #extract_final_NPT_config
                    os.remove("liqbox.xyz") #extract_final_NPT_config



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
    nmols_vap_ave = np.mean(nmols_vap)

    # calculate enthalpy of vaporization
    Hvap = (vap_enthalpy/nmols_vap) - (liq_enthalpy/nmols_liq)
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
        "Hvap" : Hvap,
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
                    "{}\t{}\t{}\t{}\n".format(
                        nblk_ops, mean_est, var_est, var_err
                    )
                )

        job.doc[name + "_unc"] = np.max(np.sqrt(vars_est))


#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _get_molec_dicts():
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
                    "R152": R152,
                    "R143": R143,
                    "R134": R134,
                    "R116": R116}
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
