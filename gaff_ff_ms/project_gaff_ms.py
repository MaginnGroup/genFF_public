from flow import FlowProject, directives
import templates.ndcrc
import warnings
from pathlib import Path
import os
import glob
import sys
import unyt as u
import copy
from pymser import pymser
import numpy as np
import matplotlib.pyplot as plt

# simulation_length must be consistent with the "units" field in custom args below
# For instance, if the "units" field is "sweeps" and simulation_length = 1000, 
# This will run a total of 1000 sweeps 
# (1 sweep = N steps, where N is the total number of molecules (job.sp.N_vap + job.sp.N_liq)
sys.path.append("..")
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
sys.path.remove("..")

warnings.filterwarnings("ignore", category=DeprecationWarning)

#Note - Must define Project class with a different name that other project.py files
class ProjectGAFF(FlowProject):
    def __init__(self):
        current_path = Path(os.getcwd()).absolute()
        #Set Project Path to be that of the current working directory
        super().__init__(path = current_path)
        
@ProjectGAFF.post.isfile("ff.xml")
@ProjectGAFF.operation
def create_forcefield(job):
    """Create the forcefield .xml file for the job"""
    #Generate content based on job sp molecule name
    molec_xml_function = _get_xml_from_molecule(job.sp.mol_name)       
    content = molec_xml_function(job)

    with open(job.fn("ff.xml"), "w") as ff:
        ff.write(content)
        
def nvt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os 

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

def npt_finished(job):
    "Confirm a given simulation is completed"
    import numpy as np
    import os

    with job:
        try:
            thermo_data = np.genfromtxt(
                "npt.eq.out.prp", skip_header=3
            )
            completed = int(thermo_data[-1][0]) == job.sp.nsteps_npt #job.sp.nsteps_liqeq
        except:
            completed = False
            pass

    return completed

@ProjectGAFF.label
def gemc_prod_complete(job):
    "Confirm gemc production has completed"
    import numpy as np

    try:
        with open(job.fn("prod.out.box1.prp"), "rb") as f:
            # Move the pointer to the end of the file, but leave space to find the last line
            f.seek(-2, os.SEEK_END)
            # Read backward until a newline is found
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            # Read the last line after finding the newline
            last_line = f.readline().decode()
        # Split the last line and extract the first number
        first_value = int(last_line.split()[0])
        completed = first_value == job.sp.nsteps_gemc_prod + job.doc.nsteps_gemc_eq
    except:
        completed = False
        pass

    return completed

def calc_box_helper(job):
    "Calculate the initial box length of the boxes"

    import unyt as u
    #Get reference data from constants file
    #Load class properies for each training and testing molecule
    class_dict = _get_class_from_molecule(job.sp.mol_name)
    class_data = class_dict[job.sp.mol_name]
    # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in ProjectGAFF_gaff.py as needed)
    # Loop over the keys of the dictionaries
    ref = {}
    #What is the best way to automate this if exp data crashes simulation?
    for t in class_data.expt_Pvap.keys():
        #Initialize rho_liq and rho_vap as the experimental values
        rho_liq = class_data.expt_liq_density[t] * u.kilogram/(u.meter)**3
        rho_vap = class_data.expt_vap_density[t] * u.kilogram/(u.meter)**3
        #If the gemc simulation failed previously, use the critical values
        if "use_crit" in job.doc and job.doc.use_crit == True:
            rho_liq = class_data.expt_rhoc * u.kilogram/(u.meter)**3
            rho_vap = class_data.expt_rhoc * u.kilogram/(u.meter)**3
        p_vap = class_data.expt_Pvap[t] * u.bar
        # Create a tuple containing the values from each dictionary
        ref[int(t)] = (rho_liq, rho_vap, p_vap)

    vap_density = ref[job.sp.T][1]
    mol_density = vap_density / (job.sp.mol_weight * u.amu)
    vol_vap = job.sp.N_vap / mol_density
    vapboxl = vol_vap ** (1.0 / 3.0)

    # Strip unyts and round to 0.1 angstrom
    vapboxl = round(float(vapboxl.in_units(u.nm).to_value()), 2)

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

@ProjectGAFF.post(lambda job: "vapboxl" in job.doc)
@ProjectGAFF.post(lambda job: "liqboxl" in job.doc)
@ProjectGAFF.operation
def calc_boxes(job):
    "Calculate the initial box length of the boxes"
    liqbox, vapbox = calc_box_helper(job)

@ProjectGAFF.pre.after(calc_boxes)
@ProjectGAFF.pre(lambda job: "gemc_failed" not in job.doc)
@ProjectGAFF.post(nvt_finished)
@ProjectGAFF.operation(directives={"omp_num_threads": 2})
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

    custom_args, custom_args_gemc = _get_custom_args(job)
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

@ProjectGAFF.pre.after(NVT_liqbox)
@ProjectGAFF.post.isfile("nvt.final.xyz")
@ProjectGAFF.post(lambda job: "nvt_liqbox_final_dim" in job.doc)
@ProjectGAFF.operation
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

@ProjectGAFF.pre.after(extract_final_NVT_config)
@ProjectGAFF.pre(lambda job: "gemc_failed" not in job.doc)
@ProjectGAFF.post(npt_finished)
@ProjectGAFF.operation(directives={"omp_num_threads": 2})
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
        "mass_density",
    ]

    # Define custom args
    custom_args, custom_args_gemc = _get_custom_args(job)
    custom_args["run_name"] = "npt.eq"
    custom_args["properties"] = thermo_props

    # Move into the job dir and start doing things
    with job:
        # Run equilibration
        #Load class properies for each training and testing molecule
        class_dict = _get_class_from_molecule(job.sp.mol_name)
        class_data = class_dict[job.sp.mol_name]
        # Reference data to compare to (i.e. experiments or other simulation studies) (load from constants file in project_gaff.py as needed)
        # Loop over the keys of the dictionaries
        for t, pvap in class_data.expt_Pvap.items():
            if t == job.sp.T:
                pressure = pvap * u.bar

        # Move into the job dir and start doing things
        try:
            # Run equilibration
            mc.run(
            system=system,
            moveset=moves,
            run_type="equilibration",
            run_length=job.sp.nsteps_npt,
            temperature=job.sp.T * u.K,
            pressure= pressure,
            **custom_args
            )
        except:
            # if GEMC failed with critical conditions as intial conditions, terminate with error
            if "use_crit" in job.doc and job.doc.use_crit == True:
                # If so, terminate with error and log failure in job document
                job.doc.gemc_failed = True
                raise Exception(
                    "NPT failed with critical and experimental starting conditions and the molecule is "
                    + job.sp.mol_name
                    + " at temperature "
                    + str(job.sp.T)
                )
            else:  # Otherwise, try with critical conditions
                job.doc.use_crit = True
                del job.doc["vapboxl"]  # calc_boxes
                del job.doc["liqboxl"]  # calc_boxes
                with job:
                    #Delete nvt, npt, and gemc equil/prod data
                    for file_path in glob.glob("nvt.*"):
                        os.remove(file_path)
                    for file_path in glob.glob("npt.*"):
                        os.remove(file_path)

@ProjectGAFF.pre.after(NPT_liqbox)
@ProjectGAFF.post.isfile("npt.final.xyz")
@ProjectGAFF.post(lambda job: "npt_liqbox_final_dim" in job.doc)
@ProjectGAFF.operation
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

@ProjectGAFF.label
def gemc_equil_complete(job):
    "Confirm gemc equilibration has completed"
    import numpy as np
    
    #Get last restart file
    try:
        last_file = get_last_checkpoint(job.fn("gemc.eq"))
        selected_file = job.fn(last_file + ".out.box1.prp")
    except:
        selected_file = job.fn("gemc.eq.out.box1.prp")

    #Check that the last step was completed
    try:
        with open(selected_file, "rb") as f:
            # Move the pointer to the end of the file, but leave space to find the last line
            f.seek(-2, os.SEEK_END)
            # Read backward until a newline is found
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            # Read the last line after finding the newline
            last_line = f.readline().decode()
        # Split the last line and extract the first number
        first_value = int(last_line.split()[0])
        #This line will fail until job.doc.nsteps_gemc_eq is defined
        if hasattr(job.doc, 'nsteps_gemc_eq'):
            completed = first_value == job.doc.nsteps_gemc_eq
        else:
            completed = False
    except:
        completed = False

    return completed

def delete_data(job, run_name):
    "Delete data from previous operations"
    del job.doc["vapboxl"]  # calc_boxes
    del job.doc["liqboxl"]  # calc_boxes
    del job.doc["nsteps_gemc_eq"]  # run_gemc
    with job:
        #Delete nvt, npt, and gemc equil/prod data
        for file_path in glob.glob("nvt.*"):
            os.remove(file_path)
        for file_path in glob.glob("npt.*"):
            os.remove(file_path)
        for file_path in glob.glob(run_name + ".*"):
            os.remove(file_path)
        for file_path in glob.glob("prod.*"):
            os.remove(file_path)
        if os.path.exists("Equil_Output.txt"):
            os.remove("Equil_Output.txt")


@ProjectGAFF.pre.after(extract_final_NPT_config)
@ProjectGAFF.pre(lambda job: "gemc_failed" not in job.doc)
@ProjectGAFF.post(gemc_prod_complete)
@ProjectGAFF.operation(directives={"omp_num_threads": 2})
def run_gemc(job):
    "Equilibrate GEMC"

    import os
    import errno
    import mbuild
    import foyer
    import mosdef_cassandra as mc
    import unyt as u
    import glob

    ff = foyer.Forcefield(job.fn("ff.xml"))

    # Load the compound and apply the ff
    compound = mbuild.load(job.sp.smiles, smiles=True)
    compound_ff = ff.apply(compound)

    with job:
        liq_box = mbuild.formats.xyz.read_xyz(job.fn("npt.final.xyz"))

    boxl_liq = job.doc.npt_liqbox_final_dim

    liq_box.box = mbuild.Box(lengths=[boxl_liq, boxl_liq, boxl_liq], angles=[90., 90., 90.])

    liq_box.periodicity = [True, True, True]

    boxl_vap = job.doc.vapboxl

    vap_box = mbuild.Box(lengths=[boxl_vap, boxl_vap, boxl_vap], angles=[90., 90., 90.])

    box_list = [liq_box, vap_box]

    species_list = [compound_ff]

    mols_in_boxes = [[job.sp.N_liq], [0]]

    mols_to_add = [[0], [job.sp.N_vap]]

    system = mc.System(box_list, species_list, mols_in_boxes=mols_in_boxes, mols_to_add=mols_to_add)

    # Create a new moves object
    moves = mc.MoveSet("gemc", species_list)


    # Edit the volume and swap move probability to be more reasonable
    orig_prob_volume = moves.prob_volume
    orig_prob_swap = moves.prob_swap
    new_prob_volume = 1.0 / (job.sp.N_vap + job.sp.N_liq)
    new_prob_swap = 4.0 / 0.05 / (job.sp.N_vap + job.sp.N_liq)
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
    custom_args, custom_args_gemc = _get_custom_args(job)
    custom_args_gemc["run_name"] = "gemc.eq"
    custom_args_gemc["properties"] = thermo_props

    custom_args_gemc["charge_cutoff_box2"] = 0.4 * (boxl_vap * u.nanometer).to("angstrom")
    custom_args_gemc["vdw_cutoff_box2"] = 0.4 * (boxl_vap * u.nanometer).to("angstrom")

    # Move into the job dir and start doing things
    try:
        with job:
            first_run = custom_args_gemc["run_name"] #gemc.eq
            # Run initial equilibration if it does not exxist
            if not has_checkpoint(first_run):
                mc.run(
                    system=system,
                    moveset=moves,
                    run_type="equilibration",
                    run_length=job.sp.nsteps_gemc_eq,
                    temperature=job.sp.T * u.K,
                    **custom_args_gemc
                )
            elif not check_complete(first_run):
                mc.restart(
                    restart_from=get_last_checkpoint(first_run),
                )

            init_gemc_liq = job.fn(first_run + ".out.box1.prp")
            init_gemc_vap = job.fn(first_run + ".out.box2.prp")
            prop_cols = [5] #Use number of moles to decide equilibrium
            # Load initial eq data from both boxes
            df_box1 = np.genfromtxt(init_gemc_liq)
            df_box2 = np.genfromtxt(init_gemc_vap)

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

            #Set production start tolerance as at least 25% of the total number of data points
            # prod_tol_eq = int(eq_data_dict[key]["data"].size/4)
            if os.path.exists("Equil_Output.txt"): #Remove the file if it exists
                os.remove("Equil_Output.txt") 

            #Set number of iterations per extension and intitialize counter and total number of steps
            eq_extend = int(job.sp.nsteps_gemc_eq/4)
            total_eq_steps = job.sp.nsteps_gemc_eq
            count = 1

            #Get the total number of equilibration restarts and steps so far
            num_restarts = len(list_with_restarts(custom_args_gemc["run_name"] + ".out.chk")) -1
            existing_eq_steps = job.sp.nsteps_gemc_eq + num_restarts*eq_extend
            
            #Inititalize max number of eq_steps
            if "max_eq_steps" not in job.doc:
                #If no value exists, set it as 10 times the original number of eq steps
                job.doc.max_eq_steps = job.sp.nsteps_gemc_eq*10
            #The max number of steps is the larger of the number of steps + the org number of steps or the current max
            max_eq_steps = np.maximum(job.doc.max_eq_steps, existing_eq_steps + job.sp.nsteps_gemc_eq)
            #Originally set the document eq_steps to the max number, it will be overwritten later
            job.doc.nsteps_gemc_eq = int(max_eq_steps)

            #While the max number of eq steps has not been reached
            while total_eq_steps <= max_eq_steps:
                #Set production start tolerance as at least 25% of the total number of data points
                prod_tol_eq = int(total_eq_steps/4)/custom_args_gemc["prop_freq"]
                #Set this run and last last run
                this_run = custom_args_gemc["run_name"] + f".rst.{count:03d}"
                prior_run = get_last_checkpoint(custom_args_gemc["run_name"])
                # Check if equilibration is reached via the pymser algorithms
                if total_eq_steps >= existing_eq_steps and total_eq_steps >= 10*job.sp.nsteps_gemc_eq:
                    is_equil = check_equil_converge(job, eq_data_dict, prod_tol_eq)
                else:
                    is_equil = False

                if is_equil:
                    break
                else:
                    #Increase the total number of eq steps by 25% of the original value and restart the simulation
                    total_eq_steps += int(eq_extend)
                    #If we've exceeded the maximum number of equilibrium steps, raise an exception
                    #This forces a retry with critical conditions or will note complete GEMC failure
                    if total_eq_steps > max_eq_steps:
                        job.doc.equil_fail = True
                        raise Exception(f"GEMC equilibration failed to converge after {max_eq_steps} steps")
                    #Otherwise continue equilibration
                    else:
                        #Check if checkpoint file exists, if so, we've already done this restart
                        # if not, restart the simulation
                        if not has_checkpoint(this_run):
                            mc.restart(
                            restart_from=prior_run,
                            run_type="equilibration",
                            total_run_length=total_eq_steps,
                            run_name = this_run )
                        elif not check_complete(this_run):
                            mc.restart(
                                restart_from=get_last_checkpoint(this_run),
                            )

                        #Add restart data to eq_col
                        # After each restart, load the updated properties data for both boxes
                        sim_box1 =  this_run + ".out.box1.prp"
                        sim_box2 =  this_run + ".out.box2.prp"
                        df_box1r = np.genfromtxt(job.fn(sim_box1))
                        df_box2r = np.genfromtxt(job.fn(sim_box2))

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
            job.doc.nsteps_gemc_eq = total_eq_steps
            job.doc.equil_fail = False
            total_sim_steps = int(job.sp.nsteps_gemc_prod + job.doc.nsteps_gemc_eq)
            # Run production
            if not has_checkpoint("prod"):
                mc.restart(
                    restart_from=prior_run,
                    run_type="production",
                    total_run_length=total_sim_steps,
                    run_name="prod",
                )
            elif not check_complete("prod"):
                mc.restart(
                    restart_from=get_last_checkpoint("prod"),
                )

    except:
        #If equilibration wasn't long enough, don't delete, we'll just extend the simulation
        if "equil_fail" in job.doc and job.doc.equil_fail == True:
            job.doc.max_eq_steps = max_eq_steps + job.sp.nsteps_gemc_eq
            job.doc.nsteps_gemc_eq = job.doc.max_eq_steps
        #if GEMC failed with critical conditions as intial conditions, terminate with error
        elif "use_crit" in job.doc and job.doc.use_crit == True:
            job.doc.gemc_failed = True
            raise Exception(
                "GEMC failed with critical and experimental starting conditions and the molecule is "
                + job.sp.mol_name
                + " at temperature "
                + str(job.sp.T)
            )
        else:  
            # If the simulation failed for another reason, try with critical conditions
            job.doc.use_crit = True
            if "equil_fail" in job.doc:
                del job.doc["equil_fail"]
            # If GEMC fails, remove files in post conditions of previous operations
            delete_data(job, custom_args_gemc["run_name"])

#@Project.post(lambda job: "liq_density_unc" in job.doc)
#@Project.post(lambda job: "vap_density_unc" in job.doc)
#@Project.post(lambda job: "Pvap_unc" in job.doc)
#@Project.post(lambda job: "Hvap_unc" in job.doc)
#@Project.post(lambda job: "liq_enthalpy_unc" in job.doc)
#@Project.post(lambda job: "vap_enthalpy_unc" in job.doc)
#Create operation to delete failed jobs
@ProjectGAFF.label
def gemc_failed(job):
    "Confirm gemc failed"
    return "gemc_failed" in job.doc

@ProjectGAFF.pre(gemc_failed)
@ProjectGAFF.operation
def del_job(job):
    "Delete job if gemc failed"
    job.remove()

@ProjectGAFF.pre.after(run_gemc)
@ProjectGAFF.post.isfile("energy.png")
@ProjectGAFF.post(lambda job: "liq_density" in job.doc)
@ProjectGAFF.post(lambda job: "vap_density" in job.doc)
@ProjectGAFF.post(lambda job: "Pvap" in job.doc)
@ProjectGAFF.post(lambda job: "Hvap" in job.doc)
@ProjectGAFF.post(lambda job: "liq_enthalpy" in job.doc)
@ProjectGAFF.post(lambda job: "vap_enthalpy" in job.doc)
@ProjectGAFF.post(lambda job: "nmols_liq" in job.doc)
@ProjectGAFF.post(lambda job: "nmols_vap" in job.doc)
@ProjectGAFF.operation
def calculate_props(job):
    """Calculate the density"""

    import numpy as np
    import pylab as plt
    sys.path.append("..")
    from utils.analyze_ms import block_average
    sys.path.remove("..")
    
    thermo_props = [
        "energy_total",
        "pressure",
        "volume",
        "nmols",
        "mass_density",
        "enthalpy",
    ]

    with job:
        df_box1 = np.genfromtxt("prod.out.box1.prp")
        df_box2 = np.genfromtxt("prod.out.box2.prp")

    energy_col = 1
    density_col = 5
    pressure_col = 2
    enth_col = 6
    n_mols_col = 4

    # pull steps
    steps = df_box1[:, 0]

    # pull energy
    liq_energy= df_box1[:, energy_col]
    vap_energy= df_box2[:, energy_col]

    # pull density and take average
    liq_density = df_box1[:, density_col]
    liq_density_ave = np.mean(liq_density)
    vap_density = df_box2[:, density_col]
    vap_density_ave = np.mean(vap_density)
    
    # pull vapor pressure and take average
    Pvap = df_box2[:, pressure_col]
    Pvap_ave = np.mean(Pvap)
    
    # pull enthalpy and take average
    liq_enthalpy = df_box1[:, enth_col]
    liq_enthalpy_ave = np.mean(liq_enthalpy)
    vap_enthalpy = df_box2[:, enth_col]
    vap_enthalpy_ave = np.mean(vap_enthalpy)
    
    # pull number of moles and take average
    nmols_liq = df_box1[:, n_mols_col]
    nmols_liq_ave = np.mean(nmols_liq)
    nmols_vap = df_box2[:, n_mols_col]
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

    font = {'weight' : 'normal',
                    'size'   : 12}
    
    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Energy')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"Energy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(steps, liq_energy, label='Liquid Energy')
    ax.plot(steps, vap_energy, label='Vapor Energy')
    ax.legend(loc="best")

    with job:
        plt.savefig("energy.png")
        plt.close(fig)

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

@ProjectGAFF.label
def plot_finished(job):
    "Confirm plots have been made"
    import numpy as np
    import os 

    last_plot = job.fn(f"all-energy-{job.sp.T}.png")
    if os.path.exists(last_plot):
        completed = True
    else:
        completed = False

    return completed

@ProjectGAFF.pre.after(run_gemc)
@ProjectGAFF.post(plot_finished)
@ProjectGAFF.operation
def plot(job):
    import pandas as pd
    import pylab as plt

    with job:

        nvt_box1 = pd.read_table("nvt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure"], skiprows=3)
        npt_box1 = pd.read_table("npt.eq.out.prp", sep="\s+", names=["step", "energy", "pressure", "density"], skiprows=3)
        gemc_eq_box1 = pd.read_table("prod.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_eq_box2 = pd.read_table("prod.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_prod_box1 = pd.read_table("prod.out.box1.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
        gemc_prod_box2 = pd.read_table("prod.out.box2.prp", sep="\s+", names=["step", "energy", "pressure", "volume", "nmols", "density", "enthalpy"], skiprows=3)
   
    font = {'weight' : 'normal',
                    'size'   : 12}
  

    #####################
    # GEMC Vapor Pressure
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Pressure (bar)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Vapor pressure vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box2["step"][20:], gemc_eq_box2["pressure"][20:], label='GEMC-eq', color='red')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["pressure"], label='GEMC-prod', color='indianred')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"gemc-pvap-{job.sp.T}.png")
        plt.close(fig)

    #####################
    # GEMC nmols
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Number of molecules')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Number of molecules vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["nmols"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["nmols"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["nmols"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["nmols"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"gemc-nmols-{job.sp.T}.png")
        plt.close(fig)

    #####################
    # GEMC volume
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Volume $\AA^3$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Volume vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["volume"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["volume"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["volume"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["volume"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"gemc-volume-{job.sp.T}.png")
        plt.close(fig)

    #####################
    # GEMC density
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Density $(kg / m^3)$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Density vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["density"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["density"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["density"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["density"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"gemc-density-{job.sp.T}.png")
        plt.close(fig)

    #####################
    # GEMC enthalpy 
    #####################

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Enthalpy (kJ/mol-ext)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    ax.title.set_text(f"Enthalpy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["enthalpy"], label='GEMC-eq-box1', color='blue')
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["enthalpy"], label='GEMC-eq-box2', color='red')
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["enthalpy"], label='GEMC-prod-box1', color='royalblue')
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["enthalpy"], label='GEMC-prod-box2', color='indianred')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"gemc-enthalpy-{job.sp.T}.png")
        plt.close(fig)


    #############
    # NPT-Density
    #############

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Density $(kg / m^3)$')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"NPT Density vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(npt_box1["step"], npt_box1["density"], label='NpT')

    ax.legend(loc="best")
    with job:
        plt.savefig(f"npt-density-{job.sp.T}.png")
        plt.close(fig)

    # Shift steps so that we get an overall plot of energy across 
    # different workflow steps

    npt_box1["step"] += nvt_box1["step"].iloc[-1]
    gemc_eq_box1["step"] += npt_box1["step"].iloc[-1]
    gemc_eq_box2["step"] += npt_box1["step"].iloc[-1]
    gemc_prod_box1["step"] += npt_box1["step"].iloc[-1]
    gemc_prod_box2["step"] += npt_box1["step"].iloc[-1]

    #############
    # Energy
    #############

    fig, ax = plt.subplots(1, 1)
    
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    
    ax.set_xlabel(r'MC steps or sweeps')
    ax.set_ylabel('Energy (kJ/mol-ext)')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    
    ax.title.set_text(f"Liquid Energy vs MC Steps or Sweeps @ {job.sp.T} K")
    ax.plot(nvt_box1["step"][20:], nvt_box1["energy"][20:], label='NVT', color="black")

    ax.plot(npt_box1["step"], npt_box1["energy"], label='NpT', color="gray")
    ax.plot(gemc_eq_box1["step"], gemc_eq_box1["energy"], label='GEMC-eq-box1', color="blue")
    ax.plot(gemc_prod_box1["step"], gemc_prod_box1["energy"], label='GEMC-prod-box1', color="royalblue")
    ax.plot(gemc_eq_box2["step"], gemc_eq_box2["energy"], label='GEMC-eq-box2', color="red")
    ax.plot(gemc_prod_box2["step"], gemc_prod_box2["energy"], label='GEMC-prod-box2', color="indianred")

    ax.legend(loc="best")
    with job:
        plt.savefig(f"all-energy-{job.sp.T}.png")
        plt.close(fig)

#####################################################################
################# HELPER FUNCTIONS BEYOND THIS POINT ################
#####################################################################
def _get_custom_args(job):
    # Define custom args
    # See page below for all options 
    # https://mosdef-cassandra.readthedocs.io/en/latest/guides/kwargs.html
    custom_args = {
        "vdw_style": "lj",
        "cutoff_style": "cut_tail",
        "vdw_cutoff": 12.0 * u.angstrom,
        "charge_style": "ewald",
        "charge_cutoff": 12.0 * u.angstrom, 
        "ewald_accuracy": 1.0e-5, 
        "mixing_rule": "lb",
        "units": "sweeps",
        "steps_per_sweep": job.sp.N_liq,
        "coord_freq": 500,
        "prop_freq": 10,
    }

    custom_args_gemc = copy.deepcopy(custom_args)
    custom_args_gemc["steps_per_sweep"] = job.sp.N_liq + job.sp.N_vap
    custom_args_gemc["vdw_cutoff_box1"] = custom_args["vdw_cutoff"] 
    custom_args_gemc["charge_cutoff_box1"] = custom_args["charge_cutoff"]

    return custom_args, custom_args_gemc

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

    content = """<ForceField name="R14 GAFF" version="1.0">
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
  <Atom type="C1" charge="0.781024"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="F1" charge="-0.195256" sigma="0.3118" epsilon="0.255221"/>
 </NonbondedForce>
</ForceField>
"""

    return content

def _generate_r50_xml(job): 

    content = """<ForceField name="R50 GAFF" version="1.0">
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
  <Atom type="C1" charge="-0.512608"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="H1" charge="0.128152" sigma="0.265" epsilon="0.06569256"/>
 </NonbondedForce>
</ForceField>"""
    return content

def _generate_r170_xml(job): 

    content = """<ForceField name="R170 GAFF" version="1.0">
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
  <Atom type="C1" charge="-0.006120"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="H1" charge="0.002040"  sigma="0.265" epsilon="0.06569256"/>
 </NonbondedForce>
</ForceField>"""

    return content

def _generate_r134a_xml(job): 

    content = """<ForceField name="HFC-134a GAFF" version="1.0">
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
  <Atom type="C1" charge="0.61542"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="C2" charge="-0.020709"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="F1" charge="-0.210427" sigma="0.3118" epsilon="0.255221"/>
  <Atom type="F2" charge="-0.193556" sigma="0.3118" epsilon="0.255221"/>
  <Atom type="H1" charge="0.115063"  sigma="0.247" epsilon="0.06569256"/>
 </NonbondedForce>
</ForceField>"""
    return content

def _generate_r143a_xml(job):

    content = """<ForceField name="HFC-143a GAFF" version="1.0">
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
  <Atom type="C1" charge="0.78821"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="C2" charge="-0.583262"  sigma="0.340" epsilon="0.45773"/>
  <Atom type="F1" charge="-0.252614" sigma="0.3118" epsilon="0.255221"/>
  <Atom type="H1" charge="0.184298"  sigma="0.265" epsilon="0.065693"/>
 </NonbondedForce>
</ForceField>"""

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
        sigma_C=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H=float((2.293 * u.Angstrom).in_units(u.nm).value),
        epsilon_C=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_C2=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_F2=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.293 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_C2=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F2=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
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
        sigma_C1= float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.471 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1= float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.115 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_C2=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.293 * u.Angstrom).in_units(u.nm).value),
        sigma_H2=float((2.650 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_C2=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H2=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_C2=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.471 * u.Angstrom).in_units(u.nm).value),
        sigma_H2=float((2.650 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_C2=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H2=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.471 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.293 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        
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
        sigma_C1=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_C2=float((3.400 * u.Angstrom).in_units(u.nm).value),
        sigma_F1=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_F2=float((3.118 * u.Angstrom).in_units(u.nm).value),
        sigma_H1=float((2.293 * u.Angstrom).in_units(u.nm).value),
        sigma_H2=float((2.471 * u.Angstrom).in_units(u.nm).value),
        epsilon_C1=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_C2=float(55.052 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F1=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_F2=float(30.696 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H1=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
        epsilon_H2=float(7.901 * (u.K * u.kb).in_units("kJ/mol")),
    )

    return content

def has_checkpoint(run_name):
    """Check whether there is a checkpoint for run_name."""
    fname = run_name + ".out.chk"
    return os.path.exists(fname)

def check_complete(run_name):
    """Check whether MoSDeF Cassandra simulation with run_name or its last restart has completed."""
    complete = False
    fname = run_name + ".out.log"
    loglist = list_with_restarts(fname)
    if not loglist:
        return complete
    with loglist[-1].open() as f:
        for line in f:
            if "Cassandra simulation complete" in line:
                complete = True
                break
    return complete

def list_with_restarts(fpath):
    """List fpath and its restart versions in order as pathlib Path objects."""
    fpath = Path(fpath)
    if not fpath.exists():
        return []
    parent = fpath.parent
    fname = fpath.name
    fnamesplit = fname.split(".out.")
    run_name = fnamesplit[0]
    suffix = fnamesplit[1]
    restarts = [
        Path(parent, f)
        for f in sorted(list(parent.glob(run_name + ".rst.*.out." + suffix)))
    ]
    restarts.insert(0, fpath)  # prepend fpath to list of restarts
    return restarts

def get_last_checkpoint(run_name):
    """Get name of last restart based on run_name."""
    fname = run_name + ".out.chk"
    return list_with_restarts(fname)[-1].name.split(".out.")[0]

def plot_res_pymser(job, eq_col, results, name, box_name):
    fig, [ax1, ax2] = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

    ax1.set_ylabel(name, color="black", fontsize=14, fontweight='bold')
    ax1.set_xlabel("GEMC Steps", fontsize=14, fontweight='bold')

    ax1.plot(range(0, len(eq_col)*10, 10), 
            eq_col, 
            label = 'Raw data', 
            color='blue')

    ax1.plot(range(0, len(eq_col)*10, 10)[results['t0']:], 
            results['equilibrated'], 
            label = 'Equilibrated data', 
            color='red')

    ax1.plot([0, len(eq_col)*10], 
            [results['average'], results['average']], 
            color='green', zorder=4, 
            label='Equilibrated average')

    ax1.fill_between(range(0, len(eq_col)*10, 10), 
                    results['average'] - results['uncertainty'], 
                    results['average'] + results['uncertainty'], 
                    color='lightgreen', alpha=0.3, zorder=4)

    ax1.set_yticks(np.arange(0, eq_col.max()*1.1, eq_col.max()/10))
    ax1.set_xlim(-len(eq_col)*10*0.02, len(eq_col)*10*1.02)
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

    bin_red = 10
    ax2.hist(results['equilibrated'], 
            orientation=u'horizontal', 
            bins=bin_red, 
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
                
            with open("Equil_Output.txt", "a") as f:
                print(statement, file=f)

    except Exception as e:
        #This will cause an error in the GEMC operation which lets us know that the job failed
        raise Exception(f"Error processing job {job.id}: {e}")

    return all(equil_matrix)
if __name__ == "__main__":
    ProjectGAFF().main()