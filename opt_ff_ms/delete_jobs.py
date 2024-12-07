import signac
import glob
import os
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job):
    "Delete data from previous operations"
    # del job.doc["nsteps_gemc_eq"] # run_gemc
    try:
        del job.doc["liq_density"] #Calc_props
        del job.doc["vap_density"] #Calc_props
        del job.doc["Hvap"] #Calc_props
        del job.doc["Pvap"] #Calc_props
    except:
        pass
    with job:
        for file_path in glob.glob("prod.*"):
            os.remove(file_path)

for job in project.find_jobs({"mol_name": "R32", "T": 321, "atom_type": 2}):
    # if job.sp.mol_name == "R134" and job.sp.T in [240]:
    print(job.id)
    delete_data(job)
    job.doc["gemc_failed"] = True

    # job.remove()