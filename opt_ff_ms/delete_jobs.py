import signac
import glob
import os
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job):
    "Delete data from previous operations"
    del job.doc["nsteps_gemc_eq"] # run_gemc
    del job.doc["liq_density"] #Calc_props
    with job:
        for file_path in glob.glob("prod.*"):
            os.remove(file_path)

for job in project.find_jobs({"mol_name": "R152", "T": 260, "atom_type": 6, "restart": 1}):
    # if job.sp.mol_name == "R134" and job.sp.T in [240]:
    print(job.id)
    delete_data(job)
    # job.remove()