import signac
import glob
import os
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job, run_name):
    "Delete data from previous operations"
    del job.doc["nsteps_gemc_eq"] # run_gemc
    with job:
        for file_path in glob.glob("prod.*"):
            os.remove(file_path)

for job in project.find_jobs({"mol_name": "R116", "T": 210, "restart": 3}):
    # if job.sp.mol_name == "R134" and job.sp.T in [240]:
    print(job.id)
    job.remove()