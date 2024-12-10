import signac
import glob
import os
import math
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job):
    "Delete data from previous operations"
    job.doc["nsteps_gemc_eq"] = 110000
    job.doc["max_eq_steps"]= 110000
    del job.doc["liq_density"]
    with job:
        for file_path in glob.glob("prod.*"):
            os.remove(file_path)

for job in project.find_jobs():
    if "Hvap" in job.doc.keys() and math.isnan(job.doc["Hvap"]):
        print(job.id)
    # print(job.id)
    # if job.sp.mol_name == "R134" and job.sp.T in [240]:
    # delete_data(job)
    # job.remove()