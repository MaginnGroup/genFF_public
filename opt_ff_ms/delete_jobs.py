import signac
import glob
import os
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job):
    "Delete data from previous operations"
    # if job.doc["nsteps_gemc_eq"] < 110000 or job.doc["max_eq_steps"] < 110000:
    #     job.doc["nsteps_gemc_eq"] = 110000 # run_gemc
    #     job.doc["max_eq_steps"] = 110000 # run_gemc
    try:
        del job.doc["Hvap"] #Calc_props
    except:
        pass
    # with job:
    #     for file_path in glob.glob("prod.*"):
    #         os.remove(file_path)

for job in project.find_jobs():
    if "Hvap" in job.doc.keys() and job.doc["Hvap"] == "NaN":
        print(job.id)
    # delete_data(job)
    # job.doc["gemc_failed"] = True

    # job.remove()