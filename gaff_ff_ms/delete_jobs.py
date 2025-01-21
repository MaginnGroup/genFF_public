import signac
import glob
import os
import math
# Load the project
project = signac.get_project()

import signac
import glob
import os
import shutil
# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
def delete_data(job):
    "Delete data from previous operations"
    # if job.doc["nsteps_gemc_eq"] < 110000 or job.doc["max_eq_steps"] < 110000:
    #     job.doc["nsteps_gemc_eq"] = 110000 # run_gemc
    #     job.doc["max_eq_steps"] = 110000 # run_gemc
    with job:
        subfolder = "old_results_5"
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for file_path in glob.glob("MSER*"):
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        for file_path in glob.glob("*_eq_col_*"):
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        for file_path in glob.glob("box*.in.xyz"):
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        for file_path in glob.glob("gemc.eq.*"):
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        for file_path in glob.glob("mosdef_cassandra_*.log"):
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        for file_path in glob.glob("prod.*"):
            # os.remove(file_path)
            shutil.move(file_path, os.path.join(subfolder, os.path.basename(file_path)))
        shutil.copy("signac_job_document.json", os.path.join(subfolder, "signac_job_document.json"))
        try:
            # del job.doc["liq_density"]
            del job.doc["gemc_failed"]
        except:
            pass
        try:
            # del job.doc["liq_density"]
            del job.doc["restart_from"]
        except:
            pass

mol_name = "R116"
#To replace
T_in = 210
restart_in = [1]
#Replace with
# T_out = 240
# restart_out = 1
for job in project.find_jobs({"mol_name":mol_name, "T":T_in, "restart": {"$in" : restart_in}}):
    # if "Hvap" in job.doc.keys() and job.doc["Hvap"] == "NaN":
    print("job",  job.id)
    # job_id = list(project.find_jobs({"mol_name":mol_name, "T":T_out, "restart": restart_out}))[0].id
    # print("rest job", job_id)
    # job.doc["restart_from"] = job_id
    # delete_data(job)
    # job.doc["vapboxl"] = 2*job.doc["vapboxl"]/5
    print(job.doc["vapboxl"])
    # try:
    #     # del job.doc["restart_from"]
    #     del job.doc["gemc_failed"]
    # except:
    #     pass


    # job.doc["gemc_failed"] = True
    # job.remove()