import signac

# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
for job in project.find_jobs({'atom_type': 8}):
    if job.sp.mol_name == "R152" or job.sp.mol_name == "R134":
        print(job.sp.T, job.id)
        job.remove()
    # if "restart" in job.sp.keys():
    #     if job.sp.restart > 1:
    #         print(job.id)
    #         # del job.doc["use_crit"]
    #         job.remove()