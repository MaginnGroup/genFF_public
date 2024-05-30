import signac

# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
for job in project:
    if job.sp.mol_name == "R161" and job.sp.T == int(240.0):
        print(job.id)
        # del job.doc["use_crit"]
        # job.remove()