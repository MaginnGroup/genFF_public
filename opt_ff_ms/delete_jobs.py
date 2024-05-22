import signac

# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
for job in project:
    if job.sp.mol_name == "R143":
        print(job.id)
        job.remove()