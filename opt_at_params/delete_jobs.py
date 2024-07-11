import signac

# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
for job in project:
    if job.sp.obj_choice == "ExpValPrior" and job.sp.atom_type == 11:
        # print(job.id)
        job.remove()