import signac

# Load the project
project = signac.get_project()

# Iterate through all jobs in the project
for i,job in enumerate(project.find_jobs({"atom_type" : 11, "obj_choice" : "ExpVal"})):
    if i == 0:
        print(job.id)
        print(job.sp)
    # Check if the job has the attribute "sp" and if the atom_type is 11
        for other_job in project.find_jobs({"atom_type" : job.sp.atom_type, "obj_choice" : job.sp.obj_choice, "training_molecules":job.sp.training_molecules}):
            print(other_job.id)
# for job in project:
#     if job.sp.obj_choice == "ExpVal" and job.sp.atom_type == 11:
#         # print(job.id)
#         job.remove()