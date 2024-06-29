import signac

# Open the project
project = signac.get_project()

# Define the specific statepoint you want to filter jobs by
statepoint_to_find = {"obj_choice": "ExpValPrior", "repeat_number": 1}

# Use find_jobs to get all jobs with the specific statepoint
jobs = project.find_jobs(statepoint_to_find)

# Iterate over the jobs and print the job ID along with another statepoint value
for job in jobs:
    job_id = job.id
    another_statepoint_value = job.sp['training_molecules']  # Replace 'b' with the key of the statepoint you want to list
    print(f"Job ID: {job_id}, b: {another_statepoint_value}")
