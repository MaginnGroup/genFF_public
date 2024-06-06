import os
import sys
import subprocess
import signac

def main():
    # Ensure the correct number of arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py mol_name")
        sys.exit(1)

    # Extract the statepoint value from the command line argument
    mol_name = sys.argv[1]

    # Locate the signac project
    project = signac.get_project()

    # Find all jobs with the specified statepoint value
    jobs = project.find_jobs({"mol_name": mol_name})

    # Iterate over the matching jobs
    for job in jobs:
        # Construct the command using the job ID and statepoint value
        print(job.id, job.sp.T)
        command = (
            f"xmgrace -block workspace/{job.id}/gemc.eq.rst.001.out.box1.prp -bxy 1:6 "
            f"-block workspace/{job.id}/gemc.eq.rst.001.out.box2.prp -bxy 1:6"
        )

        # Execute the command
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for job {job.id}: {e}")

if __name__ == "__main__":
    main()
