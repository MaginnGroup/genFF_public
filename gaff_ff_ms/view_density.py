import os
import sys
import subprocess
import signac
import numpy as np

def main():
    # Ensure the correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py mol_name")
        sys.exit(1)

    # Extract the statepoint value from the command line argument
    mol_name = sys.argv[1]
    # print(sys.argv)
    # Locate the signac project
    project = signac.get_project()

    # Find all jobs with the specified statepoint value
    # print(len(sys.argv))
    if len(sys.argv) > 2:
        jobs = project.find_jobs({"mol_name": mol_name, "T": float(sys.argv[2])})
    else:
        jobs = project.find_jobs({"mol_name": mol_name})

    # Iterate over the matching jobs
    for job in jobs:
        # Construct the command using the job ID and statepoint value
        
        # if os.path.exists(f"workspace/{job.id}/gemc.eq.rst.001.out.box1.prp") and os.path.exists(f"workspace/{job.id}/gemc.eq.rst.001.out.box2.prp"):
        #     command = (
        #         f"xmgrace -block workspace/{job.id}/gemc.eq.rst.001.out.box1.prp -bxy 1:5 "
        #         f"-block workspace/{job.id}/gemc.eq.rst.001.out.box2.prp -bxy 1:5"
        #     )
        # else:
        command = (
            f"xmgrace -block workspace/{job.id}/prod.out.box1.prp -bxy 1:5 "
            f"-block workspace/{job.id}/prod.out.box2.prp -bxy 1:5"
            )

        # Execute the command
        try:
            file1 = f"workspace/{job.id}/prod.out.box1.prp"
            file2 = f"workspace/{job.id}/prod.out.box2.prp"
            if not os.path.exists(file1) or not os.path.exists(file2):
                has_negative = False
            else:
                data1 = np.loadtxt(file1, usecols=(0, 4))  # Columns 1 (0-indexed) and 5 (4-indexed)
                data2 = np.loadtxt(file2, usecols=(0, 4))  # Columns 1 (0-indexed) and 5 (4-indexed)

                if np.average(data2[:, 1]) < 30:  
                    print("ID", job.id, "T", job.sp.T, "restart", job.sp.restart)
                    subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for job {job.id}: {e}")

if __name__ == "__main__":
    main()
