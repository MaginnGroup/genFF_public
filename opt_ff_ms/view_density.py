import os
import sys
import numpy as np
import subprocess
import signac

def main():
    # Ensure the correct number of arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py mol_name")
        sys.exit(1)

    # Extract the statepoint value from the command line argument
    mol_name = sys.argv[1]


    # Locate the signac project
    project = signac.get_project()

    # Find all jobs with the specified statepoint value
    # if sys.argv[2] != None and sys.argv[3] !=None:
    #     jobs = project.find_jobs({"mol_name": mol_name, "T": float(sys.argv[3]), "atom_type": int(sys.argv[2])})
    # else:
    jobs = project.find_jobs({"mol_name": mol_name})

    # Iterate over the matching jobs
    for job in jobs:
        # Construct the command using the job ID and statepoint value
        
        # if os.path.exists(f"workspace/{job.id}/gemc.eq.rst.001.out.box1.prp") and os.path.exists(f"workspace/{job.id}/gemc.eq.rst.001.out.box2.prp"):
        #     command = (
        #         f"xmgrace -block workspace/{job.id}/gemc.eq.rst.001.out.box1.prp -bxy 1:6 "
        #         f"-block workspace/{job.id}/gemc.eq.rst.001.out.box2.prp -bxy 1:6"
        #     )
        # else:
        command = (
            f"xmgrace -block workspace/{job.id}/prod.out.box1.prp -bxy 1:5 -block workspace/{job.id}/prod.out.box2.prp -bxy 1:5"
            # f"-block workspace/{job.id}/prod.out.box2.prp -bxy 1:6"
        )
        
        try:
            file1 = f"workspace/{job.id}/prod.out.box1.prp"
            file2 = f"workspace/{job.id}/prod.out.box2.prp"
            if not os.path.exists(file1) or not os.path.exists(file2):
                has_negative = False
            else:
                data1 = np.loadtxt(file1, usecols=(0, 4))  # Columns 1 (0-indexed) and 5 (4-indexed)
                data2 = np.loadtxt(file2, usecols=(0, 4))  # Columns 1 (0-indexed) and 5 (4-indexed)

                differences = data1[:, 1] - data2[:, 1]  # Subtract column 5 from both files

                # Check if any value in the result is negative
                has_negative = np.any(differences < 0)
            # Execute the command
            
            if has_negative:
                print("ID", job.id, "AT", job.sp.atom_type, "T", job.sp.T, "restart", job.sp.restart)
                subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command for job {job.id}: {e}")


if __name__ == "__main__":
    main()
