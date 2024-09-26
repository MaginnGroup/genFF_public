import signac
import json
import sys

# sys.path.append("..")
# from utils.molec_class_files import (
#     r14,
#     r32,
#     r50,
#     r125,
#     r134a,
#     r143a,
#     r170,
#     r41,
#     r23,
#     r161,
#     r152a,
#     r152,
#     r134,
#     r143,
#     r116,
# )
# from utils import atom_type, opt_atom_types

# sys.path.remove("..")

# Load the project
project = signac.get_project()

# # Iterate through all jobs in the project
# for job in project:
#     # Define driver class
#     training_molecules = job.sp.training_molecules
#     training_molecules = list(json.loads(training_molecules))
#     driver = opt_atom_types.Problem_Setup(
#         training_molecules, job.sp.atom_type, job.sp.obj_choice
#     )
#     pareto_save = driver.use_dir_name / "pareto_info.csv"
#     # Check if pareto info exists
#     if pareto_save.exists():
#         job.doc["pareto_info"] = True
for job in project:
    if job.sp.obj_choice == "ExpVal" and job.sp.atom_type == 11:
        # print(job.id)
        job.remove()
