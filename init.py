import signac
import json

project = signac.init_project()

#Set Initial Parameters
Atom_Type = 9
repeats  = 1
repeats_ind = 1
seed = 1
save_data = True
training_molecules = list(["R14", "R32", "R50", "R170", "R125", "R134a", "R143a"])

if isinstance(training_molecules, list):
    training_molecules_all = json.dumps(training_molecules)
Objective = "ExpVal"


#Create job parameter dict
for i in range(0, repeats):
    sp = {"atom_type": Atom_Type,
        "total_repeats": repeats,
        "repeat_number": i+1,
        "training_molecules": training_molecules_all,
        "obj_choice": Objective,
        "save_data": save_data,
        "seed": seed}
    #Create jobs for exploration bias study
    job = project.open_job(sp).init()

if len(training_molecules) > 1:
    for molec in training_molecules:
        for j in range(0, repeats_ind):
            sp = {"atom_type": Atom_Type,
                "total_repeats": repeats_ind,
                "repeat_number": j+1,
                "training_molecules": molec,
                "obj_choice": Objective,
                "save_data": save_data,
                "seed": seed}
            #Create jobs for exploration bias study
            job = project.open_job(sp).init()