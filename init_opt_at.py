import signac
import json

project = signac.init_project("opt_at_params")

#Set Initial Parameters
Atom_Type = 11
repeats  = 22 #Repeats for full optimization
repeats_ind = 22 #Repats for individual molecule optimization
seed = 1
save_data = True
training_molecules = list(["R14", "R32", "R50", "R170", "R125", "R134a", "R143a"])

if isinstance(training_molecules, list):
    training_molecules_all = json.dumps(training_molecules)
Objective = "ExpValPrior"

#Create job parameter dict
for i in range(0, repeats):
    sp = {"atom_type": Atom_Type,
        "total_repeats": repeats,
        "repeat_number": i+1,
        "training_molecules": training_molecules_all,
        "num_train_molec": len(training_molecules),
        "obj_choice": Objective,
        "save_data": save_data,
        "seed": seed}
    #Create jobs for exploration bias study
    job = project.open_job(sp).init()

if len(training_molecules) > 1:
    for molec in training_molecules:
        #Make a dumped list of the molecule to pass to the job
        molec_dump = json.dumps(list([molec]))
        for j in range(0, repeats_ind):
            sp = {"atom_type": Atom_Type,
                "total_repeats": repeats_ind,
                "repeat_number": j+1,
                "training_molecules": molec_dump,
                "num_train_molec": 1,
                "obj_choice": Objective,
                "save_data": save_data,
                "seed": seed}
            #Create jobs for exploration bias study
            job = project.open_job(sp).init()