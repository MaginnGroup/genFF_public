from utils import opt_atom_types

save_data = True  # Data to save
obj_choice = "ExpVal"  # Objective to consider
at_number = 11  # atom type to consider
seed = 1  # Seed to use
molec_names = [
    "R14",
    "R32",
    "R50",
    "R170",
    "R125",
    "R134a",
    "R143a",
    "R41",
]  # Training data to consider

# Create visualization object
visual = opt_atom_types.Vis_Results(molec_names, at_number, seed, obj_choice)
visual.check_GPs()
visual.get_uncertainty_data()
