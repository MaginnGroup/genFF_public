from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170, r41, r23, r161, r152a, r152, r134, r143, r116
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
from fffit.fffit.utils import values_real_to_scaled, values_scaled_to_real, variances_scaled_to_real, generate_lhs
from fffit.fffit.plot import plot_model_performance, calc_model_mapd
from fffit.fffit.models import run_gpflow_scipy
import os
import gpflow
from gpflow.utilities import print_summary
import warnings
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
# mpl_is_inline = 'inline' in matplotlib.get_backend()


repeats = 3
seed = 2
mode = "train"
#Get obj from a set of parameters
at_class = 11
save_data = False
obj_choice = "ExpVal"
obj_choice_p = "ExpValPrior"
molec_names = ["R14", "R32", "R50", "R125", "R134a", "R143a", "R170"]
setup = opt_atom_types.Problem_Setup(molec_names, at_class, obj_choice)
property_names = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap"]
kernels = ["RBF", "Matern32", "Matern52"] #Options include RBF, Matern32, Matern52, and RQ
# property_names = ["sim_Pvap", "sim_Hvap"]
use_white_kern = True
use_train_lik = False
anisotropic = True
typeMeanFunc = "Linear" #options: Linear, Zero

def get_hyperparams(model):
    """
    Get hyperparameters for the GP model
    
    Parameters:
    model: GP model w/ linear mean fxn, likelihood variance, and anisotropic kernel lengthscale and variance
    """
    hyperparameters = {
        'Mean Fxn A': model.mean_function.A.numpy(),
        'Mean Fxn B': model.mean_function.b.numpy(),
        'kernel_variance': model.kernel.variance.numpy(),
        'kernel_lengthscale': model.kernel.lengthscales.numpy(),
        'likelihood_variance': model.likelihood.variance.numpy()
    }
    return hyperparameters

def check_conditions(hypers):
    """
    Check if the hyperparameters meet the criteria for the lengthscales and kernel variance
    """
    cond1 = (hypers['kernel_lengthscale'] >= 1e-2)
    cond2 = (hypers['kernel_lengthscale'] <= 1e2)
    cond3 = (hypers['kernel_variance'] >= 1e-2)
    cond4 = (hypers['kernel_variance'] <= 10)

    met_all = np.all(cond1 & cond2 & cond3 & cond4)
    met_two = np.all(cond1 & cond2) or np.all(cond3 & cond4)
    met_var = np.all(cond3 & cond4)
    return met_all, met_two, met_var

dir_name = "Results/gp_val_figs/"
os.makedirs(dir_name, exist_ok=True)
pdf = PdfPages(dir_name + "/" + 'gp_models_eval_' + mode + '.pdf')
df_mapd = None
df_mapd_b = None
best_models = []
for molec in list(setup.molec_data_dict.keys()):
    print(molec)
    molec_object = setup.molec_data_dict[molec]
    vle_models = {}
    for prop in property_names:
        print(prop)
        # Get the data and bounds for the property
        if "liq_density" in prop:
            bounds = molec_object.liq_density_bounds
        elif "vap_density" in prop:
            bounds = molec_object.vap_density_bounds
        elif "Pvap" in prop:
            bounds = molec_object.Pvap_bounds
        elif "Hvap" in prop:
            bounds = molec_object.Hvap_bounds
        train_data, test_data = setup.get_train_test_data(molec, property_names)
        x_train, y_train = train_data["x"], train_data[prop]
        x_test, y_test = test_data["x"], test_data[prop]

        if mode == "train":
            x_anal, y_anal = x_train, y_train
        else:
            x_anal, y_anal = x_test, y_test

        # Fit models
        models = {}

        lenscls = np.ones(molec_object.n_params + 1)
        for kernel in kernels:
            gpConfig={'kernel': kernel,
                    'useWhiteKernel':use_white_kern,
                    'trainLikelihood':use_train_lik,
                    'anisotropic':anisotropic,
                    'mean_function':typeMeanFunc}

            models[kernel] = run_gpflow_scipy(x_train, y_train, gpConfig, restarts = repeats)
    
        # Get MAPD for each model + sort by lowest MAPD
        mapd_dict = calc_model_mapd(models, x_anal, y_anal, bounds)
        min_mapd_keys = sorted(mapd_dict, key=mapd_dict.get)

        #Initialize dictionaries to store if the hyperparameters meet the criteria
        cond_dict_all = {}
        cond_dict_half = {}
        cond_dict_var = {}
        model_data = {}

        #Loop over models in order of accuracy
        for i in range(len(mapd_dict)):
            #Get the key of the model with the i lowest MAPD
            min_mapd_key = min_mapd_keys[i]
            #Get the hyperparameters for the model
            hypers = get_hyperparams(models[min_mapd_key])
            #Check conditions for the hyperparameters + save results
            met_all, met_two, met_var = check_conditions(hypers)
            cond_dict_all[min_mapd_key] = met_all
            cond_dict_half[min_mapd_key] = met_two
            cond_dict_var[min_mapd_key] = met_var

            #Save data to a dataframe
            new_row = pd.DataFrame({"Molecule": [molec], 
                                    "Property": [prop], 
                                    "Model": [min_mapd_key], 
                                    "MAPD": [mapd_dict[min_mapd_key]],
                                    "Mean Fxn A": [hypers["Mean Fxn A"].flatten()],
                                    "Mean Fxn B": [hypers["Mean Fxn B"]],
                                    "kernel_variance": [hypers["kernel_variance"]],
                                    "kernel_lengthscale": [hypers["kernel_lengthscale"]],
                                    "likelihood_variance": [hypers["likelihood_variance"]],
                                    "best_model": [False]})
            model_data[min_mapd_key] = new_row
            if df_mapd is None:
                df_mapd = new_row
            else:
                df_mapd = pd.concat([df_mapd, new_row], ignore_index=True)

        #Define default models
        if prop == "sim_vap_density" or prop == "sim_Pvap":
            default, backup = "Matern52", "RBF"
        else:
            default, backup = "RBF", "Matern52"

        #If the default model does not meet the criteria, but the backup does save the backup model
        if cond_dict_var[default] == False and cond_dict_all[backup] == True:
            save_model=models[backup]
        #Otherwise save the default model
        else:
            save_model=models[default]
        vle_models[prop]=save_model

        #Get the best model that meets the criteria
        if any(cond_dict_all.values()):
            min_mapd_key = next(key for key, value in cond_dict_all.items() if value)
        elif any(cond_dict_half.values()):
            warnings.warn("No hyperparameters meet both criteria " + molec + " " + prop, UserWarning)
            min_mapd_key = next(key for key, value in cond_dict_half.items() if value) 
        else:
            warnings.warn("No hyperparameters meet any criteria " + molec + " " + prop, UserWarning)
            min_mapd_key = min_mapd_keys[0]
        best_models.append(models[min_mapd_key])

        #Save which model was the best
        df_mapd.loc[(df_mapd['Molecule'] == molec) & 
                    (df_mapd['Property'] == prop) & 
                    (df_mapd['Model'] == min_mapd_key), 'best_model'] = True
        
        #Save the model predictions to a pdf
        title = molec + " " + prop + " " + mode + " Model Performance"
        fig = plot_model_performance(models, x_anal, y_anal, bounds, title = title)
        if save_data:
            pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()

    #Save models to molec_gp_data/RXX-vlegp/gp-vle.py (ensure original files have moved to go-vle-org.py)
    gp_mod_dir = "molec_gp_data/" + molec + "-vlegp"
    save_path_gp = gp_mod_dir + "/" + 'vle-gps.pkl'
    if save_data:
        if not os.path.exists(save_path_gp) and mode == "test":
            os.makedirs(gp_mod_dir, exist_ok=True)
            print("Saving model to: ", save_path_gp)
            pickle.dump(vle_models, open(save_path_gp, 'wb'))
pdf.close()

#Sort lists by molecule, property, and kernel
df_mapd['Molecule'] = pd.Categorical(df_mapd['Molecule'], categories=molec_names, ordered=True)
df_mapd['Property'] = pd.Categorical(df_mapd['Property'], categories=property_names, ordered=True)
df_mapd['Model'] = pd.Categorical(df_mapd['Model'], categories=kernels, ordered=True)
df_mapd = df_mapd.sort_values(by=['Molecule', 'Property', 'Model'])
df_mapd.to_csv(dir_name + "/" + "gp_models_eval_" +  mode + ".csv", index=False, header=True)
#Shorten df to only include the lowest MAPD for each molecule and property
optimal_df = df_mapd[df_mapd['best_model'] == True]
if save_data:
    optimal_df.to_csv(dir_name + "/" + "gp_models_eval_min_" + mode + ".csv", index=False, header=True)