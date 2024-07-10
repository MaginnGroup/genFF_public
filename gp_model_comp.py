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
# property_names = ["sim_Pvap"]


dir_name = "Results/gp_val_figs/"
os.makedirs(dir_name, exist_ok=True)
pdf = PdfPages(dir_name + "/" + 'gp_models_eval_' + mode + '.pdf')
df_mapd = None
df_mapd_b = None
best_models = []
for molec in list(setup.molec_data_dict.keys()):
    print(molec)
    molec_object = setup.molec_data_dict[molec]
    for prop in property_names:
        print(prop)
        # Get the data
        if "liq_density" in prop:
            bounds = molec_object.liq_density_bounds
        elif "vap_density" in prop:
            bounds = molec_object.vap_density_bounds
        elif "Pvap" in prop:
            bounds = molec_object.Pvap_bounds
        elif "Hvap" in prop:
            bounds = molec_object.Hvap_bounds

        train_data, test_data = setup.get_train_test_data(molec, property_names)
        x_train = train_data["x"]
        y_train = train_data[prop]
        x_test = test_data["x"]
        y_test = test_data[prop]

        # Fit model
        models = {}
        models["RBF"] = run_gpflow_scipy(
            x_train,
            y_train,
            gpflow.kernels.RBF(lengthscales=np.ones(molec_object.n_params + 1)),
            seed = seed,
            restarts= repeats
        )
        models["Matern32"] = run_gpflow_scipy(
            x_train,
            y_train,
            gpflow.kernels.Matern32(lengthscales=np.ones(molec_object.n_params + 1)),
            seed = seed,
            restarts= repeats
        )

        models["Matern52"] = run_gpflow_scipy(
            x_train,
            y_train,
            gpflow.kernels.Matern52(lengthscales=np.ones(molec_object.n_params + 1)),
            seed = seed,
            restarts= repeats
        )

        # # Plot model performance on train and test points
        if mode == "train":
            mapd_dict = calc_model_mapd(models, x_train, y_train, bounds)
        else:
            mapd_dict = calc_model_mapd(models, x_test, y_test, bounds)
        print(mapd_dict)
        found_best = False
        #Get best model with good hyperparameter values
        for i in range(len(mapd_dict)):
            #Get the key of the model with the i lowest MAPD
            min_mapd_key = sorted(mapd_dict, key=mapd_dict.get)[i]
            hyperparameters = {
        'Mean Fxn A': models[min_mapd_key].mean_function.A.numpy(),
        'Mean Fxn B': models[min_mapd_key].mean_function.b.numpy(),
        'kernel_variance': models[min_mapd_key].kernel.variance.numpy(),
        'kernel_lengthscale': models[min_mapd_key].kernel.lengthscales.numpy(),
        'likelihood_variance': models[min_mapd_key].likelihood.variance.numpy()
    }       
            cond1 = (hyperparameters['kernel_lengthscale'] >= 1e-2)
            cond2 = (hyperparameters['kernel_lengthscale'] <= 1e2)
            cond3 = (hyperparameters['kernel_variance'] >= 1e-2)
            cond4 = (hyperparameters['kernel_variance'] <= 10)
            #If the lengthscales are good values, then save the model
            if np.all(cond1 & cond2 & cond3 & cond4):
                best_models.append(models[min_mapd_key])
                found_best = True
                break

        if not found_best:
            for i in range(len(mapd_dict)):
                #Get the key of the model with the i lowest MAPD
                min_mapd_key = sorted(mapd_dict, key=mapd_dict.get)[i]
                hyperparameters = {
            'Mean Fxn A': models[min_mapd_key].mean_function.A.numpy(),
            'Mean Fxn B': models[min_mapd_key].mean_function.b.numpy(),
            'kernel_variance': models[min_mapd_key].kernel.variance.numpy(),
            'kernel_lengthscale': models[min_mapd_key].kernel.lengthscales.numpy(),
            'likelihood_variance': models[min_mapd_key].likelihood.variance.numpy()
        }       
                cond1 = (hyperparameters['kernel_lengthscale'] >= 1e-2)
                cond2 = (hyperparameters['kernel_lengthscale'] <= 1e2)
                cond3 = (hyperparameters['kernel_variance'] >= 1e-2)
                cond4 = (hyperparameters['kernel_variance'] <= 10)
                #If the lengthscales are good values, then save the model
                if np.all(cond3 & cond4) or np.all(cond1 & cond2):
                    warnings.warn("No hyperparameters meet both criteria " + molec + " " + prop, UserWarning)
                    best_models.append(models[min_mapd_key])
                    found_best = True
                    break

        if not found_best:
            warnings.warn("No hyperparam sets meet any criteria", UserWarning)
            min_mapd_key = sorted(mapd_dict, key=mapd_dict.get)[0]
            best_models.append(models[min_mapd_key])
            hyperparameters =  {
        'Mean Fxn A': models[min_mapd_key].mean_function.A.numpy(),
        'Mean Fxn B': models[min_mapd_key].mean_function.b.numpy(),
        'kernel_variance': models[min_mapd_key].kernel.variance.numpy(),
        'kernel_lengthscale': models[min_mapd_key].kernel.lengthscales.numpy(),
        'likelihood_variance': models[min_mapd_key].likelihood.variance.numpy()
    }                      

        print("Lowest MAPD valid model hypers: ", hyperparameters)

        for mod_key in models.keys():
            new_row = pd.DataFrame({"Molecule": [molec], 
                                    "Property": [prop], 
                                    "Model": [mod_key], 
                                    "MAPD": [mapd_dict[mod_key]],
                                    "Mean Fxn A": [models[mod_key].mean_function.A.numpy().flatten()],
                                    "Mean Fxn B": [models[mod_key].mean_function.b.numpy()],
                                    "kernel_variance": [models[mod_key].kernel.variance.numpy()],
                                    "kernel_lengthscale": [models[mod_key].kernel.lengthscales.numpy()],
                                    "likelihood_variance": [models[mod_key].likelihood.variance.numpy()]})
            if df_mapd is None:
                df_mapd = new_row
            else:
                df_mapd = pd.concat([df_mapd, new_row], ignore_index=True)

            if mod_key == min_mapd_key:
                new_b_row = pd.DataFrame({"Molecule": [molec], 
                                    "Property": [prop], 
                                    "Model": [mod_key], 
                                    "MAPD": [mapd_dict[mod_key]],
                                    "Mean Fxn A": [models[mod_key].mean_function.A.numpy().flatten()],
                                    "Mean Fxn B": [models[mod_key].mean_function.b.numpy()],
                                    "kernel_variance": [models[mod_key].kernel.variance.numpy()],
                                    "kernel_lengthscale": [models[mod_key].kernel.lengthscales.numpy()],
                                    "likelihood_variance": [models[mod_key].likelihood.variance.numpy()]})
                best_models.append(models[mod_key])
                if df_mapd_b is None:
                    df_mapd_b = new_b_row
                else:
                    df_mapd_b = pd.concat([df_mapd_b, new_b_row], ignore_index=True)

        title = molec + " " + prop + " " + mode + " Model Performance"
        if mode == "train":
            pdf.savefig(plot_model_performance(models, x_train, y_train, bounds, title = title))
            # plt.close()
        else:
            pdf.savefig(plot_model_performance(models, x_test, y_test, bounds, title = title))
        # plt.close()
pdf.close()

df_mapd.to_csv(dir_name + "/" + "gp_models_eval_" +  mode + ".csv", index=False, header=True)
#Shorten df to only include the lowest MAPD for each molecule and property
df_mapd_b.to_csv(dir_name + "/" + "gp_models_eval_min_" + mode + ".csv", index=False, header=True)