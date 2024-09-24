import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import pickle
import os

from sklearn import svm
import scipy.optimize as optimize

from fffit.utils import (
    shuffle_and_split,
    values_real_to_scaled,
    values_scaled_to_real,
    variances_scaled_to_real,
)

from fffit.plot import (
    plot_model_performance,
    plot_slices_temperature,
    plot_slices_params,
    plot_model_vs_test,
)

from fffit.models import run_gpflow_scipy

sys.path.append("../")
from utils.id_new_samples import (
    prepare_df_vle,
    classify_samples,
    rank_samples,
)


sys.path.append("..")
from utils.molec_class_files import (
    r14,
    r32,
    r50,
    r125,
    r134a,
    r143a,
    r170,
    r41,
    r23,
    r161,
    r152a,
    r152,
    r134,
    r143,
    r116,
)
from utils import atom_type, opt_atom_types

sys.path.remove("..")

r14_class = r14.R14Constants()
r32_class = r32.R32Constants()
r50_class = r50.R50Constants()
r125_class = r125.R125Constants()
r134a_class = r134a.R134aConstants()
r143a_class = r143a.R143aConstants()
r170_class = r170.R170Constants()
r41_class = r41.R41Constants()

train_molec_data = {
    "R14": r14_class,
    "R32": r32_class,
    "R50": r50_class,
    "R170": r170_class,
    "R125": r125_class,
    "R134a": r134a_class,
    "R143a": r143a_class,
    "R41": r41_class,
}

name_ref = {
    "r14": "R14",
    "r32": "R32",
    "r50": "R50",
    "r170": "R170",
    "r125": "R125",
    "r134a": "R134a",
    "r143a": "R143a",
    "r41": "R41",
}

############################# QUANTITIES TO EDIT #############################
##############################################################################

gp_shuffle_seed = 42  # 3945872 #GP seed

##############################################################################
##############################################################################

# Loop over training molecules

for molec in list(name_ref.keys()):
    capital_name = name_ref[molec]
    csv_name = "csv/" + str(molec) + "-vle.csv"
    direc_save_to = "molec_gp_data/" + str(capital_name) + "-vlegp/"
    os.makedirs(direc_save_to, exist_ok=True)
    molec_class = train_molec_data[capital_name]
    # Read file
    df_molec = pd.read_csv(csv_name)
    print(df_molec)
    # scale all values
    df_vle = prepare_df_vle(df_molec, molec_class)
    # Fit classifier
    print(df_vle)
    # Create training/test set
    param_names = list(molec_class.param_names) + ["temperature"]
    property_names = ["sim_liq_density", "sim_Pvap", "sim_Hvap", "sim_vap_density"]

    vle_models = {}
    for property_name in property_names:
        # Get train/test
        x_train, y_train, x_test, y_test = shuffle_and_split(
            df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
        )
        # save train/test data
        if property_name == "sim_vap_density":
            df_xtrain = pd.DataFrame(x_train, columns=param_names)
            df_xtest = pd.DataFrame(x_test, columns=param_names)
            df_xtrain.to_csv(direc_save_to + "x_train.csv", index=False)
            df_xtest.to_csv(direc_save_to + "x_test.csv", index=False)
        df_ytrain = pd.DataFrame(y_train, columns=[property_name])
        df_ytest = pd.DataFrame(y_test, columns=[property_name])
        df_ytrain.to_csv(direc_save_to + "%s_y_train.csv" % property_name, index=False)
        df_ytest.to_csv(direc_save_to + "%s_y_test.csv" % property_name, index=False)
