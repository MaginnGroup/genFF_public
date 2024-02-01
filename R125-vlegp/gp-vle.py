import sys
import gpflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import pickle

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

from utils.r125 import R125Constants
R125 = R125Constants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/generalizedFF/csv/"
in_csv_names = "r125-vle.csv"
# Read file
df_R125=pd.read_csv(csv_path + in_csv_names)
print(df_R125)
#scale all values 
df_vle = prepare_df_vle(df_R125, R125)
#Fit classifier
print(df_vle)
# Create training/test set
param_names = list(R125.param_names) + ["temperature"]
property_names = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap"]

vle_models = {}
for property_name in property_names:
    # Get train/test
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
    )

    # Fit model
    vle_models[property_name] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(R125.n_params + 1)),
    )

# For vapor density replace with Matern52 kernel
property_name = "sim_vap_density"
# Get train/test
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
)
# Fit model
vle_models[property_name] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(R125.n_params + 1)),
)

pickle.dump(vle_models, open('vle-gps.pkl', 'wb'))
