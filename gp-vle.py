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

from utils.rXXXX import RXXXXConstants
RXXXX = RXXXXConstants()

############################# QUANTITIES TO EDIT #############################
##############################################################################

cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/generalizedFF/csv/"
in_csv_names = "rXXXX-vle.csv"
# Read file
df_RXXXX=pd.read_csv(csv_path + in_csv_names)
print(df_RXXXX)
#scale all values 
df_vle = prepare_df_vle(df_RXXXX, RXXXX)
#Fit classifier
print(df_vle)
# Create training/test set
param_names = list(RXXXX.param_names) + ["temperature"]
property_names = ["sim_liq_density", "sim_vap_density", "sim_Pvap", "sim_Hvap"]

vle_models = {}
for property_name in property_names:
    # Get train/test
    x_train, y_train, x_test, y_test = shuffle_and_split(
        df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
    )
    # save train/test data
    #df_xtrain = pd.DataFrame(x_train,columns=param_names)
    df_ytrain = pd.DataFrame(y_train,columns=[property_name])
    #df_xtest = pd.DataFrame(x_test,columns=param_names)
    df_ytest = pd.DataFrame(y_test,columns=[property_name])
    #df_xtrain.to_csv('%s_x_train.csv' %property_name, index=False)
    df_ytrain.to_csv('%s_y_train.csv' %property_name, index=False)
    #df_xtest.to_csv('%s_x_test.csv' %property_name, index=False)
    df_ytest.to_csv('%s_y_test.csv' %property_name, index=False)

    # Fit model
    vle_models[property_name] = run_gpflow_scipy(
        x_train,
        y_train,
        gpflow.kernels.RBF(lengthscales=np.ones(RXXXX.n_params + 1)),
    )

# For vapor density replace with Matern52 kernel
property_name = "sim_vap_density"
# Get train/test
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_vle, param_names, property_name, shuffle_seed=gp_shuffle_seed
)
# save train/test data
df_xtrain = pd.DataFrame(x_train,columns=param_names)
df_ytrain = pd.DataFrame(y_train,columns=[property_name])
df_xtest = pd.DataFrame(x_test,columns=param_names)
df_ytest = pd.DataFrame(y_test,columns=[property_name])
df_xtrain.to_csv('x_train.csv' , index=False)
df_ytrain.to_csv('%s_y_train.csv' %property_name, index=False)
df_xtest.to_csv('x_test.csv' , index=False)
df_ytest.to_csv('%s_y_test.csv' %property_name, index=False)
# Fit model
vle_models[property_name] = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.Matern52(lengthscales=np.ones(RXXXX.n_params + 1)),
)

pickle.dump(vle_models, open('vle-gps.pkl', 'wb'))
