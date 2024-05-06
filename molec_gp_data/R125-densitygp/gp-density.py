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
    prepare_df_density,
    classify_samples,
    rank_samples,
)

from utils.r125 import R125Constants
R125 = R125Constants()

'''from utils.r14 import R14Constants
R14 = R14Constants()
from utils.r50 import R50Constants
R50 = R50Constants()
from utils.r170 import R170Constants
R170 = R170Constants()
from utils.r134a import R134aConstants
R134a = R134aConstants()
from utils.r143a import R143aConstants
R143a = R143aConstants()
from utils.r32 import R32Constants
R32 = R32Constants()
from utils.r125 import R125Constants
R125 = R125Constants()'''

############################# QUANTITIES TO EDIT #############################
##############################################################################

cl_shuffle_seed = 6928457 #classifier
gp_shuffle_seed = 3945872 #GP seed 

liquid_density_thresholdR14 = 1100
liquid_density_thresholdR50 = 200
liquid_density_thresholdR170 = 320
liquid_density_thresholdR134a = 500
liquid_density_thresholdR143a = 500 
liquid_density_thresholdR32 =  500
liquid_density_thresholdR125 = 500

##############################################################################
##############################################################################

csv_path = "/scratch365/nwang2/ff_development/generalizedFF/csv/"
#in_csv_names = [ i + "-density.csv" for i in ['r14','r50','r170','r134a','r143a','r32','r125']]
in_csv_names = "r125-density.csv"
# Read file
'''df_csvs = [pd.read_csv(csv_path + in_csv_name) for in_csv_name in in_csv_names]
df_R14 = df_csvs[0]
df_R50 = df_csvs[1]
df_R170 = df_csvs[2]
df_R134a = df_csvs[3]
df_R143a = df_csvs[4]
df_R32 = df_csvs[5]
df_R125 = df_csvs[6]'''
df_R125=pd.read_csv(csv_path + in_csv_names)

#for refrigerant in [R14,R50,R170,'R134a','R143a','R32','R125']:
#scale all values and separate liquid and vapor
df_all, df_liquid, df_vapor = prepare_df_density(
    df_R125, R125, liquid_density_thresholdR125
)

#Fit classifier

# Create training/test set
param_names = list(R125.param_names) + ["temperature"]
property_name = "is_liquid"
x_train, y_train, x_test, y_test = shuffle_and_split(
df_all, param_names, property_name, shuffle_seed=cl_shuffle_seed
)

# Create and fit classifier
classifier = svm.SVC(kernel="rbf")
classifier.fit(x_train, y_train)
test_score = classifier.score(x_test, y_test)
print(f"Classifer is {test_score*100.0}% accurate on the test set.")
#plot_confusion_matrix(classifier, x_test, y_test)  
#plt.savefig("classifier.pdf")

#Fit LD GP

# Create training/test set
param_names = list(R125.param_names) + ["temperature"]
property_name = "md_density"
x_train, y_train, x_test, y_test = shuffle_and_split(
    df_liquid, param_names, property_name, shuffle_seed=gp_shuffle_seed
)

# Fit model
model = run_gpflow_scipy(
    x_train,
    y_train,
    gpflow.kernels.RBF(lengthscales=np.ones(R125.n_params + 1)),
)

#means_scaled, vars_scaled = model.predict_f(x_test)
#print(means_scaled,vars_scaled)

pickle.dump(model, open('MD-densitygp.pkl', 'wb'))
