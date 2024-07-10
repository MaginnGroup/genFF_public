import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary
import warnings
import copy

def buildGP(x_train, y_train, kernel, mean_function="linear", fmt="notebook", seed = None, lenscls = None):

    """Create and train a GPFlow model

    Parameters
    ----------
    x_train : np.ndarray, shape=(n_samples, n_parameters)
        The x training data
    y_train : np.ndarray, shape=(n_samples, 1)
        The y training data
    kernel : string
        Kernel to use for the GP model
    mean_function: string or None, default = "linear"
        Type of mean function for the GP model
        Options are "linear", or None
    fmt : string, optional, default="notebook"
        The formatting type for the GPFlow print_summary
    """
    kernel_use = copy.deepcopy(kernel)
    if lenscls is not None:
        kernel_use.lengthscales.assign(lenscls)
    # print_summary(kernel)
    if seed != None:
        np.random.seed(seed)
        tf.compat.v1.get_default_graph()
        tf.compat.v1.set_random_seed(seed)
        tf.random.set_seed(seed)
    if mean_function is not None:
        if mean_function == "linear":
            mean_function = gpflow.mean_functions.Linear(
                A=np.zeros(x_train.shape[1]).reshape(-1, 1)
            )
        elif mean_function.lower() == "none":
            mean_function = None
        else:
            raise ValueError(
                "Only supported mean functions are 'linear' and 'none'"
            )

    # Create the model
    model = gpflow.models.GPR(
        data=(x_train, y_train.reshape(-1, 1)),
        kernel=kernel_use,
        mean_function=mean_function
    )

    # Optimize model with scipy
    optimizer = gpflow.optimizers.Scipy()
    aux = optimizer.minimize(model.training_loss, model.trainable_variables)

    return model, aux

def init_hyper_parameters(count_fix, args):
    # kernel = args[2]
    # if count_fix != 0:
        # Randomize the lengthscales after 1st opt
        # x_train = args[0]
        # kernel.variance.assign((1.0))
        # lenscls = np.random.uniform(low=1e-2, high=5.0, size=x_train.shape[1])
        # kernel.lengthscales.assign(lenscls)
    lenscls = np.random.uniform(low=1e-2, high=10.0, size=args[0].shape[1]) if count_fix != 0 else None
    return lenscls

def fit_GP(count_fix, retrain_GP, args):
    """
    Fit a GP and fix Cholesky decomposition failure and optimization failure by random initialization

    Parameters
    ----------
    count_fix: int, the number of times the GP has been retrained

    Returns
    --------
    fit_successed: bool, whether the GP was fit successfully
    model: instance of gpflow.models.GPR, the trained GP model
    count_fix: int, the number of times the GP has been retrained
    """
    #Randomize Seed
    np.random.seed(count_fix+1)
    #Initialize fit_sucess as true
    fit_successed = True   
    #Get hyperparam guess list
    lenscls = init_hyper_parameters(count_fix, args)
    # print(args[2])
    # print_summary(args[2])
    # print("count_fix: ", count_fix)
    try:
        #Make model and optimizer and get results
        model, res = buildGP(*args, lenscls)

        #If result isn't successful, remake and retrain model w/ different hyperparameters
        if not(res.success):
            if count_fix < retrain_GP:
                # print('model failed to optimize, fix it by random initialization.')
                count_fix += 1 
                fit_successed,model,count_fix = fit_GP(count_fix, retrain_GP, args)
            else:
                fit_successed = False
    #If an error is thrown becauuse of bad hyperparameters, reoptimize them
    except tf.errors.InvalidArgumentError as e:
        if count_fix < retrain_GP:
            # print('bad initial hyperparameters, fix it by random initialization.')
            count_fix += 1
            fit_successed,model,count_fix = fit_GP(count_fix, retrain_GP, args)
        else:
            fit_successed = False

    if fit_successed:
        kern_var = model.kernel.variance.numpy()
        kern_lensc = model.kernel.lengthscales.numpy()
        #Check that all params are withing 1e-3 and 1e3
        all_params = [kern_var, kern_lensc]
        good_params = np.all((kern_lensc >= 1e-2) & (kern_lensc <= 1e3) & (kern_var >= 1e-2) & (kern_var <= 1e3))
        #If the kernel parameters are too large or too small, reoptimize them
        if not good_params:
            if count_fix < retrain_GP:
                # print('bad final hyperparameters, fix it by random initialization.')
                count_fix = count_fix + 1 
                fit_successed, model, count_fix = fit_GP(count_fix, retrain_GP, args)
            else:
                fit_successed = False
        # #Otherwise check fit. Not using check fit because using the mean isn't a good way to check fit in these cases
        # else:    
        #     X_Train = args[0]
        #     Y_Train = args[1]
        #     min_values = np.min(X_Train, axis=0)
        #     max_values = np.max(X_Train, axis=0)
        #     # Generate random data points within these ranges for each feature
        #     num_samples = 100
        #     random_values = np.random.uniform(0, 1, size=(num_samples, X_Train.shape[1]))
        #     xtest = min_values + random_values * (max_values - min_values)
        #     mean, var = model.predict_y(xtest)
        #     mean = mean.numpy()
        #     var = var.numpy()

        #     y_mean = np.mean(Y_Train)
        #     mean_mean = np.mean(mean)
        #     y_max = np.max(Y_Train)
        #     mean_max = np.max(mean)
        #     y_min = np.abs(np.min(Y_Train))
        #     mean_min = np.abs(np.min(mean))
            
        #     #If fit is bad, reoptimize
        #     if y_mean > 1e-7 and (mean_max > y_max or mean_min < y_min):
        #         if abs(((mean_mean-y_mean)/y_mean)) > 0.5 or np.isclose(mean_mean, 0.0, 1e-7):
        #             print(abs(((mean_mean-y_mean)/y_mean)))
        #             if count_fix < retrain_GP:
        #                 print('bad solution, fix it by random initialization.')
        #                 count_fix = count_fix + 1 
        #                 fit_successed, model, count_fix = fit_GP(count_fix, retrain_GP, args)

    return fit_successed, model, count_fix


def run_gpflow_scipy(x_train, y_train, kernel, mean_function="linear", fmt="notebook", seed = None, restarts = 1):
    """Create and train a GPFlow model

    Parameters
    ----------
    x_train : np.ndarray, shape=(n_samples, n_parameters)
        The x training data
    y_train : np.ndarray, shape=(n_samples, 1)
        The y training data
    kernel : string
        Kernel to use for the GP model
    mean_function: string or None, default = "linear"
        Type of mean function for the GP model
        Options are "linear", or None
    fmt : string, optional, default="notebook"
        The formatting type for the GPFlow print_summary
    """
    # Train the model multiple times and keep track of the model with the lowest minimum training loss
    best_minimum_loss = float('inf')
    best_model = None
    first_model = None
    best_model_success = None
    args = [x_train, y_train, kernel, mean_function, fmt, seed]
    #Initialize number of counters
    count_fix_tot = 0
    
    #Initialize everything with vanilla parameters
    first_mod_succ, first_model, count_fix = fit_GP(0, 0, args)
    best_minimum_loss = first_model.training_loss().numpy()
    # print('First model training loss: ', best_minimum_loss)
    best_model = first_model

    #While you still have retrains left
    while count_fix_tot <= restarts:
        #Create and fit the model
        fit_successed, gp_model, count_fix = fit_GP(count_fix_tot, restarts, args)
        #The new counter total is the number of counters used + 1
        count_fix_tot += count_fix + 1
        
        if fit_successed:
            # Compute the training loss of the model
            training_loss = gp_model.training_loss().numpy()
            # print("training loss", training_loss)
            # Check if this model has the best minimum training loss
            if training_loss < best_minimum_loss:
                #If the 1st model succeeds it will be a backup plan
                best_minimum_loss = training_loss
                best_model = gp_model
                best_model_success = True
        #or we have no good models
        elif count_fix_tot >= restarts:
            #If we have no good models, use the first model
            if best_model is None:
                best_model_success = False
                best_model = first_model
                warnings.warn('GP optimizer failed to converge.')

    # gpflow.utilities.print_summary(best_model)
    return best_model

# def run_gpflow_scipy(x_train, y_train, kernel, mean_function="linear", fmt="notebook", seed= None, restarts = 1):
#     """Create and train a GPFlow model

#     Parameters
#     ----------
#     x_train : np.ndarray, shape=(n_samples, n_parameters)
#         The x training data
#     y_train : np.ndarray, shape=(n_samples, 1)
#         The y training data
#     kernel : string
#         Kernel to use for the GP model
#     mean_function: string or None, default = "linear"
#         Type of mean function for the GP model
#         Options are "linear", or None
#     fmt : string, optional, default="notebook"
#         The formatting type for the GPFlow print_summary
#     """
#     if seed != None:
#         np.random.seed(seed)
#         tf.compat.v1.get_default_graph()
#         tf.compat.v1.set_random_seed(seed)
#         tf.random.set_seed(seed)

#     if mean_function is not None:
#         if mean_function == "linear":
#             mean_function = gpflow.mean_functions.Linear(
#                 A=np.zeros(x_train.shape[1]).reshape(-1, 1)
#             )
#         elif mean_function.lower() == "none":
#             mean_function = None
#         else:
#             raise ValueError(
#                 "Only supported mean functions are 'linear' and 'none'"
#             )

#     # Create the model
#     model = gpflow.models.GPR(
#         data=(x_train, y_train.reshape(-1, 1)),
#         kernel=kernel,
#         mean_function=mean_function
#     )

#     # Print initial values
#     # print_summary(model, fmt=fmt)

#     # Optimize model with scipy
#     optimizer = gpflow.optimizers.Scipy()
#     optimizer.minimize(model.training_loss, model.trainable_variables)

#     # Print the optimized values
#     # print_summary(model, fmt="notebook")

#     # Return the model
#     return model