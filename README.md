# Machine Learning to Optimize Transferable Hydrofluorocarbon Refrigerant Force Fields
Authors: Montana N. Carlozo, Ke Wang, and Alexander W. Dowling
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**genFF_public** is a repository used to calibrate a transferable FF for one- and two-carbon single-bonded refrigerants with elements of C, F, and H given experimental data. The key feature of this work is using machine learning (ML) tools in the form of Gaussian processes (GPs) and estimability analysis techniques to smartly design atom type schemes for tranferable FFs and optimize their LJ parameters. This work features the comparison of four atom typing schemes designed and optimized with ML and GAFF.

**Note**: For all files in this repository, AT-4 (main text) corresponds to AT-1 (repository files). Similarly, AT-6a corresponds to AT-2, AT-6b corresponds to AT-6, and AT-8 corresponds to AT-8.

## Citation
Please cite as:

Montana N. Carlozo, Ning Wang, Alexander W. Dowling, Edward J. Maginn “Machine Learning to Optimize Transferable Hydrofluorocarbon Refrigerant Force Fields”, 2025

## Available Data

### Repository Organization
The repository is organized as follows: <br />
genFF_public/ is the top level directory. <br />
It contains the following: <br />
1. .gitignore prevents large files from the signac workflow and plots from being tracked by git and prevents tracking of other unimportant files. <br />
2. hfcs-fffit.yaml is the conda environment for use with this work. <br />
3. gen-gp-vle.py is the code used to create the GP models in this work from the data in our previous study. <br />
4. AT-results.xlsx is an excel file of results containing the numerical results of the eigen-decomposition of the FIM and estimability analysis as well as the final LJ parameters for each atom type scheme. <br />
5. init_gaff_ms.py is the initialization file for molecular simulations for the gaff LJ parameters. <br />
6. init_opt_at.py is the initialization file for atom type optimization. <br />
7. init_optff_ms.py is the initialization file for molecular simulations for the validation of LJ parameters for atom type schemes. <br />
8. molecule_exp_unc_data.csv is a csv of the weights and uncertainties used in atom type (AT) optimization. <br />
9. param-comp.xlsx is an excel file comparing the different LJ parameters used for each AT scheme for each molecule. <br />
10. post_analysis_ms.py is the script used to gather validation including the data for Table 4. Also generates Figures 3, 4, 5, and 6 and the files h-p-vap.pdf and vle.pdf. <br />
11. post_analysis_opt.py is the script used to optimize the transferable FF parameters. Generates the data for Table 7. <br />
12. rcc_opt_at_analysis.py is the script used to perform the estimability analysis and eigen-decomposition of the FIM. Generates the data for Table 6. <br />

Directories gaff_ff_ms/, opt_at_params/, and opt_ff_ms/ are initially created via init_gaff_ms.py, init_opt_at.py, and init_optff_ms.py in the top directory through signac. <br /> 
Each contains the following files/subdirectories: <br />
1. project_gaff_ms.py, project_opt_at.py, or project_optff_ms.py; The script for running the workflow using signac. <br />
1. templates/ are the templates required to run this workflow in signac on the crc. <br />
2. workspace/ will appear to save all raw results generated during the workflow after running init_gpbo*.py. This file is not tracked by git due to its size. the workspace/ folder for this study can be downloaded on Google Drive (see section 'Workflow Files and Results') <br />
3. signac_project_document.json will also appear to track the status of jobs in the signac workflow <br />

Directory csv/ contains data used to train the GP models. <br />
It contains the following files: <br />
1. rXX-density.csv; The MD density data. <br />
2. rXX-vle.csv; The GEMC data which is used to train the GP models. <br />

Directory example_mcf_files/ contains sample .mcf files for all models and HFCs evaluated in this work. <br />
It contains the following files/subdirectories: <br />
1. AT-Y/RXX-species1.mcf are the sample .mcf files for each FF model and refrigerant. <br />

Directory fffit/fffit is a package which contains some critical functions for running the workflow. <br />
It contains the following files/subdirectories: <br />
1. tests/ contains the tests for the functions in fffit/fffit. <br />
2. __init__.py intializes the package. <br />
3. models.py contains functions related to building GP models. <br />
4. pareto.py contains functions related to locating pareto-optimal parameter sets. <br />
5. plot.py contains some functions for plotting. <br />
6. signac.py contains functions related to parsing data from signac workspaces. <br />
7. utils.py contains utility functions necessary for this package. <br />

Directory molec_gp_data/ contains the GPs and training/testing data for each refrigerant. <br />
It contains the following files/subdirectories: <br />
1. RXX-vlegp are the subdirectories for each refrigerant. <br />
2. RXX-vlegp/sim_PROP_y_train.csv are the output training data for each property. <br />
3. RXX-vlegp/sim_PROP_y_test.csv are the output testing data for each property. <br />
4. RXX-vlegp/x_train.csv are the input training data for all properties. <br />
5. RXX-vlegp/x_test.csv are the input testing data for all properties. <br />
6. RXX-vlegp/vle-gps.pkl are the pickled GP models for each property. <br />

The pymser/ directory is a clone of the pymser repository. Refer to their [GitHub Page](https://github.com/IBM/pymser) for more information. <br />

The utils/ directory consists of the functions and files required for generalized FF optimization. <br />
It contains the following files/subdirectories: <br />
1. molec_class_files/rXX.py are files containing class objects with the experimental data and relevant information for each refrigerant.  <br />
2. __init__.py intializes the package. <br />
3. analyze_ms.py contains all functions necessary to analyze the molecular simulation data. <br />
4. atom_type.py contains classes for each data-informed atom type studied in this work. <br />
5. opt_atom_types.py contains classes and functions relevant for optimizing transferable FF parameters. <br />

Running the analysis will cause results directories to appear in genFF_public/ with relevant human readable data and plots. Subdirectories further categorize the results by transferable FF. <br />
1. Results/ shows data where we analyze the results from transferable FF parameter optimization. <br />
2. Results_MS/ shows data where we analyze the results of the molecular simulations used to validate our transferable FFs and compare them with GAFF. <br />
3. Results_gp/ shows data where we analyze the best results based on how efficiently the GP predicted SSE was optimized. <br />

We note that this repository is based on the branch ``public`` in the ``dowlinglab/generalizedFF`` repository, which is private.

### Workflow Files and Results
All workflow iterations were performed inside either ``genFF_public/gaff_ff_ms/``, ``genFF_public/opt_at_params/``, or , ``genFF_public/opt_ff_ms/`` where it exists.
Each iteration was managed with ``signac-flow``. Inside ``gaff_ff_ms``, ``opt_at_params``, or ``opt_ff_ms/`` you will find all the necessary files to
run the workflow. Note that you may not get the exact same simulation results due to differences in software versions, random seeds, etc.

### Workflow Code
All of the scripts for running the workflow are provided in this repository. post_analysis_ms.py, post_analysis_opt.py, and rcc_opt_at_analysis.py are the scripts used to perform data analysis. 

### Figures
All scripts required to generate the primary figures in the
manuscript and SI are reported under ``genFF_public/post_analysis_ms.py``. When running analysis scripts, these figures are saved under ``Results_MS_/at_yy/RXX``

## Installation
To run this software, you must have access to all packages in the hfcs-fffit environment (hfcs-fffit.yaml) which can be installed using the instructions in the next section.
<!-- Installation: Provide instructions on how to install and set up the project, including any dependencies that need to be installed. -->
This package has a number of requirements that can be installed in
different ways. We recommend using a conda environment to manage
most of the installation and dependencies. However, some items will
need to be installed from source or pip. <br />

Running the simulations will also require an installation of pymser.
This can be installed separately (see installation instructions
[here](https://github.com/IBM/pymser) ). <br />

An example of the procedure is provided below:

    # Install pip/conda available dependencies
    # with a new conda environment named gpbo-emul
    conda env create -f hfcs-fffit.yaml
    conda activate hfcs-fffit
    pip install pymser

## Usage

**NOTE**: We use Signac and [signac flow](https://signac.io/)
to manage the setup and execution of the workflow. These
instructions assume a working knowledge of that software. <br />

**WARNING**: Running these scripts will overwrite your local copy of our data (``Results/*`` and ``Results/*``) with the data from your workflow runs. <br />

### LJ Parameter Optimization (OptAT)
To run LJ parameter optimization, follow the following steps:
1. Use init_optff_ms.py to initialize files for simulation use. Change init_opt_at.py as necessary
   ```
     cd genFF_public
     python init_opt_at.py
   ```  
2. Do the following in opt_at_params directory:
3. Generate pareto sets for 1st repeats
   ```
     python project_opt_at.py submit -o gen_pareto_sets -f obj_choice [val] atom_type [val]
   ```   
4. Run the optimization algorithm with repeats
   ```
     python project_opt_at.py submit -o run_obj_alg -f obj_choice [val] atom_type [val]
   ```
5. Run the post analysis algorithm
   ```
     cd genFF_public
     python post_analysis_opt.py
   ```
### VLE Optimization (OptFF)
To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Use init_optff_ms.py to initialize files for simulation use
   ```
     cd genFF_public
     python init_optff_ms.py
   ```          
2. Do the following in opt_ff_ms directory:
3. Check status a few times throughout the process
   ```
     python project_optff_ms.py status 
   ```       
4. Create force fields
   ```
     python project_optff_ms.py run -o create_forcefield
   ```         
5. Calculate vapor/liquid box size
   ```
     python project_optff_ms.py run -o calc_boxes
   ```         
6. Run simulation and check for overlap
   ```
     python project_optff_ms.py submit -o NVT_liqbox --bundle=12 --parallel
     python project_optff_ms.py run -o extract_final_NVT_config
     python project_optff_ms.py submit -o NPT_liqbox --bundle=12 --parallel
     python project_optff_ms.py run -o extract_final_NPT_config
     python project_optff_ms.py submit -o run_gemc --bundle=12 --parallel
     python project_optff_ms.py run -o check_prod_overlap
   ```   
7. Calculate VLE Properties
   ```
     python project_optff_ms.py run -o calculate_props
   ```

### VLE Optimization (GAFF)
To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Use init_optff_ms.py to initialize files for simulation use
   ```
     cd genFF_public
     python init_optff_ms.py
   ```          
2. Do the following in the gaff_ff_ms directory:
3. Check status a few times throughout the process
   ```
     python project_gaff_ms.py status 
   ```       
4. Create force fields
   ```
     python project_gaff_ms.py run -o create_forcefield
   ```         
5. Calculate vapor/liquid box size
   ```
     python project_gaff_ms.py run -o calc_boxes
   ```         
6. Run simulation and check for overlap
   ```
     python project_gaff_ms.py submit -o NVT_liqbox --bundle=12 --parallel
     python project_gaff_ms.py run -o extract_final_NVT_config
     python project_gaff_ms.py submit -o NPT_liqbox --bundle=12 --parallel
     python project_gaff_ms.py run -o extract_final_NPT_config
     python project_gaff_ms.py submit -o run_gemc --bundle=12 --parallel
     python project_gaff_ms.py run -o check_prod_overlap
   ```   
7. Calculate VLE Properties
   ```
     python project_gaff_ms.py run -o calculate_props
   ```

When both GAFF and OptFF Molecular Simulations Are Finished 
1. cd to genFF_public/ (top level directory)
2. Extract VLE properties, Save to Results_MS directory and create pdf of plots
   ```
     python post_analysis_ms.py
   ```

### Known Issues
The instructions outlined above seem to be system-dependent. In some cases, users have the following error:
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
If you observe this, please try the following in the terminal
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
which should fix the problem. This is not an optimal solution and is something we would like to address. We found that related projects [1](https://github.com/openmm/openmm/issues/3943), [2](https://github.com/conda/conda/issues/12410) have similar issues.
If you are aware of a robust solution to this issue, please let us know by raising an issue or sending an email!

## Credits
This research is based upon work supported by the National Science Foundation under award number ERC-2330175 for the Engineering Research Center EARTH as well as grants EFRI 2029354 and CBET-1917474. Computing resources were provided by the Center for Research Computing (CRC) at the University of Notre Dame. MC acknowledges support from the Graduate Assistance in Areas of National Need fellowship from the Department of Education, grant number P200A210048.

## Contact
Please contact Montana Carlozo (mcarlozo@nd.edu) or Dr. Edward Maginn (ed@nd.edu) with any questions, suggestions, or issues.

## Software Versions
This section lists software versions for the most important packages. <br />
cassandra==1.3.1 <br />
foyer==0.12.1 <br />
gpflow==2.9.2 <br />
matplotlib==3.10.1 <br />
mosdef_cassandra==0.4.0 <br />
numdifftools==0.9.41 <br />
numpy==1.26.4 <br />
packmol==20.16.1 <br />
pandas==2.2.3 <br />
panedr==0.8.0 <br />
pymser==1.0.21 <br />
python==3.12.10 <br /> 
scipy==1.15.2 <br />
signac==2.3.0 <br />
signac-flow==0.29.0 <br />
