# Machine Learning to Optimize Transferable Hydrofluorocarbon Refrigerant Force Fields
Authors: Montana N. Carlozo, Ke Wang, and Alexander W. Dowling
<!-- Introduction: Provide a brief introduction to the project, including its purpose, goals, and any key features or benefits. -->
## Introduction
**genFF_public** is a repository used to calibrate a transferable FF for one- and two-carbon single-bonded refrigerants with elements of C, F, and H given experimental data. The key feature of this work is using machine learning (ML) tools in the form of Gaussian processes (GPs) and estimability analysis techniques to smartly design atom type schemes for tranferable FFs and optimize their LJ parameters. This work features the comparison of four atom typing schemes designed and optimized with ML and GAFF.

## Citation
Please cite as:

Montana N. Carlozo, Ning Wang, Alexander W. Dowling, Edward J. Maginn “Machine Learning to Optimize Transferable Hydrofluorocarbon Refrigerant Force Fields”, 2025

## Available Data

### Repository Organization
The repository is organized as follows: <br />
genFF_public/ is the top level directory. It contains: <br />
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

Directory csv/ contains data used to train the GP models
It contains the following files: <br />
1. rXX-density.csv; The MD density data. <br />
2. rXX-vle.csv; The GEMC data which is used to train the GP models. <br />

Directory example_mcf_files/ contains sample .mcf files for all models and HFCs evaluated in this work
It contains the following files/subdirectories: <br />
1. AT-Y/RXX-species1.mcf are the sample .mcf files for each FF model and HFC

Directory fffit/fffit is a package which contains some critical functions for running the workflow <br />
It contains the following files/subdirectories: <br />
1. tests/ contains the tests for the functions in fffit/fffit
2. __init__.py intializes the package. <br />
3. models.py contains functions related to building GP models. <br />
4. pareto.py contains functions related to locating pareto-optimal parameter sets. <br />
5. plot.py contains some functions for plotting. <br />
6. signac.py contains functions related to parsing data from signac workspaces. <br />
7. utils.py contains utility functions necessary for this package. <br />

Directory molec_gp_data/ contains the GPs and training/testing data for each HFC <br />
It contains the following files/subdirectories: <br />
1. RXX-vlegp are the subdirectories for each HFC. <br />
2. RXX-vlegp/sim_PROP_y_train.csv are the output training data for each property. <br />
3. RXX-vlegp/sim_PROP_y_test.csv are the output testing data for each property. <br />
4. RXX-vlegp/x_train.csv are the input training data for all properties. <br />
5. RXX-vlegp/x_test.csv are the input testing data for all properties. <br />
6. RXX-vlegp/vle-gps.pkl are the pickled GP models for each property. <br />

The pymser directory is a clone of the pymser repository. Refer to their [GitHub Page](https://github.com/IBM/pymser) for more information.

### LJ Parameter Optimization (OptAT)
To run LJ parameter optimization, follow the following steps:
1. Make weight dictionary in Results/at_zz/Rxx/weight_sclrs.json. Use form {"Rxx": wt1, "Ryy": wt2}
2. Use init_optff_ms.py to initialize files for simulation use. Change init_opt_at.py as necessary
   ```
     cd generalizedFF
     python init_opt_at.py
   ```  
3. Do the following in opt_at_params directory:
4. Generate pareto sets for 1st repeats
   ```
     python project_opt_at.py submit -o gen_pareto_sets -f obj_choice [val] atom_type [val]
   ```   
5. Run the optimization algorithm with repeats
   ```
     python project_opt_at.py submit -o run_obj_alg -f obj_choice [val] atom_type [val]
   ```
6. Run the post analysis algorithm
   ```
     cd generalizedFF
     python post_analysis_opt.py
   ```
### VLE Optimization (OptFF)
To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Use init_optff_ms.py to initialize files for simulation use
   ```
     cd generalizedFF
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
     cd generalizedFF
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
1. cd to generalizedFF/ (top level directory)
8. Extract VLE properties, Save to Results_MS directory and create pdf of plots
   ```
     python post_analysis_ms.py
   ```