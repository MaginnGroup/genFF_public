# generalizedFF
Generalized force field for one- and two-carbon single-bonded refrigerants with elements of C, F, and H

### LJ Parameter Optimization (OptAT)
To run LJ parameter optimization, follow the following steps:
1. Make weight dictionary in Results/at_zz/Rxx/weight_sclrs.json. Use form {"Rxx": wt1, "Ryy": wt2}
2. Use init_optff_ms.py to initialize files for simulation use. Change init_opt_at.py as necessary
   ```
     cd generalizedFF
     python init_opt_at.py
   ```  
3. Do the following in opt_at_params directory:
   ```  
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
6. Run simulation
   ```
     python project_optff_ms.py submit -o NVT_liqbox
     python project_optff_ms.py run -o extract_final_NVT_config
     python project_optff_ms.py submit -o NPT_liqbox
     python project_optff_ms.py run -o extract_final_NPT_config
     python project_optff_ms.py submit -o run_gemc --bundle=12 --parallel
   ```   
7. Calculate VLE Properties
   ```
     python project_optff_ms.py run -o calculate_props
   ```

### VLE Optimization (GAFF)
To run vapor-liquid-equilibrium iterations, follow the following steps:
1. Use init_gaff_ms.py to initialize files for simulation use
   ```
     cd generalizedFF
     python init_gaff_ms.py
   ```          
2. Do the following in gaff_ff_ms directory:
3. Check status a few times throughout the process
   ```
     python project_gaff_ms.py status 
   ```       
4. Create force fields
   ```
     python project_gaff_ms.py run -o create_gaff_forcefield
   ```         
5. Calculate vapor/liquid box size
   ```
     python project_gaff_ms.py run -o calc_boxl
     python project_gaff_ms.py submit -o NVT_liqbox
     python project_gaff_ms.py run -o extract_final_NVT_config
     python project_gaff_ms.py submit -o NPT_liqbox
     python project_gaff_ms.py run -o extract_final_NPT_config
   ```         
6. Run simulation
   ```
     python project_gaff_ms.py submit -o GEMC
   ```   
7. Calculate VLE Properties (and make plots for GEMC)
   ```
     python project_gaff_ms.py run -o calculate_props_gaff
     python project_gaff_ms.py run -o plot
   ```

When both GAFF and OptFF Molecular Simulations Are Finished 
1. cd to generalizedFF/ (top level directory)
8. Extract VLE properties, Save to Results_MS directory and create pdf of plots
   ```
     python post_analysis_ms.py
   ```