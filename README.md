# generalizedFF
Generalized force field for one- and two-carbon single-bonded refrigerants with elements of C, F, and H

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
     python project.py status 
   ```       
4. Create force fields
   ```
     python project.py run -o create_forcefield
   ```         
5. Calculate vapor/liquid box size
   ```
     python project.py run -o calc_vapboxl
     python project.py run -o calc_liqboxl
   ```         
6. Run simulation
   ```
     python project.py submit -o equilibrate_liqbox --bundle=12 --parallel
     python project.py run -o extract_final_liqbox
     python project.py submit -o run_gemc --bundle=12 --parallel
   ```   
7. Calculate VLE Properties
   ```
     python project.py run -o calculate_props
   ```
8. Extract VLE properties using the following after each VLE iteration in the analysis/ folder
   ```
     python extract_rXX_vle.py ZZ
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
7. Calculate VLE Properties
   ```
     python project_gaff_ms.py run -o calculate_props_gaff
     python project_gaff_ms.py run -o plot
   ```
8. Extract VLE properties using the following after each VLE iteration in the analysis/ folder
   ```
     python extract_rXX_vle.py ZZ
   ```