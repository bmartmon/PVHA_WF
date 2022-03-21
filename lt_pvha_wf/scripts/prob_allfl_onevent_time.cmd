#!/bin/bash

#MSUB -q ...
#MSUB -m ...
#MSUB -n 48
#MSUB -x ...
#MSUB -A ...
###MSUB -T ...

args=("$@")
echo "Analyzing fall3d simulations to calculate arrival time probabilities"
echo "Args: ${args[0]}, ${args[1]}, ${args[2]}, ${args[3]}, ${args[4]}, ${args[5]}, ${args[6]}, ${args[7]}"

##args[0] size
##args[1] directory where the scenarios are
##args[2] file with scenarios and weights
##args[3] directory where the results will be
##args[4] exposuretime
##args[5] threshold
##args[6] number of grid points
##args[7] flight level


if [ $args[7] ]
then
 
   # Flight
   ccc_mprun python3 /path_to_script/prob_allfl_onevent_time.py --size ${args[0]} --dirscen ${args[1]} --scen ${args[2]} --fly ${args[7]} --time ${args[4]} --thre ${args[5]} --dir_result ${args[3]} --num_points ${args[6]}

   # Example: 
   # ccc_mprun python3 /ccc/scratch/cont005/ra5114/ra5114/martmbea/CampiFlegrei/prob_allfl_onevent_time.py --size "L" --dirscen "../Fall3D_sims_LT" --scen "ScenarioslistAndWeight_L_1500scen.npy" --fly "0" --thre "0.002" --time "48" --num_points 761761 --dir_result "./resultLT_CF_L_time"

else

  # Ground
  ccc_mprun python3 /path_to_script/prob_allfl_onevent_time.py --size ${args[0]} --dirscen ${args[1]} --scen ${args[2]} --time ${args[4]} --thre ${args[5]} --dir_result ${args[3]} --num_points ${args[6]}

  # Example:
  # ccc_mprun python3 /path_to_script/prob_allfl_onevent_time.py --size "L" --dirscen "../Fall3D_sims_LT" --scen "ScenarioslistAndWeight_L_1500scen.npy" --thre "0.01" --time "48" --num_points 761761 --dir_result "./resultLT_CF_L_time"


