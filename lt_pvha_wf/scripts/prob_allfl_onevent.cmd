#!/bin/bash

#MSUB -q ...
#MSUB -m ...
#MSUB -n 48
#MSUB -x
#MSUB -A ...
#MSUB -T ...


args=("$@")
echo "Analyzing fall3d simulations to calculate probabilities to exceed thresholds"
echo "Args: ${args[0]}, ${args[1]}, ${args[2]}, ${args[3]}, ${args[4]}, ${args[5]}, ${args[6]}, ${args[7]}"

##args[0] size
##args[1] directory where the scenarios are
##args[2] file with scenarios and weights
##args[3] directory where the results will be
##args[4] exposuretime
##args[5] number of grid points
##args[6] persistency
##args[7] flight level


if [ $args[8] ] 
then

   # Flight
   ccc_mprun python3 /path_to_script/prob_allfl_onevent.py --size ${args[0]} --dirscen ${args[1]} --scen ${args[2]} --fly ${args[7]} --pers ${args[6]} --time ${args[4]} --dir_result ${args[3]} --num_points ${args[5]}
   
   # Example:
   # ccc_mprun python3 /path_to_script/prob_allfl_onevent_time.py --size "L" --dirscen "../Fall3D_sims_LT" --scen "ScenarioslistAndWeight_L_1500scen.npy" --pers "all" --fly "all" --time "48" --num_points 761761 --dir_result "./resultLT_CF_L_time"

else
   # Ground
   ccc_mprun python3 /path_to_script/prob_allfl_onevent.py --size ${args[0]} --dirscen ${args[1]} --scen ${args[2]} --time ${args[4]} --dir_result ${args[3]} --num_points ${args[5]}

   #Example:
   #ccc_mprun python3 /ccc/scratch/cont005/ra5114/ra5114/martmbea/CampiFlegrei/prob_allfl_onevent.py --size "L" --time "48" --dirscen "../Fall3D_sims_LT" --scen "ScenarioslistAndWeight_L_1500scen.npy" --dir_result "./resultLT_CF_L_time" --num_points 761761 

