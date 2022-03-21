#*******************************************************************************************
#:)
#:) Script to shift probabilities at each vent position
#:) @authors
#:) Beatriz Martinez (INGV-Bologna) (starting from scripts BET@OV) 
#:)
#********************************************************************************************

import numpy as np
from sys import argv
import os
import argparse


def opts_parser():
    parser = argparse.ArgumentParser(description="PVHA_WF")

    parser.add_argument('--matrix', required=True,
                        help='Probability matrix to be traslated (the same name for L, M and H)')
    parser.add_argument('--map', required=True,
                        help='Mapping between tephra grid and vents')
    parser.add_argument('--dir', required=True,
                        help='Directory to store results and where to find matrices')
    return parser



def main(args):

   #dir_onevent    = argv[1]   # e.g. resultLT_CF
   #matrix_onevent = argv[3]   # e.g. prob_fl0_pers3_timeexp48.npy
   # For each size, matrices are: resultLT_CF_"size"/prob_fl0_pers3_timeexp48.npy
   #ex: dir_onevent    = "resultLT_CF" for LT CF in Irene
   #    matrix_onevent = "prob_fl0_pers3_timeexp48.npy"


   opts = vars(opts_parser().parse_args(args[1:]))
   matrix_onevent = opts['matrix']   # prob_fl0_pers3_timeexp48.npy
   dir_onevent    = opts['dir']  # resultLT_CF  # Then, matrix will be in resultLT_CF_(L, M, H) /prob_fl0_pers3_timeexp48.npy

   matrix_complet = os.path.join(dir_onevent, matrix_onevent) 

   sizes = ["E", "L", "M", "H"]
   n_sizes = len(sizes)

   # Load probability matrix (Fall3D grid (big) shape (points=761761, fly thresholds=3))  
   prob_L = np.load(os.path.join(dir_onevent+"_L", matrix_onevent))
   prob_M = np.load(os.path.join(dir_onevent+"_M", matrix_onevent))
   prob_H = np.load(os.path.join(dir_onevent+"_H", matrix_onevent))
   n_points     = prob_L.shape[0]
   n_thresholds = prob_L.shape[1]
   print ("Loaded probability matrix", os.path.join(dir_onevent+"_L", matrix_onevent), "whith shape min max", np.shape(prob_L), prob_L.min(), prob_L.max())
   print ("Loaded probability matrix", os.path.join(dir_onevent+"_M", matrix_onevent), "whith shape min max", np.shape(prob_M), prob_M.min(), prob_M.max())
   print ("Loaded probability matrix", os.path.join(dir_onevent+"_H", matrix_onevent), "whith shape min max", np.shape(prob_H), prob_H.min(), prob_H.max())

   # Matrix with the corresponding points for every vent (shape (vents=40, gridpoints=534681) the values of gridpoints are from 0 to 761761)
   grids_mapping_latlon = "./GRIDS/n78_grids_mapping_LATLON_high_40vents.npy"
   vent_tephra_haz_map_latlon = np.load(grids_mapping_latlon)
   vent_grid_n          = vent_tephra_haz_map_latlon.shape[0]     # len = 40 vents
   hazard_grid_latlon_n = vent_tephra_haz_map_latlon.shape[1]     # len = 534681
   print ("Loaded", grids_mapping_latlon, "whith shape", np.shape(vent_tephra_haz_map_latlon))

   # New matrix to save all and use to the BET workflow
   prob = np.zeros((hazard_grid_latlon_n, vent_grid_n, n_sizes, n_thresholds), dtype=np.float32)
   print("shape prob", np.shape(prob))

   isel_tephra = vent_tephra_haz_map_latlon[:, :]
   print("shape isel_tephra", np.shape(isel_tephra))

   for i in range(prob.shape[1]):      # vents
         prob[:,i, 1,:] = prob_L[isel_tephra[i],:]
         prob[:,i, 2,:] = prob_M[isel_tephra[i],:]
         prob[:,i, 3,:] = prob_H[isel_tephra[i],:]
   print("Dumping probability on", matrix_complet)
   np.save(matrix_complet, prob)


if __name__ == "__main__":
    main(argv)
    exit(0)

