#*******************************************************************************************
#:)
#:) Script to calculate probabilities of exceeding tephra load and ash concentration
#:) thresholds from Fall3D simulations
#:) @authors
#:) Beatriz Martinez (INGV-Bologna), Oleksandr Rudyy (HLRS) (starting from scripts BET@OV) 
#:)
#********************************************************************************************

import numpy as np
import sys
from sys import argv
from datetime import date, timedelta, datetime
import netCDF4 as nc
import os
import time
from mpi4py import MPI
import argparse
from numba import njit, jit
from datetime import datetime
from scipy.stats import weibull_min
import resource

def opts_parser():
    parser = argparse.ArgumentParser(description="PVHA_WF")
    parser.add_argument('--size', required=True,
                       help='Eruption size (E, L, M or H)')
    parser.add_argument('--pers', required=False,
                       help='Persistence threshold (1, 3, 6, 12, 18, 24)')
    parser.add_argument('--time', help='Time steps to load', required=False)
    parser.add_argument('--fly', required=False,
                       help='Flight level (0 to 7) or all (for all flight levels) or None (for ground)')
    parser.add_argument('--dirscen', required=True,
                       help='Folder where to find the fall3d simulations')
    parser.add_argument('--scen', required=True,
                       help='File with the scenarios and the weight corresponding to each of them')
    parser.add_argument('--season', required=False,
                       help='Season 1 - 4')
    parser.add_argument('--dir_result', required=False,
                       help='directory to put probability matrix')
    parser.add_argument('--num_points', required=False,
                       help='Number of grid points of the fall3d grid')
    return parser

def load_tephra_matrix(rank, load_file, tephra_levels, exposure_time=None):
    nc_format = "NETCDF4_CLASSIC"
    try:
        rootgrp = nc.Dataset(load_file, "r", format=nc_format)
    except Exception as e:
        print ("Rank:::", rank, str(e))
        raise
    lenroot = len(rootgrp.dimensions['time'])
    if exposure_time == None or (exposure_time != None and lenroot < int(exposure_time)):
        print("WARNING: time steps in tephra results is smaller than", exposure_time, load_file, lenroot)
        exposure_time_ = lenroot
    else:
        exposure_time_ = int(exposure_time)
    if tephra_levels[0] == -1: # ground
      try:
        r = rootgrp['tephra_grn_load'][exposure_time_ - 1]
      except Exception as e:
        print ("Rank:", "tephra_levels", tephra_levels, "exposure_time_", exposure_time_, str(e))
        raise
    else:
      try:
        r = rootgrp['tephra_fl'][0:exposure_time_, tephra_levels]
      except Exception as e:
        print ("Rank::", "tephra_levels", tephra_levels, "exposure_time_", exposure_time_, str(e))
        raise
    #r[r=="--"] = 0
    return r

def compute_step1(rank, size, W, list_scen, dir_fileScen, prob_fly,
                  float_th_fly, fly_levels, float_pers, time, tephra_level):
        for i_scen in range(len(list_scen)):
            #print(f"RANK {rank}: Step 1 starting at {datetime.now().time()}")
            #t0 = time.perf_counter()
            peso = W[i_scen]
            #if rank == 0:
            print(f"Rank %s Size %s , weight of scenario %s: %s" % (rank, size, list_scen[i_scen], peso))
            if peso == 0:
                print("WARNING: The weight of this simulation is zero")
                continue
            folder = "Fall_"+list_scen[i_scen][len(list_scen[i_scen])-13:len(list_scen[i_scen])-9]+"_all"
            file_nc = os.path.join(dir_fileScen,  folder)
            file_nc = os.path.join(file_nc, list_scen[i_scen])
            file_nc = os.path.join(file_nc, list_scen[i_scen] + '.res.nc')
            try:
                load_orig_fly = load_tephra_matrix(rank, file_nc, fly_levels, time)
            except Exception as e:
                #if rank == 0:
                print("WARNING:: loading tephra matrix for file", file_nc, str(e), "fly levels", fly_levels, "time", time)
                continue
            compute_probability(prob_fly, load_orig_fly,
                                 peso, float_th_fly, float_pers, tephra_level) #, len(load_orig_fly))
            #print(f"RANK {rank}: Step 1 elapsed-time {time.perf_counter() - t0}.\nFinished at {datetime.now().time()}")
    #print("End step 1")

def compute_step1_ground(rank, size, W, list_scen, dir_fileScen, prob_fly,
                  float_th_fly, fly_levels, float_pers, time, tephra_level, convert_units_thr):
        for i_scen in range(len(list_scen)):
            #print(f"RANK {rank}: Step 1 starting at {datetime.now().time()}")
            #t0 = time.perf_counter()
            peso = W[i_scen]
            #if rank == 0:
            #print(f"GROUND Rank %s Size %s , weight of scenario %s: %s" % (rank, size, list_scen[i_scen], peso))
            if peso == 0:
                print("WARNING: The weight of this simulation is zero")
                continue
            folder = "Fall_"+list_scen[i_scen][len(list_scen[i_scen])-13:len(list_scen[i_scen])-9]+"_all"
            file_nc = os.path.join(dir_fileScen,  folder)
            file_nc = os.path.join(file_nc, list_scen[i_scen])
            file_nc = os.path.join(file_nc, list_scen[i_scen] + '.res.nc')
            try:
                load_orig_fly = load_tephra_matrix(rank, file_nc, fly_levels, time)
            except Exception as e:
                #if rank == 0:
                print("WARNING: loading tephra matrix for file", file_nc, str(e))
                continue
            load_orig_fly = np.nan_to_num(load_orig_fly)   
            compute_probability_ground(prob_fly, load_orig_fly, convert_units_thr,
                                 peso, float_th_fly, float_pers, tephra_level) #, len(load_orig_fly))
            #print(f"RANK {rank}: Step 1 elapsed-time {time.perf_counter() - t0}.\nFinished at {datetime.now().time()}")
    #print("End step 1")

def compute_probability_ground(prob, load_orig, convert_units_thr,
                         peso, float_th, float_pers, tephra_level):
    load_orig = np.transpose(load_orig)  # Transposes the matrix
    load_orig = load_orig.ravel()  # Flattens the matrix
    load_iarea_grn = load_orig * convert_units_thr  # Loads data
    load_iarea_grn = load_iarea_grn[ :, np.newaxis]  # Adapts matrix's dimensions for broadcasting
    prob[0, :, :,0] += (load_iarea_grn >= float_th) * np.float32(peso)

def compute_probability(prob_fly, load_orig_fly,
                         peso, float_th_fly, float_pers, tephra_level):#, exposure_time):
    load_iarea_fly_lv = np.transpose(load_orig_fly[:], axes=(1, 0, 3, 2))  # Transposes last 2 dimensions
    load_iarea_fly_lv = load_iarea_fly_lv.reshape(load_iarea_fly_lv.shape[0], load_iarea_fly_lv.shape[1], load_iarea_fly_lv.shape[2] * load_iarea_fly_lv.shape[
            3])  # Flattens last 2 dimensions
    load_iarea_fly_lv = load_iarea_fly_lv[:, :, :]
    load_iarea_fly_lv = load_iarea_fly_lv[:, :, :, np.newaxis]  # Adapts matrix's dimensions for broadcasting
    count_concentration = np.count_nonzero(np.greater(load_iarea_fly_lv, float_th_fly), axis=1)
    count_concentration = count_concentration[:, :, :, np.newaxis]
    if tephra_level == "all":   #all flys plus the maximum of all
       prob_fly[0:prob_fly.shape[0]-1, :, :, :] += (count_concentration >= float_pers) * np.float32(peso)
       prob_fly[prob_fly.shape[0]-1, :, :, :] += np.max((count_concentration >= float_pers) * np.float32(peso),axis=0)
    else:
       prob_fly[0:prob_fly.shape[0], :, :, :] += (count_concentration >= float_pers) * np.float32(peso)

 
def main(args):

    # arguments ##################################################################
    opts = vars(opts_parser().parse_args(args[1:]))
    size         = opts['size']
    pers         = opts['pers']
    tephra_level = opts['fly']
    exposure_time= opts['time']
    dir_fileScen = opts['dirscen']
    file_scen_W  = opts['scen']
    season       = opts['season']
    dir_result   = opts['dir_result']
    hazard_grid_n= int(opts['num_points'])

    if hazard_grid_n == None:
       #hazard_grid_latlon = "tephra_grid_LATLON_high.txt" #with hazard_grid_n points
       hazard_grid_n = 761761 #958761

    if dir_result == None:
       dir_result   = "./resultLT_CF"
    dir_result   = dir_result + "_" + size
    #dir_fileScen = "../Fall3D_sims_LT"
    file_scen_W = os.path.join(dir_fileScen, file_scen_W)

    if pers == "all":
       pers_thresholds      = [1, 3, 6, 12, 18, 24]     #1 is equivalent to the maximum
    elif pers == None:
       pers_thresholds      = [1]
    else:
       pers_thresholds      = [pers]

    if tephra_level  == None: #ground
       tephra_levels = [-1]
    elif tephra_level  == "all":
       tephra_levels = [0, 1, 2, 3, 4, 5, 6, 7]  # fl 0, fl 1, ...
    else:
       tephra_levels = [tephra_level]

    if tephra_level  == None: #ground
        fl_thresholds = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, \
                       2.0, 2.5, 3, 3.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        convert_units_thr = 0.00980665  # Kg/m2 to KPa
    else:
        fl_thresholds        = [0.0002, 0.002, 0.004] # g/m3    [0.2, 2, 4]     # mg/m3

    if season != None:
        str_season = "_season" + str(season)
    else:
        str_season = ""

    float_th_fly    = np.array([float(v) for v in fl_thresholds])
    float_pers      = np.array([float(v) for v in pers_thresholds])
    float_lev       = np.array([int(v)   for v in tephra_levels])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:

        print ("Working on size", size)
        #print ("Files scenarios:", file_scen)
        print ("Simulations from:", dir_fileScen)
        if tephra_level == None:
            print ("Thresholds:", fl_thresholds, "KPa")
        else:
            print ("Thresholds:", fl_thresholds, "g/m3")
        print ("Flight levels:", tephra_levels)
        print ("Thresholds for persistence:", pers_thresholds)
        print ("Exposure time:", exposure_time)
        print("Season:", str_season)

        # Looking scenarios and their weight
        if os.path.isfile(file_scen_W):
           print("Loading file with scenarios already weighted:", file_scen_W)
           scenarios = np.load(file_scen_W)[:,[0,1]] 
           scenarios[:,1] = np.float32(scenarios[:,1]) # weight
        else:
           print(file_scen_W, "not found")
           exit()
        print("Number of scenarios:", len(scenarios), np.shape(scenarios), "Sum:", str(sum(np.float32(scenarios[:,1])))) 

        # spliting scenarios for parallelization
        scen_chunks = np.array_split(scenarios,nprocs,axis=0)
        scen_chunk = scen_chunks[0]
        for i in range(1, nprocs):
            comm.send(scen_chunks[i], dest=i, tag=11)

    else:
        scen_chunk = comm.recv(source=0, tag=11)

    list_scen = scen_chunk[:,0]
    W = scen_chunk[:,1] 

    if tephra_level == "all": #(all flys)
      prob_fly = np.zeros((len(float_lev)+1, hazard_grid_n, len(float_th_fly), len(float_pers)), np.float32)
    else:
      prob_fly = np.zeros((len(float_lev), hazard_grid_n, len(float_th_fly), len(float_pers)), np.float32)

    #mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #print(f"RANK {rank} pid{ os.getpid()} mem{ mem}") #.format(rank, datetime.now(), os.getpid(), mem))

    if tephra_level  == None: #ground
       compute_step1_ground(rank, size, W, list_scen, dir_fileScen, prob_fly, float_th_fly, float_lev, float_pers, exposure_time, tephra_level, convert_units_thr)
    else:
       compute_step1(rank, size, W, list_scen, dir_fileScen, prob_fly, float_th_fly, float_lev, float_pers, exposure_time, tephra_level)

    #mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    #print(f"RANK {rank} After step1 pid{ os.getpid()} mem{ mem}") #.format(rank, datetime.now(), os.getpid(), mem))

    if rank == 0:
        if tephra_level == "all": #(all flys)
          prob_fly_global = np.zeros((len(float_lev)+1, hazard_grid_n, len(float_th_fly), len(float_pers)),np.float32)
        else:
          prob_fly_global = np.zeros((len(float_lev), hazard_grid_n, len(float_th_fly), len(float_pers)),np.float32)
    else:
        prob_fly_global = None

    comm.Reduce(prob_fly, prob_fly_global, MPI.SUM, root=0)
    
    #mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss    
    #print(f"RANK {rank} After Reduce pid{ os.getpid()} mem{ mem}")

    if rank == 0:

        # Step 2, Calculate VH alpha-beta values
        thresholds_n_fly = prob_fly_global.shape[-2]
        persistency_n = prob_fly_global.shape[-1]
        fly_levels_n = prob_fly_global.shape[0]

        #print(f"Rank {rank} Calculating VH alpha-beta values. Starting at {datetime.now().time()}")
        #t0 = time.perf_counter()

        if tephra_level  == None: # Ground
              print("Ground level, thresholds_n_fly", thresholds_n_fly)
              print("Dumping probability on disk")
              if exposure_time == None:
                 exposure_time = "All"
              prob_path_fly = os.path.join(dir_result, "prob_ground" + "_timeexp" + exposure_time + str_season + ".npy")
              print(prob_path_fly, "min, max", prob_fly_global[0,:,:,0].min(), prob_fly_global[0,:,:,0].max())
              np.save(prob_path_fly, prob_fly_global[0,:,:,0])
              print("prob ground saved on %s" % prob_path_fly)
        else:
          if tephra_level == "all": #(all flys)
            num_tephra_levels = len(float_lev)+1
          else:
            num_tephra_levels = len(float_lev)

          for ilevel in range(num_tephra_levels): 
           for ipers in range(persistency_n):
             print("ilevel, ipers, thresholds_n_fly", ilevel, ipers, thresholds_n_fly)
             try:
               fly_level = str(int(float_lev[ilevel]))
             except:
               fly_level = "alllevels"
             pers_value = str(int(float_pers[ipers]))
             print("Dumping probability on disk")
             if exposure_time == None:
                 exposure_time = "All"
             prob_path_fly = os.path.join(dir_result, "prob_fl" + str(fly_level) + "_pers" + str(pers_value) + "_timeexp" + exposure_time + str_season +".npy")
             np.save(prob_path_fly, prob_fly_global[ilevel,:,:,ipers])
             print("prob" + str(fly_level) + str(pers_value) + "saved on %s" % prob_path_fly)

    return


if __name__ == "__main__":
    main(argv)
    exit(0)
