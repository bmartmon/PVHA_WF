#*******************************************************************************************
#:)
#:) Script to calculate arrival times to exceed tephra load and ash concentration 
#:) thresholds from Fall 3D simulations
#:) @authors
#:) Beatriz Martinez (INGV-Bologna)
#:)
#********************************************************************************************

import numpy as np
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

def opts_parser():
    parser = argparse.ArgumentParser(description="PVHA_WF")
    parser.add_argument('--size', required=True,
                       help='Eruption size (E, M, L or S)')
    parser.add_argument('--fly', required=False,
                       help='Flight level (0,...,7)')
    parser.add_argument('--dirscen', required=True,
                       help='Folder where to find the fall3d simulations')
    parser.add_argument('--scen', required=True,
                       help='File with the scenarios and the weight corresponding to each of them')
    parser.add_argument('--thre', required=False,
                       help='Flight threshold (0.0002, 0.002 or 0.004)     # g/m3 ')
    parser.add_argument('--time', help='Time steps to load', required=True)
    parser.add_argument('--num_points', required=False,
                       help='Number of grid points of the fall3d grid')
    parser.add_argument('--dir_result', required=False,
                       help='directory to put probability matrix')
    return parser

def load_tephra_matrix(rank, load_file, tephra_levels, exposure_time=None):
    nc_format = "NETCDF4_CLASSIC"
    try:
        rootgrp = nc.Dataset(load_file, "r", format=nc_format)
    except Exception as e:
        print ("Rank", rank, str(e))
        raise
    lenroot = len(rootgrp.dimensions['time'])
    if exposure_time == None or (exposure_time != None and lenroot < int(exposure_time)):
        print("WARNING: time steps in tephra results is smaller than", exposure_time, load_file, lenroot)
        exposure_time_ = lenroot
    if lenroot < exposure_time:
        print("WARNING: time steps in tephra results is smaller than", exposure_time, load_file, lenroot)
        exposure_time_ = lenroot
    else:
        exposure_time_ = int(exposure_time)
    if tephra_levels[0] == -1: # ground
       return rootgrp['tephra_grn_load'][0:exposure_time_]
    else:
       return rootgrp['tephra_fl'][0:exposure_time_, tephra_levels]

def compute_step1(rank, size, W, list_scen, dir_fileScen, prob_fly,
                  float_th_fly, fly_levels, convert_units_thr):#, float_pers):
        for i_scen in range(len(list_scen)):
            peso = W[i_scen]
            print(f"Rank %s Size %s , weight of scenario %s: %s" % (rank, size, list_scen[i_scen], peso))
            #if rank == 0:
            #print(f"Size %s , weight of scenario %s: %s" % (size, list_scen[i_scen], peso))
            if peso == 0:
                print("WARNING: The weight of this simulation is zero")
                continue
            folder = "Fall_"+list_scen[i_scen][len(list_scen[i_scen])-13:len(list_scen[i_scen])-9]+"_all"
            file_nc = os.path.join(dir_fileScen,  folder)
            file_nc = os.path.join(file_nc, list_scen[i_scen])
            file_nc = os.path.join(file_nc, list_scen[i_scen] + '.res.nc')
            try:
                load_orig_fly = load_tephra_matrix(rank, file_nc, fly_levels, prob_fly.shape[0])
                #if (load_orig_fly.shape[0] < prob_fly.shape[0]):
                #  continue
            except Exception as e:
                #if rank == 0:
                print("WARNING: loading tephra matrix for file", file_nc, str(e))
                continue
            if fly_levels[0] == -1: # ground
              load_orig_fly_extend = np.zeros((prob_fly.shape[0], load_orig_fly.shape[1], load_orig_fly.shape[2]), np.float32)
              load_orig_fly_extend[:load_orig_fly.shape[0]] = load_orig_fly
              compute_probability_ground(prob_fly, load_orig_fly_extend,
                                 peso, float_th_fly, convert_units_thr)
            else:
              load_orig_fly_extend = np.zeros((prob_fly.shape[0], load_orig_fly.shape[1], load_orig_fly.shape[2], load_orig_fly.shape[3]), np.float32)
              load_orig_fly_extend[:load_orig_fly.shape[0]] = load_orig_fly
              compute_probability(prob_fly, load_orig_fly_extend,
                                 peso, float_th_fly, convert_units_thr)
            #print(f"RANK {rank}: Step 1 elapsed-time {time.perf_counter() - t0}.\nFinished at {datetime.now().time()}")
    #print("End step 1")

def compute_probability_ground(prob_fly_time, load_orig_fly,
                         peso, float_th_fly, convert_units_thr):
    load_iarea_fly_lv = np.transpose(load_orig_fly[:], axes=(0, 2, 1))
    load_iarea_fly_lv = load_iarea_fly_lv.reshape(load_iarea_fly_lv.shape[0], load_iarea_fly_lv.shape[1] * load_iarea_fly_lv.shape[
            2])  # Flattens last 2 dimensions
    load_iarea_fly_lv = load_iarea_fly_lv[:,:] * convert_units_thr  # Load data
    load_iarea_fly_lv = load_iarea_fly_lv[:, :, np.newaxis]  # Adapts matrix's dimensions for broadcasting
    for i in range(load_iarea_fly_lv.shape[1]): #grid
       for t in range(load_iarea_fly_lv.shape[0]): #time
          if load_iarea_fly_lv[t,i] >= float_th_fly:
             prob_fly_time[t,:,i,:] +=  np.float32(peso)
             break

def compute_probability(prob_fly_time, load_orig_fly,
                         peso, float_th_fly, convert_units_thr): #, float_pers):#, exposure_time):
    load_iarea_fly_lv = np.transpose(load_orig_fly[:], axes=(1, 0, 3, 2))  # Transposes last 2 dimensions
    load_iarea_fly_lv = load_iarea_fly_lv.reshape(load_iarea_fly_lv.shape[0], load_iarea_fly_lv.shape[1], load_iarea_fly_lv.shape[2] * load_iarea_fly_lv.shape[
            3])  # Flattens last 2 dimensions
    load_iarea_fly_lv = load_iarea_fly_lv[:, :, :] * convert_units_thr  # Load data
    load_iarea_fly_lv = load_iarea_fly_lv[:, :, :, np.newaxis]  # Adapts matrix's dimensions for broadcasting
    for i in range(load_iarea_fly_lv.shape[2]): #grid
       for t in range(load_iarea_fly_lv.shape[1]): #time
          if load_iarea_fly_lv[:,t,i] >= float_th_fly:
             prob_fly_time[t,:,i,:] +=  np.float32(peso)
             break

def main(args):

    # arguments ##################################################################
    opts = vars(opts_parser().parse_args(args[1:]))
    size          = opts['size']
    fly           = opts['fly']
    dir_fileScen  = opts['dirscen']
    file_scen_W   = opts['scen']
    thre          = opts['thre']
    exposure_time = int(opts['time'])
    dir_result    = opts['dir_result']
    hazard_grid_n = int(opts['num_points'])

    if hazard_grid_n == None:
       #hazard_grid_latlon = "tephra_grid_LATLON_high.txt" #with hazard_grid_n points
       hazard_grid_n = 761761 #958761

    if dir_result == None:
       dir_result   = "./resultLT_CF_" + size + "_time"
    #dir_fileScen = "../Fall3D_sims_LT"
    file_scen_W = os.path.join(dir_fileScen, file_scen_W)

    if fly  == None: #ground
       if thre == None:
          fl_thresholds      = [0.1] ## [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, \
                                    ##      2.0, 2.5, 3, 3.5, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
       else:
          fl_thresholds      = [thre]
       convert_units_thr = 0.00980665  # Kg/m2 to KPa
       tephra_levels = [-1]
    else:
        if thre == None:
           fl_thresholds      = [0.002] #[0.0002 , 0.002, 0.004]     # g/m3
        else:
           fl_thresholds      = [thre]
        convert_units_thr = 1.0
        if fly == "all":
           tephra_levels =   [0, 1, 2, 3, 4, 5, 6, 7]  #fl 0, fl 1, ...
        else:
           tephra_levels =   [fly]

    float_th_fly    = np.array([float(v) for v in fl_thresholds])
    float_lev       = np.array([int(v) for v in tephra_levels])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:

        print ("Working on size", size)
        print ("Simulations from:", dir_fileScen)
        if fly == None:
            print ("Thresholds:", fl_thresholds, "KPa")
        else:
            print ("Thresholds:", fl_thresholds, "g/m3")
        print ("Flight levels:", tephra_levels)
        print ("Exposure time:", exposure_time)
        print ("file_scen_W:", file_scen_W)
        print ("Results will be in", dir_result)

        print("float_th_fly", float_th_fly)
        print("float_lev", float_lev)


        # Looking scenarios and their weight
        if os.path.isfile(file_scen_W):
           print("Loading file with scenarios already weighted:", file_scen_W)
           scenarios = np.load(file_scen_W)[:,[0,1]]
           scenarios[:,1] = np.float32(scenarios[:,1]) # weight
        else:
           print("File", file_scen_W, "not found")
           exit()
        print("Number of scenarios:", len(scenarios), np.shape(scenarios), "Sum:", str(sum(np.float32(scenarios[:,1]))))

        # spliting scenarios for parallelization
        scen_chunks = np.array_split(scenarios,nprocs,axis=0)
        scen_chunk = scen_chunks[0]
        for i in range(1, nprocs):
            comm.send(scen_chunks[i], dest=i, tag=11)

    else:
        scen_chunk = comm.recv(source=0, tag=1)

    list_scen = scen_chunk[:,0]
    W = scen_chunk[:,1]

    prob_fly = np.zeros((exposure_time, len(float_lev), hazard_grid_n, len(float_th_fly)),np.float32) #, len(float_pers)),np.float32)

    compute_step1(rank, size, W, list_scen, dir_fileScen, prob_fly, float_th_fly, float_lev, convert_units_thr)  #, float_pers)

    if rank == 0:
        prob_fly_global = np.zeros((exposure_time, len(float_lev), hazard_grid_n, len(float_th_fly)),np.float32) #, len(float_pers)),np.float32)
    else:
        prob_fly_global = None

    comm.Reduce(prob_fly, prob_fly_global, MPI.SUM, root=0)
 
    if rank == 0:
        
        # Step 2, Calculate VH alpha-beta values
        thresholds_n_fly = prob_fly_global.shape[-1]
        fly_levels_n = prob_fly_global.shape[1]
       
        if tephra_levels[0] == -1: # ground
          for ilevel in range(len(float_lev)):
            for ithr in range(thresholds_n_fly):
               print("Dumping probability on disk for ground, thres", fl_thresholds[ithr])
               prob_path_fly = os.path.join(dir_result, "prob_time_ground_thr" + str(fl_thresholds[ithr]) + "_time" + str(exposure_time) + ".npy")
               print("prob_path_fly shape, max, min", np.shape(prob_fly_global), prob_fly_global.max(), prob_fly_global.min())
               print("-", np.shape(np.squeeze(prob_fly_global[:,ilevel,:,ithr])))
               print("--", np.shape(np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr])) ))
               np.save(  prob_path_fly, np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr]))  )
               print( "saved on %s" % prob_path_fly)

        else:
          for ilevel in range(len(float_lev)): #+1): # +1 because the last one is for the maximum around fly levels
           for ithr in range(thresholds_n_fly):
             print("Dumping probability on disk for level", tephra_levels[ilevel], "thres", fl_thresholds[ithr]) #ilevel, ithr
             prob_path_fly = os.path.join(dir_result, "prob_time_fl" + str(tephra_levels[ilevel]) + "_thr" + str(fl_thresholds[ithr]) + "_time" + str(exposure_time) + ".npy")
             print("prob_path_fly shape, max, min", np.shape(prob_fly_global), prob_fly_global.max(), prob_fly_global.min())
             print("-", np.shape(np.squeeze(prob_fly_global[:,ilevel,:,ithr])))
             print("--", np.shape(np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr])) ))
             np.save(  prob_path_fly, np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr]))  )
             print( "saved on %s" % prob_path_fly)


        '''
        for ilevel in range(len(float_lev)): #+1): # +1 because the last one is for the maximum around fly levels
          #for ipers in range(persistency_n):
          for ithr in range(thresholds_n_fly):
           #print("ilevel, ipers, thresholds_n_fly", ilevel, ipers, thresholds_n_fly)
 
           #try:
           #  fly_level = str(int(float_lev[ilevel]))
           #except:
           #  fly_level = "alllevels"
           ##pers_value = str(int(float_pers[ipers]))
           #thr_value = str(ithr)    

           print("Dumping probability on disk for level", tephra_levels[ilevel], "thres", fl_thresholds[ithr]) #ilevel, ithr

           prob_path_fly = os.path.join(dir_result, "prob_time_fl" + str(tephra_levels[ilevel]) + "_thr" + str(fl_thresholds[ithr]) + "_time" + str(exposure_time) + ".npy") # + "_pers" + str(pers_value) + "_float32.npy")
           print("prob_path_fly shape, max, min", np.shape(prob_fly_global), prob_fly_global.max(), prob_fly_global.min())
           print("-", np.shape(np.squeeze(prob_fly_global[:,ilevel,:,ithr])))
           print("--", np.shape(np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr])) ))

           #np.save(  prob_path_fly, np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr,ipers]), axes=(1,0))  )
           np.save(  prob_path_fly, np.transpose(np.squeeze(prob_fly_global[:,ilevel,:,ithr]))  )
           print( "saved on %s" % prob_path_fly)
        '''
    return


if __name__ == "__main__":
    #print ("Starting")
    main(argv)
    exit(0)
