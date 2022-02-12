#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-09-02
Modified Oct. 2021 - to read all_inversion+2.dat

NOTE: run shell_classify_volume_horizon.py first which separates the raw file
Classify portions of the Volume between horizons

Save volumes:
    1. As boolean, where True = above the horizon
    2. As a distance volume, where each point contains a distance from that horizon

Horizons were gridded/interpolated in shell_classify_volume_horizon.py
    + They were saved in /Gridded_horizons
    + Figures were made and saved in /Figures/Horizon_grids/
    
@author: talongi
"""
import glob, h5py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors


# ==== Read the Data === 
# Import - CHEVRON Volume
h5_file = '../TFL_chev/all_data_2021.h5'
V = h5py.File(h5_file, 'r')
Vxy, Vz_x = V['coords']['xy'][:], V['coords']['z'][:]
Vx, Vy = Vxy[:,0], Vxy[:,1]
survey_sample_rate = 4 # ms

# Format the data
data_arr_shape = V['tfl']['values'].shape
data_xy = np.vstack((Vx,Vy)).T
del Vxy, Vx, Vy, Vz_x

# Lets actually work in time -- these are different limits than Shell
Vzt = np.arange(24, 3972, 4); len(Vzt) # Timelimits output for TFL by ODT

# Horizons to work with
hor_dir = 'Gridded_horizons/'
hor_files = sorted(glob.glob(hor_dir + '*'))
[print(f) for f in hor_files]

#%% Loop through horizons
algorithm = 'ball_tree'
for file in hor_files:
    unit_name = file.split('/')[-1][:-4]
    t0 = datetime.now()
    
    # Load previously gridded horizon
    horizon_points = pd.read_csv(file, sep = '\s+').values
    
    # Use time for Z gives best results
    z_arr = Vzt
    
    # Remove np.nan for distance calc.
    horizon_points = horizon_points[~np.isnan(horizon_points).any(axis = 1)]
    n_z = len(z_arr)
      
    # Make model
    model = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(horizon_points)   
    print('{} horizon loaded & nearest neighbors model created in {}'.format(unit_name, datetime.now() - t0))
    print('    {} min / {} max Z values'.format(horizon_points[:,2].min(), horizon_points[:,2].max()))

    # Arrays to fill & store
    dist_arr = np.full(data_arr_shape, np.nan) 
    dir_arr = np.full(data_arr_shape, np.nan)

    # Calculate distances knn method for each xy point
    t1 = datetime.now()
    for i, xy in enumerate(data_xy[:]):
    
        # === Data formatting === 
        # Duplicate the xy point as many times as their are depth points
        tile = np.tile(xy, (n_z,1))
    
        # Combine the xy points with the depth array
        xyz_tile = np.concatenate((tile, z_arr.reshape(len(z_arr), 1)), axis = 1)
        
        # Calculate distances
        distances, indicies = model.kneighbors(xyz_tile)
        d_min_dist = distances.min(axis=1)
        horizon_points_min_dist = horizon_points[indicies].reshape(n_z,3)
        
        # Determine vector from horizon to data point
        difference = horizon_points_min_dist - xyz_tile
        
        # Positive z-values are above horizon
        above = difference[:,2] > 0
        
        # Add data to array
        dist_arr[i,:] = d_min_dist.astype(int)
        dir_arr[i,:] = above 

    
    # Save
    np.savetxt('Chev_above_horizon/{}.dat'.format(unit_name), 
               np.hstack((data_xy, dir_arr)), fmt = '%i', delimiter = ' ')
    np.savetxt('Chev_dist_from_horizon/{}.dat'.format(unit_name),
               np.hstack((data_xy, dist_arr)), fmt = '%i', delimiter = ' ')
    print("{} distances calculated from horizon and above/below determined took {}".format(unit_name, datetime.now() - t1))


#%% This is how to save data in format to read into ODT
# new = np.hstack((data_xy, m_rep2base))
# np.savetxt('mohn2base_hor_bool_test.dat', new, fmt = '%i', delimiter = ' ')
