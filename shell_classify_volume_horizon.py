#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Wed Jul 28 2021

Purpose:
    + Reads horizon data exported from ODT, grids the data (down samples)
        - Save the result for Chevron analysis
    + Classify region of seismic volume as above or below horizon

Save volumes:
    1. As boolean, where True = above the horizon
    2. As a distance volume, where each point contains
    minimum distance to the horizon

@author: talongi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TFL_definitions as tfl
from datetime import datetime
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors


# === Read the Data ===
# Import - SHELL Volume
h5_file = '/home/talongi/Gypsy/Project/TFL_shell/all_data_2021.h5'
V = h5py.File(h5_file, 'r')
Vxy, Vz_x = V['coords']['xy'][:], V['coords']['z'][:]
Vx, Vy = Vxy[:,0], Vxy[:,1]

# Working in time gives better results
Vzt = np.arange(56, 3941, 4) # Timelimits output for TFL by ODT

# Import - Horizons
df = pd.read_csv('all_inversion+.dat', sep = '\s+',
                 names = ['horizons','x','y','z'])
horizon_names = df.horizons.unique()
# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...'); [print(k) for k in dfs.keys()]
del df # attempt to keep memory use low


def interpolate_3d(data_to_grid, data_to_interpolate, n_points = 50):
    """
    data_to_grid: list or tuple of array like data
    data_to_interpolate: array
    """
    x1 = data_to_grid[0]
    x2 = data_to_grid[1]
    
    x1i = np.linspace(min(x1), max(x1), n_points)
    x2i = np.linspace(min(x2), max(x2), n_points)
    
    X1, X2 = np.meshgrid(x1i, x2i)
    
    X3 = griddata((x1, x2), data_to_interpolate, (X1, X2))
    
    return X1, X2, X3


#%% Loop through horizons

data_arr_shape = data_arr = V['tfl']['values'].shape
data_xy = np.vstack((Vx,Vy)).T; del Vx, Vy # Now unnecessary
survey_sample_rate = 4 # ms
algorithm = 'ball_tree'

for k in list(dfs.keys())[:-1]:
    unit_name = k.split('_')[0]
    t0 = datetime.now()
    
    # Grid horizon & make plots
    h = dfs[k]    
    x,y,z = interpolate_3d([h['x'], h['y']], h['z'], n_points = 150) # npts = 150 gives grid spacing of ~150m
    
    # Plot & save
    plt.figure(figsize = [10,10])
    plt.scatter(x, y, s = 2.5, c = z, cmap = 'viridis', label = 'Interpolated')
    plt.legend()
    plt.title(unit_name)
    plt.colorbar(label = 'twt [ms]')
    plt.savefig('/home/talongi/Gypsy/Project/Figures/Horizon_grids/' + unit_name + '.png')    
    
    # Save text files
    horizon_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    np.savetxt('/home/talongi/Gypsy/Project/Horizons/Gridded_horizons/' + unit_name + '.xyz', horizon_points, fmt = '%.2f', delimiter = ' ')
    print("{} is gridded, plotted and saved in {}".format(unit_name, datetime.now() - t0))

    # Arrays to fill & store
    dist_arr = np.full(data_arr_shape, np.nan) 
    dir_arr = np.full(data_arr_shape, np.nan)
    
    # Use time for Z
    z_arr = Vzt
    
    # Remove np.nan for distance calc.
    horizon_points = horizon_points[~np.isnan(horizon_points).any(axis = 1)]
    n_z = len(z_arr)
      
    # Make model
    model = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(horizon_points)   

    # Calculate distances knn method for each xy point
    t1 = datetime.now()
    for i, xy in enumerate(data_xy[:]):
    
        # Data formatting 
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
        # plt.plot(difference[:,0], difference[:,1], 'o')
        
        # Positive z-values are above horizon
        above = difference[:,2] > 0
        
        dist_arr[i,:] = d_min_dist.astype(int)
        dir_arr[i,:] = above # x coords are east/west ... x dist from fault
    
    # Convert direction array to boolean and save
    dir_arr = dir_arr.astype(bool)
    np.savetxt('/home/talongi/Gypsy/Project/Horizons/Above_horizon/{}.dat'.format(unit_name), 
               np.hstack((data_xy, dir_arr)), fmt = '%i', delimiter = ' ')
    np.savetxt('/home/talongi/Gypsy/Project/Horizons/Dist_from_horizon/{}.dat'.format(unit_name),
               np.hstack((data_xy, dist_arr)), fmt = '%i', delimiter = ' ')
    print("{} distances calculated from horizon and above/below determined took {}".format(unit_name, datetime.now() - t1))


#%%
# new = np.hstack((data_xy, m_rep2base))
# np.savetxt('mohn2base_hor_bool_test.dat', new, fmt = '%i', delimiter = ' ')
