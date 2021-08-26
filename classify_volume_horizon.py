#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:34:10 2021

@author: talongi
"""
#%% Import

import os, h5py, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import TFL_definitions as tfl
from scipy.interpolate import griddata
from sklearn.neighbors import NearestNeighbors


# Read the Data
# Import - SHELL Volume
h5_file = '../TFL_shell/all_data_2021.h5'
V = h5py.File(h5_file, 'r')
Vxy, Vz_x = V['coords']['xy'][:], V['coords']['z'][:]
Vx, Vy = Vxy[:,0], Vxy[:,1]

# Convert Vz from meters to twt
# Vz_t = tfl.depth_time_conversion(Vz_x, 'time', '../TFL_shell/depth_time_mod.txt')
# Lets actually work in time
Vzt = np.arange(56, 3941, 4) # Timelimits output for TFL by ODT

# Import - Horizons
df = pd.read_csv('L3-1-2.dat', sep = '\s+',
                 names = ['horizons','x','y','z'])
horizon_names = df.horizons.unique()
# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...'); [print(k) for k in dfs.keys()]

# Separte into separate
basement = dfs[horizon_names[1]] 
repetto = dfs[horizon_names[0]]
mohnian = dfs[horizon_names[2]]


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


# Grid the volume?
def grid(data_to_grid, n_points):
    """
    Similar to def interpolate_3d data but just returns grid
    """
    x1 = data_to_grid[0]
    x2 = data_to_grid[1]
    
    x1i = np.linspace(min(x1), max(x1), n_points)
    x2i = np.linspace(min(x2), max(x2), n_points)
    
    X1, X2 = np.meshgrid(x1i, x2i)
    return X1, X2

# Interpolate the horizon
# p = basement
# x,y,z = interpolate_3d([basement['x'], basement['y']], basement['z'], n_points = 100)
# x,y,z = interpolate_3d([repetto['x'], repetto['y']], repetto['z'], n_points = 150)
x,y,z = interpolate_3d((mohnian['x'], mohnian['y']), mohnian['z'], n_points = 100)
# Testing w/ low number of points
# X,Y = grid(Vxy, n_points = 25)

#%% Visual of data
plt.figure(figsize = [10,10])
# plt.scatter(g['x'], g['y'], s = 1.5, c= g['z'], cmap = 'YlOrRd', label = 'gridded data')
# plt.plot(p['x'], p['y'], 'k,', label = 'raw picks')
plt.plot(Vx, Vy,'k,')
plt.scatter(x, y, s = 2.5, c = z, cmap = 'PuBu', label = '2d interp')
plt.legend()
plt.colorbar(label = 'twt [ms]')

#%%

# Need to see how to split the data by horzion
# Middle depth
z_avg = np.ones_like(z) * np.nanmean(z)

# Split the data up
z_above = z > z_avg
z_below = z < z_avg

za = np.where(z_above, z, 0)
zb = np.where(z_below, z, 0)

z_range = [np.nanmin(z), np.nanmax(z)]
    
fig, (ax1,ax2) = plt.subplots(2,1,figsize = [5,10])
ax1.scatter(x, y, s = 2.5, c = za, cmap = 'terrain', vmin = z_range[0], vmax = z_range[1])
ax2.scatter(x, y, s = 2.5, c = zb, cmap = 'terrain', vmin = z_range[0], vmax = z_range[1])

#%% 
# =============================================================================
# DEV
# =============================================================================

data_arr_shape = data_arr = V['tfl']['values'].shape
data_xy = np.vstack((Vx,Vy)).T
survey_sample_rate = 4 # ms
# MAKE definition to process horizons to calc dist
horizon_points = np.column_stack((x.flat, y.flat, z.flat))
algorithm = 'ball_tree'

# Arrays to fill
dist_arr = np.full(data_arr_shape, np.nan) 
dir_arr = np.full(data_arr_shape, np.nan)

# z_arr = np.arange(data_arr_shape[1])
# z_arr = tfl.sample2depth(z_arr.reshape(len(z_arr),1), survey_sample_rate) # other function that converts sample to depth w/ vel. model.
z_arr = Vzt


# Remove np.nan for distance calc.
horizon_points = horizon_points[~np.isnan(horizon_points).any(axis = 1)]
n_z = len(z_arr)
  
# Make model
model = NearestNeighbors(n_neighbors = 1, algorithm = algorithm).fit(horizon_points)   

#%%
# Calculate distances knn method
# For each xy point
t0 = datetime.datetime.now()
for i, xy in enumerate(data_xy[:]):

    # Data formatting 
    # Duplicate the xy point as many times as their are depth points
    tile = np.tile(xy, (n_z,1))

    # Combine the xy points with the depth array
    xyz_tile = np.concatenate((tile, z_arr.reshape(len(z_arr), 1)), axis = 1)
    
    # Calculate distances
    distances, indicies = model.kneighbors(xyz_tile)
    # d_min_dist = distances.min(axis=1)
    horizon_points_min_dist = horizon_points[indicies].reshape(n_z,3)
    
    # Determine vector from horizon to data point
    difference = horizon_points_min_dist - xyz_tile
    # plt.plot(difference[:,0], difference[:,1], 'o')
    
    # Positive z-values are above horizon
    above = difference[:,2] > 0
    
    # dist_arr[i,:] = d_min_dist
    dir_arr[i,:] = above # x coords are east/west ... x dist from fault
dir_arr = dir_arr.astype(bool)
print(datetime.datetime.now() - t0)


#%%
# Save the dir_arr to respective horizon and proceed
# Below repetto & above the basement
m_rep2base = np.logical_and(~dir_mohnian, dir_basement)
plt.imshow(m_rep2base[:1000,:])


#%%

new = np.hstack((data_xy, m_rep2base))
np.savetxt('mohn2base_hor_bool_test.dat', new, fmt = '%i', delimiter = ' ')

#%%

plt.figure(figsize = [10,10])

# plt.plot(Vx, Vy, 'b*', alpha = 1)
plt.plot(data_xy[10000:11000,0], data_xy[10000:11000,1], 'r-')
plt.plot(horizon_points_min_dist[:,0], horizon_points_min_dist[:,1], 'go')
plt.plot(xy[0], xy[1], "k*")
plt.scatter(x, y, s = 1.5, c = z, cmap = 'YlOrRd')
plt.colorbar()


