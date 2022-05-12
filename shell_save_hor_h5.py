#!/usr/bin/env/ python3
"""loads the .dat files and saves them as a single h5 file."""

import glob
import h5py
import TFL_definitions as tfl

# East
h5_file = 'Shell_above_horizon_e_4/all_hor.h5'
hor_dir = 'Shell_above_horizon_e_4/'
# West
h5_file = 'Shell_above_horizon_w_4/all_hor.h5'
hor_dir = 'Shell_above_horizon_w_4/'

f = h5py.File(h5_file, mode='r+')

horizon_files = glob.glob(hor_dir + '*.dat')
print(horizon_files)

for horizon_file in horizon_files:
    hor_name = horizon_file.split('/')[1][:-4]
    print(hor_name)
    horizon, _, _ = tfl.load_odt_att(horizon_file, 4)  # top
    f[hor_name] = horizon

print(list(f.keys()))
f.close()
