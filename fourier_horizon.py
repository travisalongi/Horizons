"""

Do fourier analysis on horizons 
Want to see dominant wavelength and see if correspond to width of damage zone
I think there are still problems in this code
== This is probably not the way I want to proceed ==

author: talongi
date: sept. 2021
"""

# === Load data ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Trace
from obspy.signal.filter import bandpass
from scipy.signal import detrend


# === Import - Horizons ===
df = pd.read_csv('all_inversion+.dat', sep = '\s+',
                 names = ['horizons','x','y','z'])
horizon_names = df.horizons.unique()
# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...'); [print(k) for k in dfs.keys()]
del df # attempt to keep memory use low

# === Loop through horizons ===
for df in dfs:
    # Select data for horizon
    hor = dfs[df]
    x,y,z = hor['x'].values, hor['y'].values, hor['z'].values

    # Find where jumps in x are these are x-lines
    w = np.where(np.diff(x) < 10)[0]
    # Create colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(w)))

    # Start plots & lists to fill
    plt.close('all')
    f,ax = plt.subplots(figsize = [10,10])
    ff,axf = plt.subplots(figsize = [10,10])
    z_list = []
    psd_list = []

    # Loop through the x-lines
    for i, (low, high) in enumerate(zip(w[:-1], w[1:])):
        # Mask
        m = np.arange(low + 1,high) + 1

        # Select longer profiles & and east of the noise
        if (x[m].max() - x[m].min() > 8000) and (x[m].min() > 389800):
            # Preprocess twt data & use obspy Trace  
            z_proc = Trace(z[m]).detrend().taper(0.1)

            # Set up arrays for fourier
            N = len(z_proc)
            dx = np.median(np.diff(x[m]))
            freq = (np.arange(N)/ N/ dx) + 1e-5
            T = 1/freq
            nyquist = int(N/2)


            Z = np.abs(np.fft.fft(z_proc))
            PSD = Z ** 2 * dx / N / dx
            z_list.append(Z)
            psd_list.append(PSD)

            # Make plot of spatial data
            ax.plot(x[m], z_proc, '-', color = colors[i], alpha = 0.15)
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Demeaned Two Way Travel Time [ms]')
            ax.set_title(df.split('_')[:-1][0].capitalize())

            # Make plot of frequency data
            axf.plot(1/freq[1:nyquist], Z[1:nyquist], 
                    '-', color = colors[i], alpha = 0.15)
    axf.set_title(df.split('_')[:-1][0].capitalize())
    f.savefig('Fourier_figs/obspy_{}.png'.format(str(df)))
    ff.savefig('Fourier_figs/freq_{}.png'.format(str(df)))


    # This is not working correctly
    # Stack data and plot
    zd = np.empty((len(z_list), max([len(list) for list in z_list])))
    zd.fill(np.nan)
    for k, list in enumerate(z_list):
        inds = np.arange(len(list))
        zd[k,inds] = list

    avg = np.nanmean(zd, axis = 0)
    ffs, axfs = plt.subplots(figsize = [10,10])
    axfs.plot(avg, '-ok')
    axfs.set_title(df.split('_')[:-1][0].capitalize())
    ffs.savefig('Fourier_figs/stack_{}.png'.format(str(df)))
    
    

