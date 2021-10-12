"""

Do fourier analysis on horizons 
Want to see dominant wavelength and see if correspond to width of damage zone
I think there are still problems in this code
== This is probably not the way I want to proceed ==
== WIP and probably broken ==

author: talongi
date: sept. 2021
"""

# === Load data ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Trace
from obspy.signal.filter import bandpass
from scipy.signal import detrend, welch


# === Import - Horizons ===
df = pd.read_csv('all_inversion+2.dat', sep = '\s+',
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
            # PSD = Z ** 2 * dx / N / dx
            z_list.append(z_proc.data)
            # psd_list.append(PSD)

            # Make plot of spatial data
            ax.plot(x[m], z_proc, '-', color = colors[i], alpha = 0.15)
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Demeaned Two Way Travel Time [ms]')
            ax.set_title(df.split('_')[:-1][0].capitalize())

            # Make plot of frequency data
            axf.plot(1/freq[1:nyquist], Z[1:nyquist], 
                    '-', color = colors[i], alpha = 0.15)
            axf.grid('all')
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
    
    # Sampling Rate
    fs = np.abs(1 / np.median(np.diff(x)))
    # PSD calc
    f, Pxx = welch(avg, fs=fs, window='hann', nfft=None, 
                   return_onesided=True, scaling='density', axis=-1, average='mean')
    
    
    ffs, axfs = plt.subplots(figsize = [10,10])
    axfs.loglog(1/f, Pxx, '-ok')
    axfs.set_title(df.split('_')[:-1][0].capitalize())
    ffs.savefig('Fourier_figs/PDS_{}.png'.format(str(df)))
    
    

#%%
# Try welch poser spectral
fs = np.abs(1/np.median(np.diff(x)))

f, Pxx = welch(avg, fs=fs, window='hann', nfft=None, 
                          return_onesided=True, scaling='density', axis=-1, average='mean')

plt.figure(figsize = (10,5))
plt.loglog(1/f,Pxx)
# plt.xlim([0,5000])
plt.ylabel('Power Spectral Density ($m^2/s$)') #units are [x]^2[dt]
_ = plt.xlabel('Frequency')


#%%
rng = np.random.default_rng()
fs = 10e3
N = 1e6
amp = 2 * np.sqrt(2)
freq = 1234
time = np.arange(N) / fs
noise_power = 0.001 * fs / 2
x = amp * np.sin(2*np.pi * freq * time)
x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)

plt.figure()
plt.plot(time,x)

f, Pxx = welch(x, 1/fs, nperseg = 1024, 
               window='hann', noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=- 1, average='mean')
plt.figure()
plt.semilogy(f,Pxx)
# plt.ylim([0.5e-3,1])
