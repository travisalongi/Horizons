"""

Do fourier analysis on horizons 
Want to see dominant wavelength

author: talongi
date: sept. 2021
"""

# Load data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

# Import - Horizons
df = pd.read_csv('all_inversion+.dat', sep = '\s+',
                 names = ['horizons','x','y','z'])
horizon_names = df.horizons.unique()
# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...'); [print(k) for k in dfs.keys()]
del df # attempt to keep memory use low

# Make figures of demeaned and detrended x-dir lines
for df in dfs:
    # Select data for horizon
    hor = dfs[df]
    x,y,z = hor['x'].values, hor['y'].values, hor['z'].values

    # Find where jumps in x are
    w = np.where(np.diff(x) < 10)[0]

    # Make plots
    f,ax = plt.subplots(figsize = [10,10])
    for low,high in zip(w[:-1], w[1:]):
        m = np.arange(low + 1,high) + 1

        # Select longer profiles & and east of the noise
        if (x[m].max() - x[m].min() > 1000) and (x[m].min() > 389800):
            ax.plot(x[m], detrend(z[m] - z[m].mean()), 'k-', alpha = 0.15)
            ax.set_xlabel('Easting [m]')
            ax.set_ylabel('Demeaned Two Way Travel Time [ms]')
            ax.set_title(df.split('_')[:-1][0].capitalize())
    f.savefig('Fourier_figs/{}.png'.format(str(df)))

