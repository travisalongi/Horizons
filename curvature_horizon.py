"""
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
from obspy.signal.util import smooth
from scipy.interpolate import UnivariateSpline


#%% Proof of concept -- use spline interpolation to compute 2nd derv.
x = np.linspace(0,10,1000)
y = x**3
y_spl = UnivariateSpline(x, y, s = 0)


plt.plot(x,y,'b')
plt.plot(x,y_spl(x),'c')
plt.plot(x, y_spl.derivative(n=1)(x),'g')
plt.plot(x, y_spl.derivative(n=2)(x), 'r')
#%%


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
    # Select data for horizon from data frame
    hor = dfs[df]
    x,y,z = hor['x'].values, hor['y'].values, hor['z'].values

    # Find where jumps in x are these are x-lines
    w = np.where(np.diff(x) < 10)[0]
    # Create colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(w)))

    plt.close('all')
    f,(ax,ax2,ax3) = plt.subplots(nrows = 3, figsize = [10,10])
    # Loop through the x-lines
    for i, (low, high) in enumerate(zip(w[:-1], w[1:])):
        # Mask
        m = np.arange(low + 1,high) + 1

        # Select longer profiles & and east of the noise
        if (x[m].max() - x[m].min() > 500) and (x[m].min() > 389800):

            # Preprocess twt data & use obspy Trace  
            # z_proc = Trace(z[m]).detrend().taper(0.15)
            z_proc = Trace(z[m])
            n_smooth = 30
            dx = np.median(np.diff(x))
            sm = smooth(z_proc, n_smooth)

            # Use spline interpolation and take second derivative 
            spl = UnivariateSpline(x[m], sm, s = 0)
            z_dd = spl.derivative(n=1)(x[m])

            # Start plots
            ax.plot(x[m], z_proc, '-', color = colors[i], alpha = 0.15)
            ax2.plot(x[m], sm, '-', color = colors[i], alpha = 0.15)
            ax3.plot(x[m], z_dd, '-', color = colors[i], alpha = 0.15)

            ax.invert_yaxis() 
            ax2.invert_yaxis() 
            ax3.invert_yaxis() 
            
            ax.set_ylabel('twt [ms]')
            ax2.set_ylabel('twt [ms]')
            ax3.set_ylabel('Second derivative')
            
            ax3.set_xlabel('Easting [m]')
            
            ax.set_title('{} Raw'.format(df.split('_')[:-1][0].capitalize()),
                    fontsize = 28)
            ax2.set_title('Smoothed over N samples {} or {} m'.format(n_smooth,
                dx * n_smooth,
                fontsize = 28))
            ax3.set_title('Derivative of Smoothed Data')

    f.savefig('Curvature_figs/{}.png'.format(str(df)))
