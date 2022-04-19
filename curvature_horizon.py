"""
Want to see dominant wavelength and see if correspond to width of damage zone
Issue: Data is too noisy

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
plt.ioff()


# %% Proof of concept -- use spline interpolation to compute 2nd derv.
x = np.linspace(0, 10, 1000)
y = np.sin(x)
y_spl = UnivariateSpline(x, y, s=0)


plt.plot(x, y, 'b')
plt.plot(x, y_spl(x), 'c')
plt.plot(x, y_spl.derivative(n=1)(x), 'g')
plt.plot(x, y_spl.derivative(n=2)(x), 'r')
# %%


# === Import - Horizons ===
df = pd.read_csv('all_inversion+2.dat', sep='\s+',
                 names=['horizons', 'x', 'y', 'z'])
horizon_names = df.horizons.unique()

# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...')
del df  # attempt to keep memory use low

# %%
# === Loop through horizons ===
xlims = [389000 - 1000, 403100 + 1000]
for df in dfs:
    # Select data for horizon from data frame
    hor = dfs[df]
    print(df)
    x, y, z = hor['x'].values, hor['y'].values, hor['z'].values

    # Find where jumps in x are these are x-lines
    w = np.where(np.diff(x) < 10)[0]

    # Create colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(w)))

    plt.close('all')
    f, (ax, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=[10, 15])

    # FOR MAP
    x_map, y_map, z_dd2_map = [], [], []
    f2, axm = plt.subplots(nrows=1, figsize=[10, 10])

    # Loop through the x-lines
    for i, (low, high) in enumerate(zip(w[:-1], w[1:])):
        # Mask
        m = np.arange(low + 1, high) + 1

        # Select longer profiles & and north of the noise
        if (x[m].max() - x[m].min() > 2000) and (y[m].min() > 3709700):

            # Preprocess twt data & use obspy Trace
            # z_proc = Trace(z[m]).detrend().taper(0.15)
            z_proc = Trace(z[m])
            n_smooth = 30
            dx = np.median(np.diff(x))
            sm = smooth(z_proc, n_smooth)

            # Use spline interpolation and take second derivative
            spl = UnivariateSpline(x[m], sm, s=0)
            z_dd = spl.derivative(n=1)(x[m])
            z_dd2 = spl.derivative(n=2)(x[m])

            # Start plots
            ax.plot(x[m], z_proc, '-', color=colors[i], alpha=0.15)
            ax2.plot(x[m], sm, '-', color=colors[i], alpha=0.15)
            ax3.plot(x[m], z_dd, '-', color=colors[i], alpha=0.15)
            ax4.plot(x[m], z_dd2, '-', color=colors[i], alpha=0.15)

            ax.invert_yaxis()
            ax2.invert_yaxis()
            ax3.invert_yaxis()
            ax4.invert_yaxis()

            ax.set_xlim(xlims)
            ax2.set_xlim(xlims)
            ax3.set_xlim(xlims)
            ax4.set_xlim(xlims)

            ax4.set_ylim([-0.01, 0.01])

            ax.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])
            ax3.xaxis.set_ticklabels([])

            ax.set_ylabel('twt [ms]')
            ax2.set_ylabel('twt [ms]')
            ax3.set_ylabel('First Derivative')
            ax4.set_ylabel('Second Derivative')

            ax4.set_xlabel('Easting [m]')

            ax.set_title('{} Raw'.format(df.split('_')[:-1][0].capitalize()),
                         fontsize=28)
            ax2.set_title('Smoothed over N samples {} or {} m'.format(n_smooth,
                                                                      np.floor(
                                                                          dx * n_smooth),
                                                                      fontsize=28))
            ax3.set_title('Derivative of Smoothed Data')
            ax4.set_title('Derivative of Smoothed Data')

            # Map
            x_map.append(x[m])
            y_map.append(y[m])
            z_dd2_map.append(z_dd2)

    # Make tuples into arrays
    x_map = np.concatenate(x_map)
    y_map = np.concatenate(y_map)
    z_dd2_map = np.concatenate(z_dd2_map)

    axm.scatter(x_map, y_map, 0.5, z_dd2_map, vmin=-
                z_dd2_map.std()*2, vmax=z_dd2_map.std()*2)
    axm.axis('equal')

    f.savefig('Curvature_figs/{}.png'.format(str(df)))
    f2.savefig('Curvature_figs/map_{}.png'.format(str(df)))
