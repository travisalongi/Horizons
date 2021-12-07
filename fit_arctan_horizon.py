"""
Want to see dominant wavelength and see if correspond to width of damage zone
Idea is to fit the x,z with arc tangent functions
This currently doesn't work as expected

Produces plots in Arctan_figs

author: talongi
date: Oct. 2021
"""

# === Load data ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import Trace
from obspy.signal.util import smooth
from scipy.interpolate import UnivariateSpline
plt.ioff()

def inversion(d,G):
    """does least squares inversion | returns model"""
    inverse_G_transpose_G = np.linalg.inv(np.matmul(G.T, G)) 
    G_transpose_d = np.matmul(G.T, d)
    m = np.matmul(inverse_G_transpose_G, G_transpose_d)
    return m

def make_G(x, scale_tan):
    """
    returns a G matrix of arc tangent functions 
    where arctan(x / scale_tan)
    """
    G = np.ones((len(x), len(scale_tan) + 1))
    for i,scaler in enumerate(scale_tan):
        i = i+1
        G[:,i] = np.arctan(x/scaler)
    return G


# === Import - Horizons ===
df = pd.read_csv('all_inversion+2.dat', sep = '\s+',
                 names = ['horizons','x','y','z'])
horizon_names = df.horizons.unique()

# Load unique horizons into dictionary
dfs = dict(tuple(df.groupby('horizons')))
print('Horizons Loaded...')
del df # attempt to keep memory use low

#%%
# === Loop through horizons ===
xlims = [389000 - 1000, 403100 + 1000]
for df in dfs:
    # Select data for horizon from data frame
    hor = dfs[df]
    print(df)
    x,y,z = hor['x'].values, hor['y'].values, hor['z'].values

    # Find where jumps in x are these are x-lines
    w = np.where(np.diff(x) < 10)[0]
    
    # Create colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(w)))

    plt.close('all')
    f,(ax,ax2,ax3) = plt.subplots(nrows = 3, figsize = [10,15])
    
    # Loop through the x-lines
    for i, (low, high) in enumerate(zip(w[:-1], w[1:])):
        # Mask
        m = np.arange(low + 1,high) + 1

        # Select longer profiles & and north of the noise
        if (x[m].max() - x[m].min() > 4000) and  (y[m].min() >3709700):
            
            # Preprocess twt data & use obspy Trace  
            # z_proc = Trace(z[m]).detrend().taper(0.15)
            z_proc = Trace(z[m])
            n_smooth = 30
            dx = np.median(np.diff(x))
            sm = smooth(z_proc, n_smooth)

            # Use spline interpolation and take second derivative 
            # spl = UnivariateSpline(x[m], sm, s = 0)
            # z_dd = spl.derivative(n=1)(x[m])
            # z_dd2 = spl.derivative(n=2)(x[m])

            # Inversion
            # Scaling factors for arctan function
            s = np.logspace(-3,3,100)
            G = make_G(x[m] - np.mean(x[m]),s)
            d = z[m] - np.mean(z[m])
            model = inversion(d,G)
            # Recover inverted tangent data
            d_hat = np.matmul(G,model.reshape(len(model),1))

            # Start plots
            ax.plot(x[m], z_proc, '-', color = colors[i], alpha = 0.15)
            ax2.plot(x[m], sm, '-', color = colors[i], alpha = 0.15)
            ax3.plot(x[m], d_hat, '-', color = colors[i], alpha = 0.15)

            ax.invert_yaxis() 
            ax2.invert_yaxis() 
            ax3.invert_yaxis() 

            ax3.set_ylim([-500, 500])
            
            ax.set_xlim(xlims)
            ax2.set_xlim(xlims)

            ax.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])

            ax.set_ylabel('twt [ms]')
            ax2.set_ylabel('twt [ms]')
            ax3.set_ylabel('')
            
            ax3.set_xlabel('Easting [m]')
            
            ax.set_title('{} Raw'.format(df.split('_')[:-1][0].capitalize()),
                    fontsize = 28)
            ax2.set_title('Smoothed over N samples {} or {} m'.format(n_smooth,
                np.floor(dx * n_smooth),
                fontsize = 28))
            ax3.set_title('Tangent Fit')


    f.savefig('Arctan_figs/{}.png'.format(str(df)))
