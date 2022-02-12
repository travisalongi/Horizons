"""
Oct 2021
Goal: Fit horizons w/ arc tangent function
This is a stepping stop to fit_arctan_horizons.py - a code that doesn't work as expected

Author: Travis Alongi
"""
import numpy as np
import matplotlib.pyplot as plt

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

n_pts = 500
x = np.linspace(385000, 405000, n_pts) 
x = np.linspace(2,10, n_pts) 
d = -5 * np.arctan((x) - np.mean(x)) #+ np.random.randn(n_pts)/5  #+ np.sin(x)
# y_spl = UnivariateSpline(x, d, s = 0)

plt.figure(1)
plt.plot(x,d)

s = np.linspace(-2,5,20)
G = make_G(x,s)
m = inversion(d,G)
m = m.reshape(len(m),1)
model_number = np.argmax(m)

d_hat = np.matmul(G[:], m[:])

plt.figure(2)
plt.plot(x, d, 'ko')
plt.plot(x,d_hat, 'ro')
plt.show()

