import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

A = np.arange(25)
B = A ** 2
C = np.linspace(0,50)
print(C)

plt.figure()
plt.plot(A,B)
plt.xlabel("X-axis", fontsize = 20)
