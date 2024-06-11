# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:12:01 2024

@author: gsamu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the volatility raw data from CSV file
raw_vol_data = pd.read_csv('noisy.csv', header=None).iloc[:, 2:]
strikes = raw_vol_data.iloc[0, :].to_numpy()
vols = raw_vol_data.iloc[1, :].to_numpy()
plt.plot(strikes, vols, '.', color='b', label='Original points')

x = np.linspace(20.5, 200, 5000)

# Linear interpolation
original_curve = interp1d(strikes, vols, kind='linear')
y = interp1d(strikes, vols, kind='linear')(x)
plt.plot(x, y, '-', color='green',  label='Linear')

# Cubic interpolation
y = interp1d(strikes, vols, kind='cubic')(x)
plt.plot(x, y, '-', color='red', label='Cubic')

# Plot the curves
plt.legend()
plt.show()


# focuse around the duiscontinuity
x = np.linspace(65, 80, 500)

# Linear interpolation
original_curve = interp1d(strikes, vols, kind='linear')
y = interp1d(strikes, vols, kind='linear')(x)
plt.plot(x, y, '-', color='green',  label='Linear')

# Cubic interpolation
y = interp1d(strikes, vols, kind='cubic')(x)
plt.plot(x, y, '-', color='red', label='Cubic')

# Plot the curves
plt.legend()
plt.show()
