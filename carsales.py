#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to smooth a time series data using weighted moving average. The data-
set represents the monthly car sales in Quebec from 1960 to 1968. It is down-
loaded from: https://datamarket.com/data/set/22n4/monthly-car-sales-in-quebec-
1960-1968#!ds=22n4&display=line
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/carsales.csv')
data = np.array(data.iloc[:, 1])
data = data[~np.isnan(data)]
n = len(data)

# a 7-point Gaussian filter with standard deviation of 1
filt  = signal.gaussian(7, 1)
filt /= sum(filt)

# pad half the filter length at the head & tail of data
padded = np.concatenate( (data[0]*np.ones(7//2), data, data[n-1]*np.ones(7//2)) )

# filter the data through convolution
smooth = signal.convolve(padded, filt, mode='valid')

# plot the data and the smoothed curve
plt.figure(1, figsize=(7, 5))
plt.plot(data,   'r')
plt.plot(smooth, 'b')
plt.savefig('carsales_smooth.png')
plt.clf()

# calculate the autocorrelation function
temp = data - data.mean()
corr = signal.correlate(np.concatenate((temp, np.zeros_like(temp))), temp, \
                        mode= 'valid')
# corr /= corr[0]
print(corr)

# plot the autocorrelation function
plt.figure(2, figsize=(7, 5))
plt.plot(corr)
plt.savefig('carsales_corr.png')
plt.clf()





