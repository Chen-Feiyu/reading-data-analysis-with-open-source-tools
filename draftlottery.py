"""
Use loess to smooth the data of vietnam draft lottery, 1969. This script is mo-
dified from the workshop 2 from the book Data Analysis with Open Source Tools.
The data of draft lottery are downloaded from 
http://www.randomservices.org/random/data/Draft.html
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# x: location; h: bandwidth; xp, yp: data points (vectors)
def loess(x, h, xp, yp):
    # weight, using Gaussian kernel
    w = np.exp(-0.5*((x-xp)/h)**2) / np.sqrt(2*np.pi*h**2)

    # using linear approximation: y = a + bx
    b  = np.sum(w)*np.sum(w*xp*yp) - np.sum(w*xp)*np.sum(w*yp)
    b /= np.sum(w)*np.sum(w*xp**2) - np.sum(w*xp)**2
    a  = (np.sum(w*yp) - b*np.sum(w*xp)) / np.sum(w)

    return a + b*x

data = pd.read_table('datasets/draftlottery.txt')
# draft number from year 1969
yp = np.array(data['N69'])
# date of birth / day in year
xp = np.array(range(1, len(data)+1))
# remove the item representing Feb 29th
xp = np.delete(xp, np.where(yp == '*'))
yp = np.delete(yp, np.where(yp == '*'))
xp = xp.astype(np.float)
yp = yp.astype(np.float)

# smooth the data using loess
s1, s2 = [], []
for k in xp:
    s1.append( loess(k,   5, xp, yp) )
    s2.append( loess(k, 100, xp, yp) )

plt.xlabel('Day in Year')
plt.ylabel('Draft Number')

# set the aspect ratio of the two axes equal (a square plot)
plt.gca().set_aspect('equal')

# plot the data points and smooth curves
plt.plot(xp, yp, 'bo', alpha=0.2)
plt.plot(xp, np.array(s1), 'k-', xp, np.array(s2), 'k--')

# set the limit of the two axes
q = 4
plt.axis([1-q, 366+q, 1-q, 366+q])

plt.savefig("draftlottery.eps")

