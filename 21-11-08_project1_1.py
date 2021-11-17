"""
title : 21-11-08_project1_1.py
authors : BOUCHARD Ignace ; LEJUEZ Anthony ; DELILLE Hugo
date : 08/11/2021
"""

"""
Importations
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas as pd
import numpy as np
from random import random, seed

plt.close()
fig = plt.figure()
ax = fig.gca(projection="3d")

# Make data

x0 = np.arange(0, 1, 0.05)
y0 = np.arange(0, 1, 0.05)

x, y = np.meshgrid(x0,y0)

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)

# Plot the surface

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

# Add a color bar which maps values to colors

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# Part a)

# creation of the matrix (x,y,x²,xy)

matrix = np.zeros((len(y0)**2,6))

# First column

matrix[:,0] = 1

# Second column

X0 = []
for k in range(len(x0)):
	for i in range(len(x0)) :
		xk =  x0[k]
		X0.append(xk)

matrix[:,1] = X0

# Third column

Y0 = []
for k in range(len(y0)) :
	for i in range(len(y0)) :
		yk = x0[i]
		Y0.append(yk)

matrix[:,2] = Y0

# Fourth column

X02 = []
for k in range(len(x0)**2):
	X02.append(X0[k]**2)

matrix[:,3] = X02

# Fith column

Y02 = []
for k in range(len(y0)**2):
	Y02.append(Y0[k]**2)
	
matrix[:,4] = Y02

# six column

X0Y0 = []
for k in range(len(x0)*len(y0)):
	X0Y0.append(X0[k]*Y0[k])
	
matrix[:,5] = X0Y0


# print(matrix)



# Tests

def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)
	N = len(x)
	l = int((n+1)*(n+2)/2) # Number of elements in beta
	X = np.ones((N,l))
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)
	return X

X = create_X(x,y,2)
print(matrix)
print(len(matrix))
print(X)
print(len(X))