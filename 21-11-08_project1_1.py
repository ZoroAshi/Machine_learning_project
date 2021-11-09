"""
title : 21-11-08_project1_1.py
authors : BOUCHARD Ignace ; LEJUEZ Anthony ;
date : 08/11/2021
"""

"""
Importations
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

import sklearn as skl

"""
Franke's function
"""

def franke_fct(x,y) :
	
	terme_1 = (3/4)*np.exp(-((9*x - 2)**2)/4 -((9*y - 2)**2)/4)
	terme_2 = (3/4)*np.exp(-((9*x +1)**2)/49 -((9*y + 1))/10)
	terme_3 = (1/2)*np.exp(-((9*x -7)**2)/4 -((9*y - 3))/4)
	terme_4 = (1/5)*np.exp(-((9*x -4)**2) -((9*y - 7)**2))
	
	f = terme_1 + terme_2 + terme_3 - terme_4
	
	return f

"""
generate the datas
"""

n_col = 6
n_row = 10

X = np.random.uniform(0,1,n_row)
Y = np.random.uniform(0,1,n_row)

matrix = np.zeros(n_col)

print(matrix)

"""
tests
"""
# x,y = 0.5,0.5	# [x,y] E [0,1]²

# test = franke_fct(x,y)
# print(test)
