"""
Importations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

"""
Data from Boston
"""

global boston_data
boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['MEDV'] = boston_data.target

"""
Creation of the matrix
"""

columns_names = boston_df.columns.array
boston_array = boston_df.index.array


def multiplicity(n=1,N=1) : # n = number of features ; N = multiplicity

	# Initialisation
	count_1 = 0
	Matrix = np.zeros((N,1,n))

	# Begining of the big loop

	while count_1 < N:
			
		count_2 = 0 	# counter of position

		# fullfill the individuals values

		if count_1 == 0:
			for a in range(n):
				Matrix[count_1,0,a] = a

		elif count_1 == 1:
			x1 = 0
			x2 = 1
			L = np.array((x1,x2))
			print(L)
			a = 0

			while x2 < n :
				Matrix[count_1,0]=L
				x2 += 1
				a += 1

		count_1 += 1 # counter of layers
	return Matrix




