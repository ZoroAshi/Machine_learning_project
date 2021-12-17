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


def multiplicity(n=1,N=n) : # n = number of features ; N = multiplicity

	# Initialisation
	count_1 = 0
	Matrix = np.zeros((N,n))

	# Begining of the big loop

	while count_1 < N:
		count_1 += 1	# counter of layers
		count_2 = 0 	# counter of position

		# fullfill the individuals values

		if count_1 == 1:
			for a in range(n):
				Matrix[count_2][a] = a+1


		elif count_1 == 2:
			x1,x2 = 0,1
			a = 0

			while x2 < n :

				Matrix[count_2][a] = np.array((x1,x2))
				x2 += 1

			elif x1 < n-1 :

				x1 +=1 




