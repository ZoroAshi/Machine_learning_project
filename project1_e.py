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

	for k in range(N) :
		count_1 += 1

		# fullfill the individuals values

		if count_1 == 1:
			for a in range(n):
				Matrix[0][a] = a+1

		