# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Importations
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import sklearn.linear_model as skl
from sklearn.datasets import load_boston

"""
Data from Boston
"""

global boston_data
boston_data = load_boston()
boston_df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_df['MEDV'] = boston_data.target
#print(boston_df)
"""
Creation of the matrix
"""

columns_names = boston_df.columns.array
boston_array = boston_df.index.array

corr_matrix = boston_df.corr().round(1)
sns.heatmap(data=corr_matrix,annot=True)
col_name=(corr_matrix.head(0))
mean_=[]


for i in range(14):
    mean_.append(np.mean(boston_df.iloc[:,i]))
#print(mean_)

#variable without influence:
"""
chas
PTRATIO,RM,ZN-CRIM
RM,RAD,TAX,B,CRIM -ZN
PTRATIO,RM-INOX
B,TAX,RAD,DIS,AGE,NOX,ZN,CRIM-RM
B,PTRATIO,RM-AGE
MEDV,B,PTRATIO,RM-DIS
RM,ZN-RAD
RM,ZN-TAX
B,DIS,AGE,NOX,CRIM-PTRATIO
MEDV,PTRATIO,DIS,AGE,RM,ZN
B,DIS-MEDV


"""


val=corr_matrix.iloc[10,0]# return val(11th line, 1st col)
#print(val)


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

def creat_X():
    X=boston_df
    for j in range(14):
        
        for  i in range(14):
            X["pr"+str(i)+str(j)]=X.iloc[:,j]*X.iloc[:,i]
    return X
X=creat_X()
X_train,X_test,MEDV_train,MEDV_test=train_test_split(X,X['MEDV'],test_size=0.2)
ridge=skl.Ridge(fit_intercept=True).fit(X_train,MEDV_train)
mse_train_ri = mean_squared_error(ridge.predict(X_train),MEDV_train)
mse_test_ri = mean_squared_error(ridge.predict(X_test),MEDV_test)

print("mse train ridge {}".format(mse_train_ri))
print("mse test  ridge {}".format(mse_test_ri))
multiplicity(n=13)
