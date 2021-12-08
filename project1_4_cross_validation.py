from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn import datasets
from sklearn import svm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,cross_validate
import sklearn.linear_model as skl
import pandas as pd
import numpy as np

def noise():
    seed_noise=np.random.RandomState(314159)
    noise=seed_noise.normal(0,1)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4

    
seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
k=100
x=np.sort(seed_x.uniform(0,1,k))
y=np.sort(seed_y.uniform(0,1,k))
z=FrankeFunction(x,y)
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


"""
r2_test,r2_train=[],[]
r2_test_,r2_train_=[],[]
mse_test,mse_train=[],[]
mse_test_,mse_train_=[],[]
"""
X = create_X(x,y,5)
ols=skl.LinearRegression()
"""
X_train,X_test,z_train,z_test=cross_validate(X,z)
        
beta=np.linalg.pinv(X_train.T@ X_train)@X_train.T@z_train
z_tild=X_train@beta
z_predict=X_test@beta
"""

scores = cross_validate(ols, X, y, cv=5,scoring=('r2', 'neg_mean_squared_error'),return_train_score=True)
print(scores['test_neg_mean_squared_error'])
print(scores['train_neg_mean_squared_error'])


"""
R2_test=r2_score(z_test,z_predict)
R2_train=r2_score(z_train,z_tild)
MSE_test=mean_squared_error(z_test,z_predict)
MSE_train=mean_squared_error(z_train,z_tild)
"""
