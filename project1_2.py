from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
from random import random, seed
def noise():
    noise=np.random.normal(0,1)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4 



noise_z=noise()
N=1000
x=np.sort(np.random.uniform(0,1,N))
y=np.sort(np.random.uniform(0,1,N))

z = FrankeFunction(x,y)+noise_z


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

complexity=[]
r2_test,r2_train=[],[]
mse_test,mse_train=[],[]




for n in range(30):
    X = create_X(x,y,n)
    """print(matrix)
    print(len(matrix))
    print(X)
    print(len(X))"""

 
    X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2)
    beta=np.linalg.inv(X_train.T@ X_train)@X_train.T@z_train
    z_tild=X_train@beta
    z_predict=X_test@beta
    R2_test=r2_score(z_test,z_predict)
    R2_train=r2_score(z_train,z_tild)
    MSE_test=mean_squared_error(z_test,z_predict)
    MSE_train=mean_squared_error(z_train,z_tild)

    complexity.append(n)
    r2_test.append(R2_test)
    r2_train.append(R2_train)
    mse_test.append(MSE_test)
    mse_train.append(MSE_train)
    """
    print("MSE train ={} ".format(MSE_train))
    print("MSE test  ={} ".format(MSE_test))
    print("R2 train  ={} ".format(R2_train))
    print("R2 test   ={} ".format(R2_test))
    """
plt.figure(1)
plt.plot(complexity,mse_test,label='MSE test')
plt.plot(complexity,mse_train,label='MSE train')
#plt.axis([0,30,0,20])
plt.legend()
plt.show()
