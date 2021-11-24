from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
from random import random, seed

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



noise_z=noise()
complexity=[]
r2_test,r2_train=[0]*6,[0]*6
r2_test_,r2_train_=[],[]
mse_test,mse_train=[0]*6,[0]*6
mse_test_,mse_train_=[],[]

seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
N=[20,200,2000,10000,20000,100000]

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


for j in range (6):
    k=N[j]
    x=np.sort(seed_x.uniform(0,1,k))
    y=np.sort(seed_y.uniform(0,1,k))

    z = FrankeFunction(x,y)+noise_z

    X = create_X(x,y,5)

    #bootstrap
    
    k=100
    order=6
    for i in range (order):
        complexity.append(i)

    for i in range (k):

    
        
        X_train,X_test,z_train,z_test=train_test_split(X,z,test_size=0.2)
        
        beta=np.linalg.pinv(X_train.T@ X_train)@X_train.T@z_train
        z_tild=X_train@beta
        z_predict=X_test@beta
        
        R2_test=r2_score(z_test,z_predict)
        R2_train=r2_score(z_train,z_tild)
        MSE_test=mean_squared_error(z_test,z_predict)
        MSE_train=mean_squared_error(z_train,z_tild)
        
        r2_test_.append(R2_test)
        r2_train_.append(R2_train)
        mse_test_.append(MSE_test)
        mse_train_.append(MSE_train)
        
    r2_test=[r2_test+r2_test_ for r2_test,r2_test_ in zip(r2_test,r2_test_)]
    r2_train=[r2_test+r2_test_ for r2_test,r2_test_ in zip(r2_train,r2_train_)]
    mse_test=[r2_test+r2_test_ for r2_test,r2_test_ in zip(mse_test,mse_test_)]
    mse_train=[r2_test+r2_test_ for r2_test,r2_test_ in zip(mse_train,mse_train_)]

    r2_test  =[i*(1/k) for i in r2_test]
    r2_train =[i*(1/k) for i in r2_train]
    mse_test =[i*(1/k) for i in mse_test]
    mse_train=[i*(1/k) for i in mse_train]
    
    """
    print("MSE train true ={} ".format(mse_train[5]))
    print("MSE test true  ={} ".format(mse_test[5]))
    print("R2 train true  ={} ".format(r2_train[5]))
    print("R2 test true   ={} ".format(r2_test[5]))
    """
  
plt.figure(1)
plt.plot(N,mse_test,label='MSE test')
plt.plot(N,mse_train,label='MSE train')
plt.xscale('log')
#plt.axis([0,30,0,20])
plt.legend()

plt.figure(2)
plt.plot(N,r2_test,label='R2 test')
plt.plot(N,r2_train,label='R2 train')
plt.xscale('log')
plt.legend()


plt.show()

