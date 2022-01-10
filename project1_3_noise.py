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

def noise(sigma_):
    seed_noise=np.random.RandomState(314159)
    sigma=sigma_
    noise=seed_noise.normal(0,sigma_)

    return noise

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

	return term1 + term2 + term3 + term4 


seed_x=np.random.RandomState(123456)
seed_y=np.random.RandomState(654321)
N_max=1000

sigma=[]
complexity=[]
r2_test,r2_train=[],[]
r2_test_,r2_train_=[],[]
mse_test,mse_train=[],[]
mse_test_,mse_train_=[],[]
x=np.sort(seed_x.uniform(0,1,N_max))
y=np.sort(seed_y.uniform(0,1,N_max))

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

X = create_X(x,y,5)
sigma_=10**(-4)

while sigma_ < 100:
    sigma_=sigma_*1.02
    sigma.append(sigma_)
print(len(sigma))
    
for j in range (len(sigma)):
    mse_test_,mse_train_=[],[]
    r2_test_,r2_train_=[],[]

    noise_z=noise(sigma[j])
    z = FrankeFunction(x,y)+noise_z

    #bootstrap
    
    k=1
    order=6

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

    r2_test_mean=sum(r2_test_)*1/k
    r2_train_mean=sum(r2_train_)*1/k
    mse_test_mean=sum(mse_test_)*1/k
    mse_train_mean=sum(mse_train_)*1/k
    
    r2_test.append(r2_test_mean)
    r2_train.append(r2_train_mean)
    mse_test.append(mse_test_mean)
    mse_train.append(mse_train_mean)
"""    
print("MSE train true ={} ".format(mse_train))
print("MSE test true  ={} ".format(mse_test))

print("R2 train true  ={} ".format(r2_train[5]))
print("R2 test true   ={} ".format(r2_test))
"""
  
plt.figure(1)
plt.plot(sigma,mse_test,label='MSE test')
plt.plot(sigma,mse_train,label='MSE train')
plt.xscale('log')
#plt.axis([0,30,0,20])
plt.legend()
plt.savefig("MSE noise.png")

plt.figure(2)
plt.plot(sigma,r2_test,label='R2 test')
plt.plot(sigma,r2_train,label='R2 train')
plt.xscale('log')
plt.legend()
plt.savefig("R2 noise.png")

plt.show()

