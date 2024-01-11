##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                       Nonlinearity Assessment
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, matplotlib.pyplot as plt
from ennemi import pairwise_corr

np.random.seed(0)
plt.rcParams.update({'font.size': 20})

#%% generate data
t = np.linspace(0.01,2,100)
x1 = np.zeros((100,1))
x2 = np.zeros((100,1))
x3 = np.zeros((100,1))

for i in range(100):
    x1[i] = t[i] + np.random.normal(scale=0.05)
    x2[i] = np.power(t[i],3) - 3*t[i] + np.random.normal(scale=0.05)
    x3[i] = -np.power(t[i],4) + 3*np.power(t[i],2) +  np.random.normal(scale=0.03)


#%% 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,x3)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')

#%% compute linear and Nonlinear correlation coefficients
data = np.hstack((x1,x2,x3))

# pair-wise linear correlation coefficients
rho_xy = np.corrcoef(data, rowvar=False)
print('rho_xy:', rho_xy)

# pair-wise MI
rho_I_xy = pairwise_corr(data)
print('MI:', rho_I_xy)

# pair-wise nonlinear correlation coefficients
rxy = rho_I_xy*(1-np.abs(rho_xy))
print('rxy:', rxy)

#%% pair-wise scatter plots
plt.figure(figsize=(5,3))
plt.plot(x1,x2,'.',color='black')
plt.xlabel('x1'), plt.ylabel('x2')

plt.figure(figsize=(5,3))
plt.plot(x1,x3,'.',color='black')
plt.xlabel('x1'), plt.ylabel('x3')

plt.figure(figsize=(5,3))
plt.plot(x2,x3,'.',color='black')
plt.xlabel('x2'), plt.ylabel('x3')


