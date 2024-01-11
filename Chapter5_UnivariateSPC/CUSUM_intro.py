##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          CUSUM Introduction
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# package
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
np.random.seed(10)

#%% generate data
# NOC data
N = 250
x0 = np.random.normal(loc=10, scale=2, size=N)

# faulty data
N = 50
x1 = np.random.normal(loc=11, scale=2, size=N)

# combine data
x = np.hstack((x0,x1))

#%% plots
plt.figure(figsize=(10,3))
plt.plot(x0,'--',marker='o', markersize=4, color='teal')
plt.grid()

plt.figure(figsize=(10,3))
plt.plot(x,'--',marker='o', markersize=4, color='teal')
plt.grid()

#%% CUSUM chart
mu = np.mean(x0)

S = np.zeros((len(x),))
S[0] = 0


for i in range(1,len(S)):
    S[i] = (x[i]-mu) + S[i-1]


plt.figure(figsize=(10,3))
plt.plot(S,'--',marker='o', markersize=4, color='teal')
plt.plot([1,len(S)],[0,0], '--', color='maroon')
plt.xlabel('sample #'), plt.ylabel('CUSUM Statistic')
plt.grid()





