##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                       Dynamics Assessment
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

#%% generate data
np.random.seed(1)
N = 250
e1 = np.random.normal(loc=0, scale=1, size=N)
e2 = np.random.normal(loc=0, scale=0.2, size=N)

x1 = np.zeros((N,))
for k in range(2,N):
    x1[k] = 0.7*x1[k-1] + e1[k]
    
x2 = 0.5*x1 + e2

plt.figure(figsize=(5,3))
plt.plot(x1, x2, '.', markersize=2, color='teal')
plt.xlabel('x1'), plt.ylabel('x2')
plt.grid()

plt.figure(figsize=(5,3))
plt.plot(x1,'-', color='teal')
plt.xlabel('sample #'), plt.ylabel('x1')
plt.grid()

plt.figure(figsize=(5,3))
plt.plot(x2,'-', color='teal')
plt.xlabel('sample #'), plt.ylabel('x2')
plt.grid()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                    check for dynamics via ACF plot
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.ticker import MaxNLocator

conf_int = 2/np.sqrt(len(x1))

plot_acf(x1, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.ylim([-0.2,1])
plt.xlabel('lag')
plt.show()

plot_acf(x2, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.ylim([-0.2,1])
plt.xlabel('lag')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#           generate non-dynamic data and plot ACFs for comparison
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% generate data
np.random.seed(1)
N = 250
e1 = np.random.normal(loc=0, scale=1, size=N)
e2 = np.random.normal(loc=0, scale=0.2, size=N)

x1 = e1
x2 = 0.5*x1 + e2

plt.figure(figsize=(5,3))
plt.plot(x1, x2, '.', markersize=1, color='teal')
plt.xlabel('x1'), plt.ylabel('x2')
plt.grid()

plt.figure(figsize=(5,3))
plt.plot(x1,'-', color='teal')
plt.xlabel('sample #'), plt.ylabel('x1')
plt.grid()

plt.figure(figsize=(5,3))
plt.plot(x2,'-', color='teal')
plt.xlabel('sample #'), plt.ylabel('x2')
plt.grid()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                    check for dynamics via ACF plot
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conf_int = 2/np.sqrt(len(x1))

plot_acf(x1, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.ylim([-0.2,1])
plt.xlabel('lag')
plt.show()

plot_acf(x2, lags= 20, alpha=None) # alpha=None avoids plot_acf's inbuilt confidence interval plotting
plt.gca().axhspan(-conf_int, conf_int, facecolor='lightblue', alpha=0.5) # shaded confidence interval
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # integer xtick labels
plt.ylim([-0.2,1])
plt.xlabel('lag')
plt.show()