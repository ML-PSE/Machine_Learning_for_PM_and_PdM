##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                             FFT spectrum
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np, matplotlib.pyplot as plt

fs = 1000 # 1000 Hz 
dt = 1.0/fs
duration = 0.5 # 0.5 seconds
t = np.arange(0,0.5,dt) # sampling instants

y1 = np.sin(2*np.pi*50*t ) # 50Hz component
y2 = 2.5*np.sin(2*np.pi*20*t) # 20Hz component
y = y1+y2 # sampled signal

plt.figure(figsize = (20, 3))
plt.plot(t*1000, y, color='black', label='original signal')
plt.plot(t*1000, y, '-*',color='maroon', label='Samples')
plt.ylabel('Amplitude (g)', fontsize=25), plt.xlabel('Time (ms)', fontsize=25)
plt.grid()

#%% generate spectrum
from scipy.fft import rfft, rfftfreq

N = len(t)
Y_spectrum = rfft(y) 
freq_spectrum = rfftfreq(N, dt)

plt.figure(figsize=(8,4))
plt.plot(freq_spectrum, 2.0/N *np.abs(Y_spectrum), 'black')
plt.ylabel('Amplitude (g)', fontsize=25), plt.xlabel('frequency (Hz)', fontsize=25)
plt.grid()
plt.show()