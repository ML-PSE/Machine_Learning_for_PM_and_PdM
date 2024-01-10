##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##              FFNN model with debutanizer data and fault detection
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np
import matplotlib.pyplot as plt

#%% random number seed for result reproducibility 
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

#%% read data
data = np.loadtxt('debutanizer_data_withFault.txt') # (Drift) fault starts from last 200 sample onwards

#%% separate training and test data
data_train = data[:-300,:]
data_test = data[-300:,:]

X_train, y_train = data_train[:,0:-1], data_train[:,-1][:,np.newaxis]
X_test, y_test = data_test[:,0:-1], data_test[:,-1][:,np.newaxis]

#%% separate estimation and validation data
from sklearn.model_selection import train_test_split

X_est, X_val, y_est, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 100)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                          Fit FFNN model
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

#%% define model
model = Sequential()
model.add(Dense(40, kernel_regularizer=regularizers.L1(0.000001), activation='relu', kernel_initializer='he_normal', input_shape=(7,)))
model.add(Dense(20, kernel_regularizer=regularizers.L1(0.000001), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, kernel_regularizer=regularizers.L1(0.000001)))

#%% compile model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.005))
model.summary()

#%% fit model
es = EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(X_est, y_est, epochs=2000, batch_size=64, validation_data=(X_val, y_val), callbacks=es)

#%% plot validation curve
plt.figure()
plt.title('Validation Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'], label='fitting')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.grid()
plt.show()

#%% predict C4 content
y_test_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)
y_est_pred = model.predict(X_est)
y_train_pred = model.predict(X_train)

# metrics
from sklearn.metrics import r2_score
print('R2 for validation data:', r2_score(y_val, y_val_pred))
print('R2 for fitting data:', r2_score(y_est, y_est_pred))

#%% plots of raw and predicted data
plt.figure()
plt.plot(y_test, 'b', label='Raw data')
plt.plot(y_test_pred, 'r', label='FFNN prediction')
plt.ylabel('C4 content (test data)')
plt.xlabel('Sample #')
plt.legend()

plt.figure()
plt.plot(y_val, 'b', label='Raw data')
plt.plot(y_val_pred, 'r', label='FFNN prediction')
plt.ylabel('C4 content (validation data)')
plt.xlabel('Sample #')
plt.legend()

plt.figure()
plt.plot(y_train, 'b', label='Raw data')
plt.plot(y_train_pred, 'r', label='FFNN prediction')
plt.ylabel('C4 content (training data)')
plt.xlabel('Sample #')
plt.legend()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                Monitoring statistics for training samples
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q metric for training samples
error_train = y_train - y_train_pred
Q_train = np.sum(error_train*error_train, axis = 1)
Q_CL = np.percentile(Q_train, 95)

# Q_train plot with CL
plt.figure()
plt.plot(Q_train, color='black')
plt.plot([1,len(Q_train)],[Q_CL,Q_CL], linestyle='--',color='red', linewidth=2)
plt.xlabel('Sample #')
plt.ylabel('Q metric: training data')
plt.grid()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                Monitoring statistics for test samples
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Q metric for test samples
error_test = y_test - y_test_pred
Q_test = np.sum(error_test*error_test, axis = 1)

plt.figure()
plt.plot(Q_test, color='black')
plt.plot([1,len(Q_test)],[Q_CL,Q_CL], linestyle='--',color='red', linewidth=2)
plt.xlabel('Sample #')
plt.ylabel('Q metric: training data')
plt.grid()
plt.show()