##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                Dimensionality reduction using Autoencoder
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import required packages
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow
tensorflow.random.set_seed(2)

plt.rcParams.update({'font.size': 14})
np.random.seed(1)

#%% read data
X_train = pd.read_csv('NOC_varyingFeedFlow_outputs.csv', header=None).values
X_train = X_train[:,1:] # first column contains timestamps

#%% split data into fitting and validation datasets
from sklearn.model_selection import train_test_split
X_fit, X_val, _, _ = train_test_split(X_train, X_train, test_size=0.2, random_state=10)

#%% scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit)
X_val_scaled = scaler.transform(X_val)
X_train_scaled = scaler.transform(X_train)
    
#%% define and compile model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(X_fit_scaled.shape[1],)) # input layer
encoded = Dense(1, activation='relu')(input_layer) # encoder layer
decoded = Dense(X_fit_scaled.shape[1], activation='linear')(encoded) # decoder layer
autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

# Compile autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')

# Print model summary
autoencoder.summary()

#%% fit model
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=10)
history = autoencoder.fit(X_fit_scaled, X_fit_scaled, epochs=300, batch_size=256, validation_data=(X_val_scaled, X_val_scaled), callbacks=es)

#%% plot validation curve
plt.figure()
plt.title('Validation Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.plot(history.history['loss'], label='fitting loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.grid()
plt.show()

#%% predict for overall training dataset
X_train_scaled_pred = autoencoder.predict(X_train_scaled)
X_train_pred = scaler.inverse_transform(X_train_scaled_pred)

# compare via plots
plt.figure(figsize=[7,5])
var = 7
plt.plot(X_train[:,var],'seagreen', linewidth=1)
plt.plot(X_train_pred[:,var],'red', linewidth=1)
plt.xlabel('time (mins)')
plt.ylabel('Furnace firebox temperature (T3) ')

plt.figure(figsize=[7,5])
var = 21
plt.plot(X_train[:,var],'seagreen', linewidth=1)
plt.plot(X_train_pred[:,var],'red', linewidth=1)
plt.xlabel('time (mins)')
plt.ylabel('Feed temperature controller valve opening (V1)')

plt.figure(figsize=[7,5])
var = 38
plt.plot(X_train[:,var],'seagreen', linewidth=1)
plt.plot(X_train_pred[:,var],'red', linewidth=1)
plt.xlabel('time (mins)')
plt.ylabel('Reflux flowrate')

#%% predict latents
h_train = encoder.predict(X_train_scaled)

# plot
plt.figure(figsize=[15,5])
plt.plot(X_train_scaled[:,0],'seagreen', linewidth=1, label='Scaled actual feed flow')
plt.plot(h_train, 'purple', linewidth=1, label='latent signal')
plt.xlabel('time (mins)'), plt.ylabel('h')
plt.legend()