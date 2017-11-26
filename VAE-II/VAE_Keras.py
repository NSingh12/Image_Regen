
# coding: utf-8

# In[14]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

# https://blog.keras.io/building-autoencoders-in-keras.html


# In[30]:

ds = pd.read_csv('../train.csv')
data = ds.values

X_data = data[:, 1:]
X_std = X_data/255.0

n_train = int(0.84*X_std.shape[0])
n_val = int(0.16*X_std.shape[0])

X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print X_train.shape, X_val.shape


# In[24]:

batch_size = 128
original_dim = 784
latent_dim = 16
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0


# In[25]:

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


# In[26]:

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])


# In[27]:

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


# In[28]:

# end-to-end autoencoder
vae = Model(x, x_decoded_mean)

# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


# In[29]:

vae.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
vae.fit(X_train, X_train, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val))

# In[ ]:

X_test_encoded = encoder.predict(X_train[:100])
X_test_decoded = generator.predict(X_test_encoded)


# In[ ]:

LABELS = {}
LABELS[0] = 'T-Shirt'
LABELS[1] = 'Trouser'
LABELS[2] = 'Pullover'
LABELS[3] = 'Dress'
LABELS[4] = 'Coat'
LABELS[5] = 'Sandal'
LABELS[6] = 'Shirt'
LABELS[7] = 'Sneaker'
LABELS[8] = 'Bag'
LABELS[9] = 'Ankle_Boot'

for ix in range(100):
    plt.figure(ix)
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(X_train[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(1,2,2)
    plt.title('Variational AutoEncoder Regeneration')
    plt.imshow(X_test_decoded[ix].reshape((28, 28)), cmap='gray')
    plt.savefig(('#' + str(ix) + ': Variational_AE-Regenerated ' + LABELS[data[ix, 0]] + '.png'), dpi=326)
    plt.close()

