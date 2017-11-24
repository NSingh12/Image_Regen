
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Dense, Activation, Input
from keras.models import Model


# In[2]:

ds = pd.read_csv('../data.csv')
data = ds.values


# In[3]:

X_data = data[:, 1:]
X_std = X_data/255.0

n_train = int(0.84*X_std.shape[0])
n_val = int(0.16*X_std.shape[0])

X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print X_train.shape, X_val.shape


# In[4]:

inp = Input(shape = (784, ))
embedding_dim = 16

fc1 = Dense(embedding_dim*16)(inp)
ac1 = Activation('tanh')(fc1)

fc2 = Dense(embedding_dim*8)(ac1)
ac2 = Activation('tanh')(fc2)

fc3 = Dense(embedding_dim)(ac2)
ac3 = Activation('tanh')(fc3)

fc4 = Dense(embedding_dim*8)(ac3)
ac4 = Activation('tanh')(fc4)

fc5 = Dense(embedding_dim*16)(ac4)
ac5 = Activation('tanh')(fc5)

fc6 = Dense(784)(ac5)
ac6 = Activation('sigmoid')(fc6)

autoencoder = Model(inputs = inp, outputs = ac6)

# In[7]:

encoder = Model(inputs = inp, outputs = ac3)

dec_inp = Input(shape=(embedding_dim,))

x = autoencoder.layers[7](dec_inp)
x = autoencoder.layers[8](x)

x = autoencoder.layers[9](x)
x = autoencoder.layers[10](x)

x = autoencoder.layers[11](x)
x = autoencoder.layers[12](x)

decoder = Model(inputs=dec_inp, outputs=x)


# In[11]:

autoencoder.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
hist = autoencoder.fit(X_train, X_train, epochs=50, batch_size=100, shuffle=True, validation_data=(X_val, X_val))

auto_encoder_encodes = encoder.predict(X_train)                  # Encoder generates a hidden-dimension (64 dim) representation of original data (784 dim)
auto_encoder_decodes = decoder.predict(auto_encoder_encodes)     # Decoder decodes hidden-representation (64 dim) given by encoder to dimensions of input data (784 dim)


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
    plt.title('AutoEncoder Regeneration')
    plt.imshow(auto_encoder_decodes[ix].reshape((28, 28)), cmap='gray')
    plt.savefig(('#' + str(ix) + ': Deep_AE-Regenerated ' + LABELS[data[ix, 0]] + '.png'), dpi=326)
    plt.close()


# In[ ]:
