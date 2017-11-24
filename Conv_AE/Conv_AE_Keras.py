
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
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

X_train = X_train.reshape((len(X_train), 28, 28, 1))
X_val = X_val.reshape((len(X_val), 28, 28, 1))

print X_train.shape, X_val.shape


# In[4]:
'''
input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
'''
################################################

input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (5, 5), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

#################################################

autoencoder.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

hist = autoencoder.fit(X_train, X_train, epochs=20, batch_size=100, shuffle=True, validation_data=(X_val, X_val))

decoded_imgs = autoencoder.predict(X_train)

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
    plt.title('Conv_AutoEncoder Regeneration')
    plt.imshow(decoded_imgs[ix].reshape((28, 28)), cmap='gray')
    plt.savefig(('#' + str(ix) + ': Conv_AE-Regenerated ' + LABELS[data[ix, 0]] + '.png'), dpi=326)
    plt.close()


# In[ ]:
