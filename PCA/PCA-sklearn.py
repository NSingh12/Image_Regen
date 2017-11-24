
# coding: utf-8

# In[3]:

from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# In[5]:

ds = pd.read_csv('../data.csv')
data = ds.values

X_data = data[:, 1:]
X_std = X_data/255.0

n_train = int(0.84*X_std.shape[0])
n_val = int(0.16*X_std.shape[0])

X_train = X_std[:n_train]
X_val = X_std[n_train:n_train+n_val]

print X_train.shape, X_val.shape


# In[6]:

pca = PCA(n_components=16)
pca_dim_reducts = pca.fit_transform(X_std[:(n_train + n_val)])

pca_regenerations = pca.inverse_transform(pca_dim_reducts)

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

# In[ ]:

for ix in range(100):
    plt.figure(ix)
    plt.subplot(1,2,1)
    plt.title('Original')
    plt.imshow(X_train[ix].reshape((28, 28)), cmap='gray')
    plt.subplot(1,2,2)
    plt.title('PCA Regenration')
    plt.imshow(pca_regenerations[ix].reshape((28, 28)), cmap='gray')
    plt.savefig(('#' + str(ix) + ': PCA-Regenerated ' + LABELS[data[ix, 0]] + '.png'), dpi=326)
    plt.close()
