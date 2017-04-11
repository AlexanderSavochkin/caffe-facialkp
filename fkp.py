import numpy as np
import pandas as pd
from numpy import genfromtxt
from numpy import ravel
import pylab as pl
from skimage import transform
import h5py
from sklearn import cross_validation
import uuid
import random
from skimage import io, exposure, img_as_uint, img_as_float
from numpy import (array, dot, arccos)
from numpy.linalg import norm

df = pd.read_csv('training.csv',header=0)
dfp = pd.read_csv('test.csv',header=0)

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def certainty(row, name):
    return 0.001 if np.isnan(row[name]) else 1.0

y = df.drop(['Image'], axis=1)

dict_certainity = {c + '_certainty':y.apply (lambda row: certainty(row, c),axis=1) for c in y.columns}
certainty = pd.DataFrame(dict_certainity)

y_imp = y.copy()
for c in y.columns:
    y_imp.fillna(y_imp[c].median(), inplace=True )

y_imp = y_imp.values 
y_imp = y_imp.astype(np.float32) 
y = y_imp.reshape((-1,30))

certainty = certainty.values
certainty = certainty.astype(np.float32) 
certainty = certainty.reshape((-1,30))

y = y / 96

print 'Y shape', y.shape

# Extracting Images

df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
X = np.vstack (df['Image'].values) 

X = X.reshape(-1,96,96)

# Histogram equalization
for i in range(len(X)):
       X[i, :, :] = image_histogram_equalization(X[i, :,:])[0]


X = X.astype(np.float32)
X = X/255 
X = X.reshape(-1,1,96,96)

print 'X:', X.shape

print 'Shape', 'Labels', X.shape, y.shape

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,labels, test_size=0.30)

X_test = X[:1600]
y_test = y[:1600]
certainty_test = certainty[:1600] 

X_train = X[1600:]
y_train = y[1600:]
certainty_train = certainty[1600:] 

print 'Train, Test shapes (X,y):', X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Train data
f = h5py.File("facialkp-train.hd5", "w")
f.create_dataset("data", data=X_train,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=y_train,  compression="gzip", compression_opts=4)
f.create_dataset("certainty", data=certainty_train,  compression="gzip", compression_opts=4)
f.close()

#Test data

f = h5py.File("facialkp-test.hd5", "w")
f.create_dataset("data", data=X_test,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=y_test,  compression="gzip", compression_opts=4)
f.create_dataset("certainty", data=certainty_test,  compression="gzip", compression_opts=4)
f.close()


