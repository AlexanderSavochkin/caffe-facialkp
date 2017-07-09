import numpy as np
import pandas as pd
from numpy import genfromtxt
from numpy import ravel
import h5py
from numpy import (array, dot, arccos)
from numpy.linalg import norm

nr=3000

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
    return 0.00 if np.isnan(row[name]) else 1.0

y = df.drop(['Image'], axis=1)

print('#1')
print(y[nr:nr+1])

dict_certainity = {c + '_certainty':y.apply (lambda row: certainty(row, c),axis=1) for c in y.columns}
certainty = pd.DataFrame(dict_certainity)

y_imp = y.copy()
for c in y.columns:
    y_imp[c].fillna(y_imp[c].median(), inplace=True )

y_imp = y_imp.values 
y_imp = y_imp.astype(np.float32) 
y = y_imp.reshape((-1,30))

print('#2')
print(y[nr:nr+1])



certainty = certainty.values
certainty = certainty.astype(np.float32) 
certainty = certainty.reshape((-1,30))

y = y / 96

print ('Y shape', y.shape)

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

#print ('X:', X.shape)

print ('Shape', 'Labels', X.shape, y.shape)


dfp['Image'] = dfp['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
Xp = np.vstack (dfp['Image'].values) 

Xp = Xp.reshape(-1,96,96)
for i in range(len(Xp)):
       Xp[i, :, :] = image_histogram_equalization(Xp[i,:,:])[0]


Xp = Xp.astype(np.float32)
Xp = Xp/255 
Xp = Xp.reshape(-1,1,96,96)

#print ('X:', X.shape)

print ('Shape of predict', Xp.shape)



#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,labels, test_size=0.30)

X_test = X[:1600]
y_test = y[:1600]
certainty_test = certainty[:1600] 

X_train = X[1600:]
y_train = y[1600:]
certainty_train = certainty[1600:] 

print ('Train, Test shapes (X,y):', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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

# Full train data
f = h5py.File("facialkp-full-train.hd5", "w")
f.create_dataset("data", data=X,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=y,  compression="gzip", compression_opts=4)
f.create_dataset("certainty", data=certainty,  compression="gzip", compression_opts=4)
f.close()

# Predict data
f = h5py.File("facialkp-unlabeled.hd5", "w")
f.create_dataset("data", data=Xp,  compression="gzip", compression_opts=4)
f.close()

def flip_labels_lr(labels):
  (left_eye_center_x,left_eye_center_y, \
  right_eye_center_x,right_eye_center_y, \
  left_eye_inner_corner_x,left_eye_inner_corner_y, \
  left_eye_outer_corner_x,left_eye_outer_corner_y, \
  right_eye_inner_corner_x,right_eye_inner_corner_y, \
  right_eye_outer_corner_x,right_eye_outer_corner_y, \
  left_eyebrow_inner_end_x,left_eyebrow_inner_end_y, \
  left_eyebrow_outer_end_x,left_eyebrow_outer_end_y, \
  right_eyebrow_inner_end_x,right_eyebrow_inner_end_y, \
  right_eyebrow_outer_end_x,right_eyebrow_outer_end_y, \
  nose_tip_x,nose_tip_y, \
  mouth_left_corner_x,mouth_left_corner_y, \
  mouth_right_corner_x,mouth_right_corner_y, \
  mouth_center_top_lip_x,mouth_center_top_lip_y, \
  mouth_center_bottom_lip_x,mouth_center_bottom_lip_y) = labels

  left_eye_center_x_fl, left_eye_center_y_fl = (1.0 - right_eye_center_x), right_eye_center_y
  right_eye_center_x_fl,right_eye_center_y_fl = (1.0 - left_eye_center_x), left_eye_center_y
  left_eye_inner_corner_x_fl,left_eye_inner_corner_y_fl = (1.0 - right_eye_inner_corner_x),right_eye_inner_corner_y
  left_eye_outer_corner_x_fl,left_eye_outer_corner_y_fl = (1.0 - right_eye_outer_corner_x),right_eye_outer_corner_y
  right_eye_inner_corner_x_fl,right_eye_inner_corner_y_fl = (1.0 - left_eye_inner_corner_x),left_eye_inner_corner_y
  right_eye_outer_corner_x_fl,right_eye_outer_corner_y_fl = (1.0 - left_eye_outer_corner_x),left_eye_outer_corner_y
  left_eyebrow_inner_end_x_fl,left_eyebrow_inner_end_y_fl = (1.0 - right_eyebrow_inner_end_x),right_eyebrow_inner_end_y
  left_eyebrow_outer_end_x_fl,left_eyebrow_outer_end_y_fl = (1.0 - right_eyebrow_outer_end_x),right_eyebrow_outer_end_y
  right_eyebrow_inner_end_x_fl,right_eyebrow_inner_end_y_fl = (1.0 - left_eyebrow_inner_end_x),left_eyebrow_inner_end_y
  right_eyebrow_outer_end_x_fl,right_eyebrow_outer_end_y_fl = (1.0 - left_eyebrow_outer_end_x),left_eyebrow_outer_end_y
  nose_tip_x_fl,nose_tip_y_fl = (1.0 - nose_tip_x),nose_tip_y
  mouth_left_corner_x_fl,mouth_left_corner_y_fl = (1.0 - mouth_right_corner_x),mouth_right_corner_y
  mouth_right_corner_x_fl,mouth_right_corner_y_fl = (1.0 - mouth_left_corner_x),mouth_left_corner_y
  mouth_center_top_lip_x_fl,mouth_center_top_lip_y_fl = (1.0 - mouth_center_top_lip_x),mouth_center_top_lip_y
  mouth_center_bottom_lip_x_fl,mouth_center_bottom_lip_y_fl = (1.0 - mouth_center_bottom_lip_x),mouth_center_bottom_lip_y

  return np.array([left_eye_center_x_fl,left_eye_center_y_fl, \
    right_eye_center_x_fl,right_eye_center_y_fl, \
    left_eye_inner_corner_x_fl,left_eye_inner_corner_y_fl, \
    left_eye_outer_corner_x_fl,left_eye_outer_corner_y_fl, \
    right_eye_inner_corner_x_fl,right_eye_inner_corner_y_fl, \
    right_eye_outer_corner_x_fl,right_eye_outer_corner_y_fl, \
    left_eyebrow_inner_end_x_fl,left_eyebrow_inner_end_y_fl, \
    left_eyebrow_outer_end_x_fl,left_eyebrow_outer_end_y_fl, \
    right_eyebrow_inner_end_x_fl,right_eyebrow_inner_end_y_fl, \
    right_eyebrow_outer_end_x_fl,right_eyebrow_outer_end_y_fl, \
    nose_tip_x_fl,nose_tip_y_fl, \
    mouth_left_corner_x_fl,mouth_left_corner_y_fl, \
    mouth_right_corner_x_fl,mouth_right_corner_y_fl, \
    mouth_center_top_lip_x_fl,mouth_center_top_lip_y_fl, \
    mouth_center_bottom_lip_x_fl,mouth_center_bottom_lip_y_fl])

Xflipped = X.copy()
Yflipped = y.copy()
certainty_flipped = certainty.copy()
for i in range(X.shape[0]):
  Xflipped[i,0,:,:] = np.fliplr(Xflipped[i,0,:,:])
  Yflipped[i,:] = flip_labels_lr(Yflipped[i,:])
  certainty_flipped[i,:] = flip_labels_lr(certainty_flipped[i,:])

f = h5py.File("facialkp-flipped-train.hd5", "w")
f.create_dataset("data", data=Xflipped,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=Yflipped,  compression="gzip", compression_opts=4)
f.create_dataset("certainty", data=certainty_flipped,  compression="gzip", compression_opts=4)
f.close()

Xextended = np.vstack((X,Xflipped))
Yextended = np.vstack((y,Yflipped))
Cextended = np.vstack((certainty,certainty_flipped))

f = h5py.File("facialkp-extended-train.hd5", "w")
f.create_dataset("data", data=Xextended,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=Yextended,  compression="gzip", compression_opts=4)
f.create_dataset("certainty", data=Cextended,  compression="gzip", compression_opts=4)
f.close()
