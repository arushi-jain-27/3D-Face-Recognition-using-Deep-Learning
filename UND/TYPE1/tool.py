#!/usr/bin/env python
# coding: utf-8

# In[2]:


# In[1]:


# Import the required modules
import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout, Concatenate
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization, Convolution2D
from keras.layers import Lambda
from keras.layers.merge import concatenate
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2
from keras.utils import np_utils
import keras.backend as K
import h5py
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Comment this to run on CPU and Uncomment to run on GPU
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn import metrics
import h5py
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import pickle
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import array, os
import math


# In[2]:


def mat_mul(A, B):
    return tf.matmul(A, B)


num_points = 25000
k = 170
path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "phase3data170samples3dallfinal")

def read_asc(filename):

    f = open(filename)
    All_points = []
    selected_points = []
    while True:
        new_line = f.readline()
        new_line = new_line.strip()
        x = new_line.split(' ')
        if len(x) == 3:
            A = np.array(x[0:3], dtype='float32')
            All_points.append(A)
        else:
            break
    # if the numbers of points are less than 2000, extent the point set
    if len(All_points) < (num_points):
        print('none detected')
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_points, replace=False)
    for i in range(len(index)):
        selected_points.append(All_points[index[i]])
    return selected_points  # return N*3 array

def h5(class_1, sample_1, count ):
    d_path = os.path.join(train_path, subjectnames[class_1], "3D")
    face = sorted([d for d in os.listdir(d_path) if '.asc' in d and 'f0' in d])[sample_1]
    s = os.path.join(d_path, face)
    points = read_asc(s)

    for i in range(0, 25000):
        a_data[count, i] = [ points[i][0], points[i][1], points[i][2]]
    labels[count] = class_1


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    return (f['data'][:], f['label'][:])


def get_data_train(features):
    dim1 = 1
    dim2 = 256
    x_pair = np.zeros([1, 2, 1, dim1, dim2])
    img1 = features[0]
    img2 = features[1]
    x_pair[0, 0, 0, :, :] = img1
    x_pair[0, 1, 0, :, :] = img2
    return x_pair





class_1 = int(input("Class 1:"))
sample_1 = int(input("Sample 1:"))
class_2 = int(input("Class 2:"))
sample_2 = int(input("Sample 2:"))

"""
subjectnames = sorted([d for d in os.listdir(train_path)])
f = h5py.File("./tool.h5", 'w')
a_data = np.zeros((2, num_points, 3))
labels = np.zeros((2, 1), dtype=np.uint8)




h5(class_1, sample_1, 0)
h5(class_2, sample_2, 1)
data = f.create_dataset("data", data=a_data)
label = f.create_dataset("label", data=labels)
test_points, test_labels = load_h5(os.path.join(path, 'tool.h5'))




adam = optimizers.Adam(lr=0.05, decay=0.7)
input_points = Input(shape=(num_points, 3))
x = Convolution1D(64, 1, activation='relu',input_shape=(num_points, 3))(input_points)
x = BatchNormalization()(x)
x = Convolution1D(128, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = Convolution1D(1024, 1, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=num_points)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
input_T = Reshape((3, 3))(x)

# forward net
g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
g = BatchNormalization()(g)

# feature transform net
f = Convolution1D(64, 1, activation='relu')(g)
f = BatchNormalization()(f)
f = Convolution1D(128, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = Convolution1D(1024, 1, activation='relu')(f)
f = BatchNormalization()(f)
f = MaxPooling1D(pool_size=num_points)(f)
f = Dense(512, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(256, activation='relu')(f)
f = BatchNormalization()(f)
f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
feature_T = Reshape((64, 64))(f)

# forward net
g = Lambda(mat_mul, arguments={'B': feature_T})(g)
g = Convolution1D(64, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(128, 1, activation='relu')(g)
g = BatchNormalization()(g)
g = Convolution1D(1024, 1, activation='relu')(g)
g = BatchNormalization()(g)

# global_feature
global_feature = MaxPooling1D(pool_size=num_points)(g)

# point_net_cls
c = Dense(512, activation='relu')(global_feature)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(256, activation='relu', name='feature_dense')(c)
c = BatchNormalization()(c)
c = Dropout(rate=0.7)(c)
c = Dense(k, activation='softmax')(c)
prediction = Flatten()(c)

model = Model(inputs=input_points, outputs=prediction)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
w_name = 'iitiPointnetFull'
model.load_weights(w_name, by_name=True)
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('feature_dense').output)
test_features = intermediate_layer_model.predict(test_points)
"""
with open('und_test_full.pkl', 'rb') as f:
	test_points = pickle.load(f)


w_name = 'test_trainund'

throld = 0.85
	
test_features = []
test_features.append(test_points[class_1*6+sample_1])
test_features.append(test_points[class_2*6+sample_2])
	




print (len(test_features))
x_test = get_data_train(test_features)


out1 = Input(shape=(1, 256))
out2 = Input(shape=(1, 256))
x1 = Lambda(lambda x: x[0] * x[1])([out1, out2])
x2 = Lambda(lambda x: x[0] + x[1])([out1, out2])
x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([out1, out2])
x4 = Lambda(lambda x: K.square(x))(x3)
x = Concatenate()([x1, x2, x3, x4])
x = Reshape((4, 256, 1), name='reshape1')(x)

# Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
x = Convolution2D(32, (4, 1), activation='relu', padding='valid')(x)
x = Reshape((256, 32, 1))(x)
x = Convolution2D(1, (1, 32), activation='linear', padding='valid')(x)
x = Flatten(name='flatten')(x)

# Weighted sum implemented as a Dense layer.
prediction = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average')(x)

model = Model(input=[out1, out2], output=prediction)
model.summary()
epochs = 1
adam = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
model.load_weights(w_name, by_name=True)
pred = model.predict([x_test[:, 0].reshape(x_test.shape[0], 1, 256), x_test[:, 1].reshape(x_test.shape[0], 1, 256)], verbose=1)
print (pred)

if pred>throld:
    print ("Genuine Pair")
else:
    print ("Imposite Pair")







