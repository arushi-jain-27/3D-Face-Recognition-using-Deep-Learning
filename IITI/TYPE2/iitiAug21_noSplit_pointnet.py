# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017

@author: Gary
"""

import numpy as np
import os
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
from sklearn import metrics
import h5py
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def mat_mul(A, B):
    return tf.matmul(A, B)

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

# number of points in each sample
num_points = 25000

# number of categories
classes = 170

# define optimizer
adam = optimizers.Adam(lr=0.05, decay=0.7)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
input_points = Input(shape=(num_points, 3))
x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, 3))(input_points)
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
c = Dense(classes, activation='softmax')(c)
prediction = Flatten()(c)
# --------------------------------------------------end of pointnet

# print the model summary
model = Model(inputs=input_points, outputs=prediction)
print(model.summary())

# load train points and labels
path = os.path.dirname(os.path.realpath('__file__'))

train_points, train_labels = load_h5(os.path.join(path, 'data_aug21_random_iiti_no_split.h5'))
print(len(train_labels)/21)
#points_r = points.reshape(50, 21, 25000, 3)
#train_points_r, test_points_r,  train_labels_r, test_labels_r = train_test_split (train_points,train_labels) 
#print(train_labels_r)

label_count = np.zeros(classes)
k=1
j=0
for i in range(1, len(train_labels)):
	if train_labels[i]!=train_labels[i-1]:
		label_count[j]=k
		k = 1
		j = j+1
	else:
		k = k+1
label_count[j]=k
print(label_count)
label_count_cum = np.zeros(classes)
total = int(label_count[0]*2/3)
print(total)
for i in range(1, len(label_count)):
	total = total+int(label_count[i]*2/3)
	#label_count_cum[i] += label_count_cum[i-1]
	print(total)

total = int(total)
print(total)
count1 = 0
count2 = 0
label_count_cum[0] = label_count[0]
for i in range(1, len(label_count)):
	label_count_cum[i] = label_count_cum[i-1]+label_count[i]
total_samples = int(label_count_cum[classes-1])
train_points_r = np.zeros([total_samples,num_points,3])
test_points_r = np.zeros([total_samples,num_points,3])
train_labels_r = np.zeros(total_samples)
test_labels_r = np.zeros(total_samples)
print(label_count)
for i in range(0,classes):
	for j in range(0,int(label_count[i])):
		if(j<=int(2*label_count[i]/3)):
			train_points_r[count1] = train_points[int(label_count_cum[i-1]+j) if i>0 else 0,:,:]
			train_labels_r[count1] = train_labels[int(label_count_cum[i-1]+j) if i>0 else 0]
			count1 = count1+1
		else:
			test_points_r[count2] = train_points[int(label_count_cum[i-1]+j) if i>0 else 0,:,:]
			test_labels_r[count2] = train_labels[int(label_count_cum[i-1]+j) if i>0 else 0]
			count2 = count2+1

train_points_r = train_points_r[0:count1,:,:]
test_points_r = test_points_r[0:count2,:,:]
train_labels_r = train_labels_r[0:count1]
test_labels_r = test_labels_r[0:count2]
Y_train = np_utils.to_categorical(train_labels_r, classes)
Y_test = np_utils.to_categorical(test_labels_r, classes)
print(count1)
print(count2)
# compile classification model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

for i in range(1,5):
    if i!=0: 
    	w_name = 'iitiPointnetAug21_noSplit'
    	model.load_weights(w_name, by_name=True)
    model.fit(train_points_r, Y_train, batch_size=50, epochs=1, shuffle=True, verbose=1)
    w_name = 'iitiPointnetAug21_noSplit'
    model.save_weights(w_name)
    print(i, 'weights_saved', w_name)

intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('feature_dense').output)
train_features = intermediate_layer_model.predict(train_points_r)
test_features = intermediate_layer_model.predict(test_points_r)

with open('iiti_trainAug21_noSplit.pkl', 'wb') as f:
	pickle.dump(train_features, f)

with open('iiti_testAug21_noSplit.pkl', 'wb') as f:
	pickle.dump(test_features, f)

with open('iiti_trainlAug21_noSplit.pkl', 'wb') as f:
	pickle.dump(train_labels_r, f)

with open('iiti_testlAug21_noSplit.pkl', 'wb') as f:
	pickle.dump(test_labels_r, f)

#w_name = 'bosPointnetFeatures'
#model.load_weights(w_name, by_name=True)
score = model.evaluate(test_points_r, Y_test, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
"""
#y_score = model.predict(test_points_r)
#plot_roc(k,Y_test,y_score)

#print(Y_test)
#print(model.predict(test_points_r))
#print(metrics.confusion_matrix(Y_test, model.predict(test_points_r)))
# score the model
#score = model.evaluate(test_points_r, Y_test, verbose=1)
#print('Test loss: ', score[0])
#print('Test accuracy: ', score[1])
"""
