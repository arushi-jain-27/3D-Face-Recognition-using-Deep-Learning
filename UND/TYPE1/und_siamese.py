#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[1]:


#Import the required modules
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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #Comment this to run on CPU and Uncomment to run on GPU
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
from CMC import CMC

# In[2]:


def mat_mul(A, B):
    return tf.matmul(A, B)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def plot_roc(y_test,y_score):

	fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.ravel(), y_score.ravel())
	auc_keras = auc(fpr_keras, tpr_keras)
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras), color = "green")
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve for UND Dataset')
	plt.legend(loc='best')
	plt.savefig("roc_und.png")

def plot_far_frr(pred,labels):

        n_correct=0
        fpr = []
        fnr = []
        throld = []
        predt = np.zeros(pred.shape[0])
        i = -0.01
        while i < 1.01:
            for j in range(pred.shape[0]):
        
                if(pred[j]>i):
                    predt[j] = 1
                else:
                    predt[j] = 0
            tn, fp, fn, tp = confusion_matrix(labels,predt).ravel()
            #print(tn)
            #print(fp)
            #print(fn)
            #print(tp)
            fpr.append(fp/(fp+tn+fn+tp))
            fnr.append(fn/(fn+tp+fp+tn))
            throld.append(i+0.01)
            i = i+0.01
        plt.figure(2)
        plt.plot(throld, fpr, color = "lime",label="FAR Curve")
        plt.plot(throld, fnr, color = "teal", label="FRR Curve")
        plt.xlabel('Thresholds')
        plt.ylabel('FAR/FRR')
        plt.title('FAR FRR curve for UND Dataset')
        plt.legend(loc='best')
        plt.ylim(0,0.1)
        #plt.xlim(-0.001,1.001)
        plt.grid(True)
        plt.savefig("far_frr_und_s.png")


def plot_hist(pred, labels):
	genuine = []
	imposite = []
	for i in range(0,len(labels)):
		if(labels[i]==0):
			imposite.append(pred[i])
		else:
			genuine.append(pred[i])

	legend = ['Genuine','Imposite']
	bins = 50
	plt.figure(3)
	plt.hist([genuine, imposite],bins, color=["lime","teal"], log="True")
	plt.xlabel("Predicted probabilities")
	plt.ylabel("Frequency(Log Scale)")
	plt.legend(legend)
	plt.xticks(np.arange(0, 1.1, step=0.1))
	#plt.yticks(np.arange(0, 100))
	plt.title('Genuine Imposite Score Distribution for UND Dataset')
	plt.savefig("gen_imp_und_log.png")
	plt.figure(4)
	plt.hist([genuine, imposite],bins, color=["lime","teal"])
	plt.xlabel("Predicted probabilities")
	plt.ylabel("Frequency")
	plt.legend(legend)
	plt.xticks(np.arange(0, 1.1, step=0.1))
	#plt.yticks(np.arange(0, 100))
	plt.title('Genuine Imposite Score Distribution for UND Dataset')
	plt.savefig("gen_imp_und.png")
	"""
	genuine = (sorted(genuine))
	imposite = (sorted(imposite))
	gen_freq = np.zeros(11)
	imp_freq = np.zeros(11)
	for i in range(0,len(genuine)):
		gen_freq[int(genuine[i]*10)] += 1
	for i in range(0,len(imposite)):
		imp_freq[int(imposite[i]*10)] += 1
	print(gen_freq)
	print(imp_freq)
	"""

def plot_cmc(correct,incorrect):
	accuracy = np.zeros(257)
	for i in range(0,classes):
		accuracy[i] = correct[i]/(correct[i]+incorrect[i])
	accuracy = sorted(accuracy)
	#print(len(accuracy))
	#accuracy = sorted(accuracy)
	#print(len(accuracy))
	cmc_dict ={'Rank 1 accuracy': accuracy}
	cmc1 = CMC(cmc_dict)
	cmc1.save(title = 'CMC Curve for UND Dataset', filename='cmc_und_s',
         rank=257,xlabel='Rank Score', ylabel='Recognition Rate', 
         show_grid=False, format='png')


def initialize_weights(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, name=None):
    """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    """
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


# In[3]:


# number of points in each sample
#num_points = 25000

#Place the data in PrepData folder or change the Name as required
path = os.path.dirname(os.path.realpath('__file__'))
# test_path = os.path.join(path, "PrepData_test")


# In[4]:


with open('und_train_full.pkl', 'rb') as f:
	train_points = pickle.load(f)

with open('und_test_full.pkl', 'rb') as f:
	test_points = pickle.load(f)

with open('und_trainl_full.pkl', 'rb') as f:
	train_labels = pickle.load(f)

with open('und_testl_full.pkl', 'rb') as f:
	test_labels = pickle.load(f)

classes = 257
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
freq_count = label_count.copy()
for i in range(1, len(label_count)):
	label_count[i] += label_count[i-1]

print(label_count)

label_count_test = np.zeros(classes)
k=1
j=0
for i in range(1, len(test_labels)):
	if test_labels[i]!=test_labels[i-1]:
		label_count_test[j]=k
		k = 1
		j = j+1
	else:
		k = k+1
label_count_test[j]=k
freq_count_test = label_count_test.copy()
for i in range(1, len(label_count_test)):
	label_count_test[i] += label_count_test[i-1]

print(label_count_test)
#print(train_labels)

#data['train'] = train_points.reshape(10, 21, 25000, 3) #50 - number of categories and 21 is number of training samples of each category
#categories['train'] = train_labels
#print(categories)
#with open('bos_rotated_ok.pkl', 'rb') as f:
#	out = pickle.load(f)
#out = out.reshape(1050,4096)
# In[6]:



# total_sample_size = 10000

def get_data_train(total_sample_size, categories,out,label_count, freq_count):
    """
    count=0
    dim1 = 1
    dim2 = 256
    x_genuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])#2 is for the pairs
    y_genuine = np.zeros([total_sample_size, 1])
    z_genuine = np.zeros([total_sample_size, 2, 1])

    for i in range(categories):
        for j in range(int(freq_count[i])):
            for k in range(j+1,int(freq_count[i])):
                img1 = out[(int(label_count[i-1]) if i>0 else 0) +j]
                img2 = out[(int(label_count[i-1]) if i>0 else 0) +k]

            #store the images to the initialized numpy array
                x_genuine_pair[count, 0, 0, :, :] = img1
                x_genuine_pair[count, 1, 0, :, :] = img2

            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
                y_genuine[count] = 1
            
            #We assign the actual category here. (genuine pair)
                z_genuine[count, 0] = i
                z_genuine[count, 1] = i
            
                count += 1  
    """
    count=0
    dim1 = 1
    dim2 = 256
    x_genuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])#2 is for the pairs
    y_genuine = np.zeros([total_sample_size, 1])
    z_genuine = np.zeros([total_sample_size, 2, 1])

    for i in range(categories):
        for j in range(int(total_sample_size/categories)):
            ind1 = 0
            ind2 = 0

            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(label_count[i-1] if i>0 else 0  ,label_count[i]-1)
                ind2 = np.random.randint(label_count[i-1] if i>0 else 0  ,label_count[i]-1)

            # read the two images
            img1 = out[ind1]
            img2 = out[ind2]

            #store the images to the initialized numpy array
            x_genuine_pair[count, 0, 0, :, :] = img1
            x_genuine_pair[count, 1, 0, :, :] = img2

            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            
            #We assign the actual category here. (genuine pair)
            z_genuine[count, 0] = i
            z_genuine[count, 1] = i
            
            count += 1
    
    print(count)
    total_sample_size = count
    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    z_imposite = np.zeros([total_sample_size, 2, 1])
    
    for i in range(int(total_sample_size)):
        while True:
            ind1 = np.random.randint(categories)
            ind2 = np.random.randint(categories)
            if ind1 != ind2:
                break

            #prev1 = label_count[ind1-1] if ind1>0 else 0
            #prev2 = label_count_test[ind2-1] if ind2>0 else 0
            #count1 = label_count[ind1]-prev1
            #count2 = label_count_test[ind2]-prev2
            #print(count1)
        #m = 0
        m = np.random.randint(label_count[ind1-1] if ind1>0 else 0  ,label_count[ind1]-1)
            #print(ind1)
            #print(m)
            #print(prev1)	
        img1 = out[int(m)]
            #m = 0
        m = np.random.randint(label_count[ind2-1] if ind2>0 else 0  ,label_count[ind2]-1)
        img2 = out[int(m)]

        x_imposite_pair[count, 0, 0, :, :] = img1
        x_imposite_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
        y_imposite[count] = 0
            
            #We assign the actual category here. (imposite pair)
        z_imposite[count, 0] = ind1
        z_imposite[count, 1] = ind2
        count += 1

    print(count)
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_genuine_pair, x_imposite_pair], axis=0)
    Y = np.concatenate([y_genuine, y_imposite], axis=0)
    Z = np.concatenate([z_genuine, z_imposite], axis=0)
    
    return X, Y, Z


def get_data_test(total_sample_size, categories=classes):
    count=0
    dim1 = 1
    dim2 = 256
    x_genuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])#2 is for the pairs
    y_genuine = np.zeros([total_sample_size, 1])
    z_genuine = np.zeros([total_sample_size, 2, 1])

    for i in range(categories):
        for j in range(int(total_sample_size/categories)):
            ind1 = 0
            ind2 = 0

            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(label_count[i-1] if i>0 else 0  ,label_count[i]-1)
                ind2 = np.random.randint(label_count_test[i-1] if i>0 else 0  ,label_count_test[i]-1)

            # read the two images
            img1 = train_points[ind1]
            img2 = test_points[ind2]

            #store the images to the initialized numpy array
            x_genuine_pair[count, 0, 0, :, :] = img1
            x_genuine_pair[count, 1, 0, :, :] = img2

            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            
            #We assign the actual category here. (genuine pair)
            z_genuine[count, 0] = i
            z_genuine[count, 1] = i
            
            count += 1

    print(count)
    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    z_imposite = np.zeros([total_sample_size, 2, 1])
    
    for i in range(int(total_sample_size)):
            #read images from different directory (imposite pair)
        while True:
            ind1 = np.random.randint(categories)
            ind2 = np.random.randint(categories)
            if ind1 != ind2:
                break

            #prev1 = label_count[ind1-1] if ind1>0 else 0
            #prev2 = label_count_test[ind2-1] if ind2>0 else 0
            #count1 = label_count[ind1]-prev1
            #count2 = label_count_test[ind2]-prev2
            #print(count1)
        #m = 0
        m = np.random.randint(label_count[ind1-1] if ind1>0 else 0  ,label_count[ind1]-1)
            #print(ind1)
            #print(m)
            #print(prev1)	
        img1 = train_points[int(m)]
            #m = 0
        m = np.random.randint(label_count_test[ind2-1] if ind2>0 else 0  ,label_count_test[ind2]-1)
        img2 = test_points[int(m)]

        x_imposite_pair[count, 0, 0, :, :] = img1
        x_imposite_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
        y_imposite[count] = 0
            
            #We assign the actual category here. (imposite pair)
        z_imposite[count, 0] = ind1
        z_imposite[count, 1] = ind2
        count += 1


    print(count)
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_genuine_pair, x_imposite_pair], axis=0)
    Y = np.concatenate([y_genuine, y_imposite], axis=0)
    Z = np.concatenate([z_genuine, z_imposite], axis=0)
    
    return X, Y, Z

# In[ ]:


x_train, y_train, labels1 = get_data_train(70000, classes,train_points,label_count, freq_count)
x_test, y_test, labels2 = get_data_test(30000)

# In[ ]:


#x_train, x_test, y_train, y_test, labels1, labels2 = train_test_split(X, Y, Z, test_size=.3)


# In[8]:


out1 = Input(shape=(1,256))
out2 = Input(shape=(1,256))
x1 = Lambda(lambda x: x[0] * x[1])([out1, out2])
x2 = Lambda(lambda x: x[0] + x[1])([out1, out2])
x3 = Lambda(lambda x: K.abs(x[0] - x[1]))([out1, out2])
x4 = Lambda(lambda x: K.square(x))(x3)
x = Concatenate()([x1, x2, x3, x4])
x = Reshape((4, 256, 1), name='reshape1')(x)
print(x.shape)
# Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
x = Convolution2D(32, (4, 1), activation='relu', padding='valid')(x)
print(x.shape)
x = Reshape((256, 32, 1))(x)
print(x.shape)
x = Convolution2D(1, (1, 32), activation='linear', padding='valid')(x)
print(x.shape)
x = Flatten(name='flatten')(x)
print(x.shape)

# Weighted sum implemented as a Dense layer.
prediction = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average')(x)
print(prediction.shape)
model = Model(input=[out1, out2], output=prediction)
model.summary()


# In[9]:


epochs = 1
adam = Adam(lr = 0.001)


model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])


# In[ ]:


out1 = x_train[:, 0].reshape(x_train.shape[0], 1, 256)
out2 = x_train[:, 1].reshape(x_train.shape[0], 1, 256)


# In[ ]:


#Enter the name of the weight file you want to load
"""
w_name = 'train20'
model.load_weights(w_name, by_name=True)
print('weights loaded', w_name)
pred = model.predict([x_test[:, 0].reshape(x_test.shape[0], 25000, 3), x_test[:, 1].reshape(x_test.shape[0], 25000, 3)], batch_size=4, verbose=1)
n_correct=0
for i in range(x_test.shape[0]):
    if(np.where(pred>0.5, 1, 0)[i] == y_test[i]):
        n_correct+=1
        
acc=n_correct/x_test.shape[0]       
print(acc)
print("pred shape")
print(pred.shape)
plot_roc(y_test,pred)
# In[ ]:
"""

target_names = ['Imposite', 'Genuine']
for i in range(1,2):
    if i!=0: 
    	w_name = 'test_trainund'
    	model.load_weights(w_name, by_name=True)
    #model.fit([out1, out2], y_train, validation_split=.125, batch_size=4, verbose=1, shuffle=True, epochs=5)
    w_name = 'test_trainund'
    model.save_weights(w_name)
    print(i, 'weights_saved', w_name)
    if(i==1):
        correct = np.zeros(classes)
        incorrect = np.zeros(classes)
        print('The test accuracy for ', i, 'epoch is')
        #pred_train = model.predict([x_train[:, 0].reshape(x_train.shape[0], 1, 256), x_train[:, 1].reshape(x_train.shape[0], 1, 256)], batch_size=4, verbose=1)
        pred = model.predict([x_test[:, 0].reshape(x_test.shape[0], 1, 256), x_test[:, 1].reshape(x_test.shape[0], 1, 256)], batch_size=4, verbose=1)

        n_correct=0
        predt = np.zeros(x_test.shape[0])
        for j in range(x_test.shape[0]):
            #if(np.where(pred>0.5, 1, 0)[i] == y_test[i]):
             #   predt[i] = y_test[i]
              #  n_correct+=1
        
            if(pred[j]>0.5):
                predt[j] = 1
            else:
                predt[j] = 0
            if(predt[j]==y_test[j]):
                n_correct += 1
                correct[int(labels2[j,1])]+=1
            else:
                incorrect[int(labels2[j,1])]+=1


        acc=n_correct/x_test.shape[0]  
        #plot_roc(y_test,pred)
        #plot_far_frr(pred_train,y_train)  
        #plot_hist(pred_train, y_train)
        plot_cmc(correct,incorrect)   
        print(acc)
        print(confusion_matrix(y_test,predt))

        print(classification_report(y_test, predt, target_names=target_names))



# In[ ]:





# In[ ]:




