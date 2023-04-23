import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
    

path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(path, "asc")
save_dir = os.path.join(path, "asc")
facenames = sorted([d for d in os.listdir(data_dir)])

for k in range(0, len(facenames)):
	path = os.path.join(data_dir, facenames[k])
	print(facenames[k])
	data = pd.read_csv(path, delimiter="  ")
	X = data.values
	X = X.reshape(-1, 3)
	#name = facenames[k].split(".")[0]+'_21.asc'
	np.savetxt(os.path.join(save_dir, facenames[k]), X, delimiter=' ', fmt = '%f')
	#rotate_point_cloud(facenames[k], X)


