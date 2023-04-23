import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os

path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(path, "asc")
facenames = [d for d in os.listdir(path) if '.obj' in d]

for k in range(0, len(facenames)):
	path = os.path.dirname(os.path.realpath(__file__))
	path = os.path.join(path, facenames[k])
	path2 = os.path.join(data_dir, facenames[k].replace(".obj",'.asc'))
	f = open(path, 'r')
	f2 = open(path2,'w')
	new_line = f.readline()
	new_line = f.readline()
	while True:
		new_line = f.readline()
		new_line = new_line.strip()
		x = new_line.split(' ')
		#print(x)
		if x[0] == 'v':
			data = x[1] + " "+x[2] + " "+x[3]+"\n"
			#print(data)
			f2.write(data)
		else:
			break

