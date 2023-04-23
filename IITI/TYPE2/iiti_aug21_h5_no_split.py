
import h5py
import numpy as np
import array, os
import math

def read_asc(filename):
    num_select = 25000
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
    if len(All_points) < (num_select):
        print('none detected')
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_select, replace=False)
    for i in range(len(index)):
        selected_points.append(All_points[index[i]])
    return selected_points  # return N*3 array

path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "phase3data170samples3dallfinal")
subjectnames = sorted([d for d in os.listdir(train_path)])
f = h5py.File("./data_aug21_random_iiti_no_split.h5", 'w')
a_data = np.zeros((len(subjectnames)*21*4, 25000, 3))
labels=np.zeros((len(subjectnames)*21*4, 1), dtype = np.uint8)
count = 0
for l in range(0,len(subjectnames)):
	d_path = os.path.join(train_path, subjectnames[l], "3D")
	facenames = sorted([d for d in os.listdir(d_path) if '.asc' in d and 'f0' in d])
	print(facenames)
	i = 0
	while i<len(facenames):
		samples = 0
		while samples < 21:
			s = os.path.join(d_path, facenames[i])
			#print(s)
			points = read_asc(s)
			if (points == None):
				break
			samples = samples + 1
			for k in range(0,25000):
				a_data[count,k] = [points[k][0],points[k][1],points[k][2]]
			labels[count] = l
			count = count+1
		i = i+1

print(labels[0:count])
data = f.create_dataset("data", data = a_data[0:count])
#pid = f.create_dataset("pid", data = a_pid)
label = f.create_dataset("label", data = labels[0:count])
print(data.shape)
print(label.shape)
#print(label.data)
