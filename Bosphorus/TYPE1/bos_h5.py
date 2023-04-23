
import h5py
import numpy as np
import array, os
import math
num_select = 20000
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
    if len(All_points) < (num_select):
        print('none detected')
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_select, replace=False)
    for i in range(len(index)):
        selected_points.append(All_points[index[i]])
    return selected_points  # return N*3 array

path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, "asc")
samplenames = sorted([d for d in os.listdir(train_path)])
f = h5py.File("./data_aug_random_bos_neutral.h5", 'w')
a_data = np.zeros((105*21, num_select, 3))
labels=np.zeros((105*21, 1), dtype = np.uint8)
count = 0
i = 0
while i<len(samplenames):
	c = samplenames[i].split('_')[0]
	freq = 0
	while i<len(samplenames) and samplenames[i].split('_')[0]==c:
		freq = freq+1
		i = i+1
	print("freq"+str(freq))
	samples = 0
	j = 1
	while samples < 21:
		s = os.path.join(train_path, samplenames[i-j])
		#print(s)
		points = read_asc(s)
		j = (j % freq) + 1
		if (points == None):
			print(samplenames[i-j])	
			continue
		samples = samples + 1
		for k in range(0,num_select):
			a_data[count,k] = [points[k][0],points[k][1],points[k][2]]
		labels[count] = math.floor(count/21)
		#print(count/21)
		print(labels[count])
		count = count+1

print(labels)
data = f.create_dataset("data", data = a_data[0:count])
#pid = f.create_dataset("pid", data = a_pid)
label = f.create_dataset("label", data = labels[0:count])
print(data.shape)
print(label.shape)
#print(label.data)
