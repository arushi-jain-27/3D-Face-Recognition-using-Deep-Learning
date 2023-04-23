
import h5py
import numpy as np
import array, os
import math

num_select = 15000
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
train_path = os.path.join(path, "asc_mtlb")
samplenames = sorted([d for d in os.listdir(train_path)])
f = h5py.File("./data_aug_random_und_full_matlab.h5", 'w')
lCount = 1
c = samplenames[0].split('d')[0]
for i in range(1,len(samplenames)):
	if(samplenames[i-1].split('d')[0]!=samplenames[i].split('d')[0]):
		lCount += 1

print(lCount)	
a_data = np.zeros((lCount*21, num_select, 3))
labels=np.zeros((lCount*21, 1), dtype = np.uint16)
count = 0

i = 0
while i<len(samplenames):
	c = samplenames[i].split('d')[0]
	freq = 0
	while i<len(samplenames) and samplenames[i].split('d')[0]==c:
		freq = freq+1
		i = i+1
	print("freq"+str(freq))
	if (freq==1):
		continue
	samples = 0
	j = 1
	fr = freq
	while samples < 21 and fr:
		s = os.path.join(train_path, samplenames[i-j])
		#print(s)
		points = read_asc(s)
		j = (j % freq) + 1
		if (points == None):
			print(samplenames[i-j])	
			fr = fr-1
			continue
		samples = samples + 1
		for k in range(0,num_select):
			a_data[count,k] = [points[k][0],points[k][1],points[k][2]]
		labels[count] = math.floor(count/21)
		#print(count/21)
		print(labels[count])
		count = count+1

print(labels[0:count])
data = f.create_dataset("data", data = a_data[0:count])
#pid = f.create_dataset("pid", data = a_pid)
label = f.create_dataset("label", data = labels[0:count])
print(data.shape)
print(label.shape)
#print(label.data)
