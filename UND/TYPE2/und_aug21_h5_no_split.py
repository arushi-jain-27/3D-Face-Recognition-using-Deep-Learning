
import h5py
import numpy as np
import array, os
import math

num_select = 4000

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
    print(len(All_points)) 
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
f1 = h5py.File("./data_aug21_und_orig_noSplit.h5", 'w')
lCount = 1
c = samplenames[0].split('d')[0]
for i in range(1,len(samplenames)):
	if(samplenames[i-1].split('d')[0]!=samplenames[i].split('d')[0]):
		lCount += 1

print(lCount)	
a_data1 = np.zeros((lCount*21*8, num_select, 3))
labels1=np.zeros((lCount*21*8, 1), dtype = np.uint16)
count1 = 0
label = 0
i = 0
while i<len(samplenames):
	if(i and samplenames[i-1].split('d')[0]!=samplenames[i].split('d')[0]):
		label = label+1
	samples = 0
	while samples < 21:
		s = os.path.join(train_path, samplenames[i])
		#print(s)
		points = read_asc(s)
		if (points == None):
			print(samplenames[i])	
			break
		samples = samples + 1
		for k in range(0,num_select):
			a_data1[count1,k] = [points[k][0],points[k][1],points[k][2]]
		labels1[count1] = label
		count1 = count1+1
	i = i+1

print(labels1[0:count1])
data1 = f1.create_dataset("data", data = a_data1[0:count1])
#pid = f.create_dataset("pid", data = a_pid)
label1 = f1.create_dataset("label", data = labels1[0:count1])
print(data1.shape)
print(label1.shape)
