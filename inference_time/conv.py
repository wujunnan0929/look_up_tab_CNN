import time
import numpy as np
import scipy.signal
import math
import torch
import torch.nn as nn

def r_to_e(r,N,tab):
	for i in range(1,N+1):
		if (r<tab[i]):
			return tab[i-1]

a=1.2
delta = 1e-4
N=math.ceil(abs(math.log(delta,a)))
tab=[]
for i in range(0,N+1):
	tab.append(pow(a,-i))
tab[-1]=0
print("tab:\n",tab)

dim=3
data_in = np.random.random((dim,dim))
p_data_in = np.random.random((dim,dim,dim))
print("in:\n",data_in)
kernel = np.random.random((dim,dim))
print("kernel:\n",kernel)

##convolution
n = 10000
start = time.time()
for i in range(0,n):
	out = scipy.signal.convolve2d(data_in,kernel)
end = time.time()
t1 = end -start
print("out:\n",out)
print("%d conv used %f s"%(n,t1))

##pytorch convolution
n = 10000
tensor = torch.FloatTensor(data_in)
tensor = tensor.unsqueeze_(0)
tensor = tensor.unsqueeze_(0)
print("tensor:",tensor)
print("tensor shape:",tensor.size())
start = time.time()
for i in range(0,n):
	#conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(dim,dim),stride=1,padding=0,dilation=1,groups=1,bias=False)
	conv1 = nn.Conv2d(1,1,dim,stride=1)
	out = conv1(tensor)
end = time.time()
t2 = end -start
print("out:\n",out)
print("%d pytorch conv used %f s"%(n,t2))

##look up table: real to element
for element in data_in.flat:
	element = r_to_e(element, N, tab)

print("r_to_e data_in\n",data_in)

for element in kernel.flat:
	element = r_to_e(element, N, tab)

print("r_to_e kenel\n",kernel)

##add
start = time.time()
for i in range(0,n):
	tmp = data_in
	data_in = kernel
	kernel = tmp 
	out = data_in+kernel
result = np.sum(out)
end = time.time()
t3 = end -start
print("out:\n",out)
print("result:\n",result)
print("%d look up table add used %f s, speed up %f:%f times"%(n,t3,t1/t3,t2/t3))


