import time
import numpy as np
import scipy.signal
import math
import torch
import torch.nn as nn
from bisect import bisect_left

def r_to_e(r,tab):
	low = 0
	high = len(tab)-1
	while low<high:
		mid = int((low+high)/2)
		#print("%d:%d:%d"%(low,mid,high))
		if (r<tab[mid]):
			low = mid+1
		else:
			high = mid-1
	return tab[mid]
def r_to_e_lib(r,tab):
	i = bisect_left(tab, r)
	j = np.searchsorted(tab, r)
	print("lib i:",i,j)
	if(i>=0):
		return tab[i-1]
	else:
		return tab[0]

def r_to_e_fake(r,tab):
	return tab[1]

def get_tab(a=1.2,delta=1e-3):
	N=math.ceil(abs(math.log(delta,a)))
	print("N:",N)
	tab=[]
	for i in range(0,N+1):
		tab.append(pow(a,-i))
	tab[-1]=0
	print("tab size:",len(tab))
	print("tab:\n",tab)
	return tab

a=1.2
delta=1e-2
tab = get_tab(a,delta)
#print("e:",r_to_e(0.5,tab),r_to_e_lib(0.5,tab))
loop=100
start = time.time()
for i in range(0,loop):
	e = r_to_e(0.5,tab)
end = time.time()
t0 = (end-start)
print("r_to_e used time %f:%f s"%(t0,t0/loop))
w=3
h=3
dim=3
data_in = np.random.random((dim,dim))
b_data_in = np.random.random((w,h))
print("in:\n",data_in)
kernel = np.random.random((dim,dim))
print("kernel:\n",kernel)

n = 100
##convolution
start = time.time()
for i in range(0,n):
	out = scipy.signal.convolve2d(b_data_in,kernel)
end = time.time()
t1 = (end -start)/n
print("out:\n",out)
print("=====================================================\n")
print("n = %d, scipy conv used %f s"%(n,t1))
print("=====================================================\n")

##pytorch convolution
tensor = torch.FloatTensor(b_data_in)
tensor = tensor.unsqueeze_(0)
tensor = tensor.unsqueeze_(0)
print("tensor shape:",tensor.size())
print("tensor:\n",tensor)
conv1 = nn.Conv2d(1,1,dim,stride=1)
start = time.time()
for i in range(0,n):
	#conv = torch.nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(dim,dim),stride=1,padding=0,dilation=1,groups=1,bias=False)
	out = conv1(tensor)
end = time.time()
t2 = (end -start)/n
print("out:\n",out,out.size())
print("=====================================================\n")
print("n = %d, pytorch conv used %f s"%(n,t2))
print("=====================================================\n")

##look up table: real to element
for element in data_in.flat:
	element = r_to_e(element, tab)
	print("element in date_in:",element)

print("r_to_e data_in\n",data_in)

for element in kernel.flat:
	element = r_to_e(element, tab)

print("r_to_e kenel\n",kernel)

##add
start = time.time()
for i in range(0,n):
	for element in b_data_in.flat:
		element = r_to_e(element, tab)
		#element = r_to_e_fake(element, tab)
		#element = 1
	for j in range(0, w-dim+1):
		for j in range(0, h-dim+1):
			out = data_in+kernel
			result = np.sum(out)
end = time.time()
t3 = (end -start)/n
print("out:\n",out)
print("result:\n",result)
print("=====================================================\n")
print("n = %d, scipy conv used %f s"%(n,t1))
print("n = %d, pytorch conv used %f s"%(n,t2))
print("n = %d, look up table add used %f s"%(n,t3))
print("tab len:%d,speed up %f:%f times"%(len(tab),t1/t3,t2/t3))
print("=====================================================\n")
