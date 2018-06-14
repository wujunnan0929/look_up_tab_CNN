import torch
import torch.nn as nn
import torch.nn.functional as Function
import numpy as np

def r_to_e(r,N,tab):
        for i in range(1,N+2):
                if (r>tab[i]):
                        return tab[i-1]

def tab_init(a = 1.4, delta = 1e-4):
	a=1.2
	delta = 1e-4
	N=math.ceil(abs(math.log(delta,a)))
	tab=[]
	for i in range(0,N+1):
		tab.append(pow(a,-i))
	tab[-1]=0
	return tab


class Conv2add(Function):
    def forward(self,input,weight,output_channel,kernel,stride,padding):
        r_to_e(weight)
        r_to_e(input)
        if padding:
            add_padding(input)
        c = input.size()[1]
        w = input.size()[2]
        h = input.size()[3]
        dim = kernel 
        ic = 0
        iw = 0
        ih = 0
        oc = 0
        while oc < output_channel:
                while ic<c:
                    while iw+dim<w:
                        while ih+dim<h:
                            in_tmp = input[ic,[iw,iw+dim],[ih,ih+dim]]
                            result_tmp = in_tmp + weight[oc]
                            r = e_to_r(result_tmp)
                            r = r.sum()
                            result[ic/dim,iw/dim,ih/dim] = r
                            ih += stride
                        iw += stride
                    ic += 1
                    result[0].sum
                    out.append.result
        return out

class Alexnet(nn.model):
    def __init__(self,,num_classes,tab):
        self.tab=tab
	super(AlexNet,self).__init__()
	self.feature = nn.Sequential(
	nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2)
	nn.ReLU(inplace = True),
	nn.MaxPool2d(kernel_size = 3, stride = 2),
	nn.Conv2d(64, 192, kernel_size=5, padding = 2),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(kernel_size = 3, stride = 2)
	nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.MaxPool2d(kernel_size = 3, stride = 2),	
	)
	self.classifier = nn.Sequential(
	nn.Dropput(),
	nn.Linear(256 * 6 * 6, 4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,4096),
	nn.ReLU(inplace=True),
	nn.Linear(4096, num_classes),	
	)

    def forward(self,input):
	x = self.feature(input)
	x = x.view(x.size(), 256*6*6)
	x = self.classifier(x)
	return x
	        
class MyAlexnet(nn.model):
    def __init__(self,,num_classes,tab):
        self.tab=tab
	super(AlexNet,self).__init__()
	self.feature = nn.Sequential(
	nn.Conv2add(3, 64, kernel_size = 11, stride = 4, padding = 2)
	nn.ReLU(inplace = True),
	nn.MaxPool2d(kernel_size = 3, stride = 2),
	nn.Conv2add(64, 192, kernel_size=5, padding = 2),
	nn.ReLU(inplace=True),
	nn.MaxPool2d(kernel_size = 3, stride = 2)
	nn.Conv2add(192, 384, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.Conv2add(384, 256, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.Conv2add(256, 256, kernel_size = 3, padding = 1),
	nn.ReLU(inplace = True),
	nn.MaxPool2d(kernel_size = 3, stride = 2),	
	)
	self.classifier = nn.Sequential(
	nn.Dropput(),
	nn.Linear(256 * 6 * 6, 4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,4096),
	nn.ReLU(inplace=True),
	nn.Linear(4096, num_classes),	
	)

    def forward(self,input):
	x = self.feature(input)
	x = x.view(x.size(), 256*6*6)
	x = self.classifier(x)
	return x
