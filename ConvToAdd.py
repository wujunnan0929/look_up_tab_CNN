import torch
import torch.nn as nn
import torch.nn.functional as Function
import numpy as np

class conv2add(Function):
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
    def __init__(self,tab):
        self.tab=tab

    def forward(self.input):
        
