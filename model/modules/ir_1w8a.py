import torch.nn as nn
from typing import Optional, Any

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init
# from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm
# from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from . import binaryfunction
import torch
from torch import Tensor
import math
# from quantizeInt8 import activation_quantize_fn
from torch.nn.parameter import Parameter
import pdb
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from general.parameters import *
from general.adc_noise import *


    
class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
        weight_q = binaryfunction.qfn2().apply(x, self.w_bit) 
    else:
      weight = torch.tanh(x)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * binaryfunction.qfn2().apply(weight,self.w_bit-1) - 1)
    return weight_q
    

class IRConv2d_cim(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, precision = 4):
        super(IRConv2d_cim, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.precision = ACTIVATION_BIT
        self.quantize_fn = weight_quantize_fn(w_bit=WEIGHT_BIT)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()
        self.bw = 0

    def forward(self, input):
        
        device = input.device
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0),device=device).float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = self.quantize_fn(bw)
        
        #*********************************************************************#
        results = []
        results_final = 0
        data_1 = input #1,3,32,32
        for i in range(self.precision):
            bit = data_1 % 2
            data_1 = torch.floor(data_1 / 2)
            
            
            result_sw =blockwise_conv2d(bit, bw, self.bias,
                                                stride=self.stride,
                                                padding=self.padding,
                                                dilation=self.dilation,
                                                groups=self.groups)

            
            result_bit = result_sw * sw.view(1,-1,1,1)
            
            if i == self.precision -1:
                results_final += result_bit * (-(2**(i)))
            else: results_final += result_bit *(2**(i))

    
            unfolded = F.unfold(bit,kernel_size=bw.shape[-1], padding=self.padding, stride=self.stride)
            # print(unfolded.shape)

            new_weight = bw.view(bw.shape[0],-1,1)
            input_list = new_weight * unfolded
            # print(input_list.shape)
            results.append(input_list)

            del unfolded, input_list
            torch.cuda.empty_cache()

        output_list = torch.stack(results,dim=2)
        

        # self.resu1 = torch.clamp(self.resu1, -128, 127)
        # self.resu2 = torch.clamp(self.resu2, -128, 127)
        # self.resu3 = torch.clamp(self.resu3, -128, 127)
        # self.resu4 = torch.clamp(self.resu4, -128, 127)

        # result = self.resu1 + self.resu2 * 2 + self.resu3 * 4 + self.resu4 * (-8)
        sw = sw.view(1, -1, 1, 1)
        # return results_final, sw, output_list
        return results_final, sw, output_list

class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()

        self.bw = 0
        self.quantize_fn = weight_quantize_fn(w_bit=1)
        # self.groups = groups

    def forward(self, input):
        device = input.device
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0),device=device).float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = self.quantize_fn(bw)
        # bw = binaryfunction.BinaryQuantize().apply(bw.cuda(), self.k.to(device), self.t.to(device))
        bw = bw * sw
        output = F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output
    



