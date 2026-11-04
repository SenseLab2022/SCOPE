import torch.nn as nn
from typing import Optional, Any

from torch.autograd import Variable
import torch.nn.functional as F
from . import binaryfunction
import torch
from torch import Tensor
import math
# from quantizeInt8 import activation_quantize_fn
import pdb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from general.parameters import *

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
        # self.k = torch.tensor([10]).float().cuda()
        # self.t = torch.tensor([0.1]).float().cuda()
        self.bw = 0

    def gen_weight(self, weight, name):
        for channel_in in range(weight.size(1) // 16):
            for channel_out in range(weight.size(0) // 16):
                resu = []
                count = 0
                str_data = ''
                for j in range(channel_in * 16, (channel_in + 1) * 16):
                    for row in range(3):
                        for col in range(3):
                            for i in range(channel_out * 16, (channel_out + 1) * 16):
                                str_data = str(int(weight[i, j, row, col])) + str_data
                                count = count + 1
                                if count == 8:
                                    resu.append('0x' + hex(int(str_data, 2))[2:].rjust(2, '0'))
                                    str_data = ''
                                    count = 0

                file_path = 'parameters/weights/' + name + "_weight_in" + str((channel_in + 1) * 16) + "_out" + str(
                    (channel_out + 1) * 16) + ".coe"
                with open(file_path, mode='w', encoding='utf-8') as file_obj:
                    for i in resu:
                        file_obj.write(i + ',\n')

    def forward(self, input, layer_idx):
        # pdb.set_trace()
        w = self.weight #16,3,3,3
        
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        sw = torch.pow(torch.tensor([2] * bw.size(0)).float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = self.quantize_fn(self.weight)

        weight_q = bw
        
        #*********************************************************************#
        results = []
        results_final = 0
        data_1 = input
        for i in range(self.precision):
            bit = data_1 % 2
            data_1 = torch.floor(data_1 / 2)
            result_bit = F.conv2d(bit, weight_q, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

            if i == self.precision -1:
                results_final += result_bit * (-(2**(i)))
            else: results_final += result_bit *(2**(i))
        
            unfolded = F.unfold(bit,kernel_size=weight_q.shape[-1], padding=self.padding, stride=self.stride)
            # print(unfolded.shape) #1,27,1024

            new_weight = weight_q.view(weight_q.shape[0],-1,1) #16,27,1
            input_list = new_weight * unfolded
    
            # print(input_list.shape)
            results.append(input_list) #16,27,1024
        
        output_list = torch.stack(results,dim=2) #16,27,4,1024

        # self.resu1 = torch.clamp(self.resu1, -128, 127)
        # self.resu2 = torch.clamp(self.resu2, -128, 127)
        # self.resu3 = torch.clamp(self.resu3, -128, 127)
        # self.resu4 = torch.clamp(self.resu4, -128, 127)

        # result = self.resu1 + self.resu2 * 2 + self.resu3 * 4 + self.resu4 * (-8)
        return results_final, sw, output_list

def adc_simulate(vin, vref=1.0, n_bits=ADC_BIT, add_noise=False, noise_std=0.01, dequantize=False):
    
    # vin = torch.clamp(vin, -128/144, 127/144)

    levels = 2 ** n_bits  # 量化等级，比如 8-bit -> 256
    if (vin<SUPPLY_VOLTAGE/2).any().item():
        adc_value = torch.floor(vin / vref * (levels/2 )).int()
    else: adc_value = torch.floor(vin / vref * (levels - 1)).int()

    if dequantize:
        vin_hat = adc_value.float() / (levels - 1) * vref
        return adc_value, vin_hat
    else:
        return adc_value
    
def sar_adc(input_voltage, Vref=1.0, n_bits=4, verbose=False):
    assert 0 <= input_voltage <= Vref, "Input voltage should be in [0, Vref] range"

    # pdb.set_trace()

    code = 0
    for i in range(n_bits - 1, -1, -1):
        trial_code = code | (1 << i)
        
        trial_voltage = Vref * trial_code / (2 ** n_bits)

        if verbose:
            print(f"try to set bit{i} to be 1 → Code: {bin(trial_code)[2:].zfill(n_bits)} → DAC电压: {trial_voltage:.4f} V")

        if trial_voltage <= input_voltage:
            code = trial_code  
        

    digital_voltage = Vref * code / (2 ** n_bits)
    b = bin(code)[2:]
    return code, digital_voltage

def add_noise(tensor, std=0.05, distribution='gaussian'):
    if distribution == 'gaussian':
        noise = torch.normal(0, std, size=tensor.shape).to(tensor.device) * tensor
    elif distribution == 'uniform':
        noise = (torch.rand(tensor.shape) - 0.5) * 2 * std
        noise = noise.to(tensor.device)
    else:
        noise = torch.zeros_like(tensor)
    return noise


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.k = torch.tensor([10]).float().cuda()
        # self.t = torch.tensor([0.1]).float().cuda()
        self.k = torch.tensor([10]).float()
        self.t = torch.tensor([0.1]).float()
        self.bw = 0

    def forward(self, input):
        w = self.weight
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)

        # sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
        #                (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
        #     bw.size(0), 1, 1, 1).detach()
        sw = torch.pow(torch.tensor([2] * bw.size(0)).float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1, 1, 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw
        output = F.conv2d(input, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output, sw


class IRlinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        super(IRlinear, self).__init__(in_features, out_features, bias=False)
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()

    def forward(self, input, Tmax=[1.0, 0.0]):
        w = self.weight
        x = input
        Tmax = torch.max(input).detach()  # Tmax[0]  # torch.max(input).detach()#
        Tmin = torch.min(input).detach()  # Tmax[1]  # torch.min(input).detach()#
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
        sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
                       (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
            bw.size(0), 1).detach()

        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        bw = bw * sw

        T = torch.max(torch.abs(Tmin), torch.abs(Tmax))
        T = torch.clamp(T, 1e-10, 255.)
        x = torch.clamp(x, 0 - T, T)
        x_s = x / T
        activation_q = binaryfunction.qfn2().apply(x_s, 4)
        activation_q = activation_q * T
        output = F.linear(activation_q, bw, self.bias)
        return output
