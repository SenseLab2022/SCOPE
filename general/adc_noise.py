import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from scipy.stats import gaussian_kde
import pdb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
# import parameters
# from parameters import *
from general.param_runtime import P


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hardware.analog import array


torch.set_printoptions(precision=4)



def adc_simulate(Vin, Vref=(0.0, 1.0), n_bits=8, dequantize =False, dtype=torch.float32):
    
    vmin, vmax = Vref
    # vmin*=1000
    # vmax*=1000
    assert vmax > vmin, "Vref must satisfy Vref[1] > Vref[0]"
    assert n_bits >= 1, "n_bits must be at least 1"
    levels = 2 ** n_bits  # 量化等级，比如 8-bit -> 256
    Vin_clamped = torch.clamp(Vin, min=vmin, max=vmax)
    code = torch.round((Vin_clamped - vmin) * levels / (vmax - vmin)).to(torch.int32)
    # code = code -(levels/2)

    
    Vout = code.to(dtype) * (vmax - vmin) / (levels) + vmin
    return code, Vout
   
    
def sar_adc(input_voltage, Vref=[0, 1.0], n_bits=4, verbose=False):
    # input_voltage *= 1000
    assert torch.all((input_voltage >= Vref[0]) & (input_voltage <= Vref[1])), "Input voltage out of range"

    device = input_voltage.device  
    input_voltage = input_voltage.clone()
    mid_voltage = (Vref[0]+Vref[1]) / 2
    

    code = torch.zeros_like(input_voltage, dtype=torch.int32, device=device)
    sign_bit_mask = 1 << (n_bits - 1)

    is_negative = input_voltage < mid_voltage
    code[is_negative] |= sign_bit_mask
    normalized_voltage = torch.where(
        is_negative,
        mid_voltage + input_voltage -Vref[0], 
        input_voltage - mid_voltage +Vref[0]  
    )

    for i in range(n_bits - 2, -1, -1):
        trial_code = code | (1 << i)
        trial_voltage = ((Vref[1]-Vref[0]) * trial_code.float() / (2 ** n_bits)) +Vref[0]
    
        keep_mask = trial_voltage <= normalized_voltage
        code = torch.where(keep_mask, trial_code, code)

        if verbose: 
            print(f"Bit {i}: trial_code = {(1 << i)} | updated {keep_mask.sum().item()} entries")

    # #Compute the final digital voltage
    digital_voltage = ((Vref[1]-Vref[0]) * code.float() / (2 ** n_bits)) +Vref[0]
    
    code_list = code.view(-1).tolist()
    binary_strs = [bin(c)[2:].zfill(n_bits) for c in code_list]
    np_str_array = np.array(binary_strs, dtype=f"<U{n_bits}")  
    np_str_array = np_str_array.reshape(code.shape) # Binary
    signed_ints = binary_string_tensor_to_signed_int(np_str_array).to(device=device) # Int
    return signed_ints, code, digital_voltage

def add_noise(tensor, std=0.05, distribution='gaussian'):
    if distribution == 'gaussian':
        # noise = torch.normal(0, std, size=tensor.shape).to(tensor.device) * tensor
        noise = torch.normal(0, std, size=tensor.shape).to(tensor.device) 
    elif distribution == 'uniform':
        noise = (torch.rand(tensor.shape) - 0.5) * 2 * std
        noise = noise.to(tensor.device)
    else:
        noise = torch.zeros_like(tensor)
    return tensor + noise

def binary_string_tensor_to_signed_int(binary_str_tensor: np.ndarray):
    """
    convert binary string tensor to signed integer tensor.
    """
    n_bits = len(binary_str_tensor.flat[0])
    flat_strs = binary_str_tensor.reshape(-1)

    signed_ints = []
    for b in flat_strs:
        if b[0] == '1':
            # negative value, value = int(b, 2) - 2^n
            val = int(b, 2) - (1 << n_bits)
        else:
            val = int(b, 2)
        signed_ints.append(val)

    int_array = np.array(signed_ints, dtype=np.int32).reshape(binary_str_tensor.shape)
    return torch.from_numpy(int_array)


def blockwise_conv2d(bit, weight_q, bias=None, stride=1, padding=1, dilation=1, groups=1):
    """
    dual-blockwise convolution simulating ADC operation with noise and variations.
    """
    B, C_in, H, W = bit.shape
    C_out, _, KH, KW = weight_q.shape

    split_in = 16
    split_out = 16
    in_block = math.ceil(C_in / split_in)   
    out_block = math.ceil(C_out / split_out)  

    

    # for i in range(C_out):
    out_chunks = []
    for j in range(in_block):
        input_chunk = bit[:, j * split_in: (j + 1) * split_in]

        # corresponding the block: [out_block, in_block, KH, KW]
        weight_chunk = weight_q[:, j * split_in: (j + 1) * split_in]
        out_chunk = F.conv2d(input_chunk, weight_chunk, bias=None,
                                stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        outputs = []

        for i in range(C_out):
            result_bit = out_chunk[:, i:i+1]

            #simulate ADC with noise and variations
            result_bit_real = result_bit.detach()
            
            
            #add parasitic capacitance effect, convert digital value to voltage
            cap_parasitic = P.cap_parasitic_tensor.to(result_bit_real.device)
            if cap_parasitic.numel() > 0:
                result_bit_real = cap_parasitic[(result_bit_real + P.CELL_NUM).long()]
            else: result_bit_real = result_bit_real/P.CELL_NUM*P.SUPPLY_VOLTAGE/2+P.SUPPLY_VOLTAGE/2
            #add variation effect
            cap_variation = P.cap_variation_tensor.to(result_bit_real.device)
            if cap_variation.numel() >0:
                variation_values = cap_variation[torch.randint(0, len(cap_variation), result_bit_real.shape)]
                result_bit_real = result_bit_real+variation_values

            #add noise effect
            if P.NOISE != 0:
                result_bit_real = add_noise(result_bit_real, std=P.NOISE, distribution='gaussian') 

            code, Vout  = adc_simulate(result_bit_real, Vref=(0.00, P.SUPPLY_VOLTAGE),n_bits=P.ADC_BIT)
            result_bit_adc = torch.round((Vout-P.SUPPLY_VOLTAGE/2)*P.CELL_NUM/P.SUPPLY_VOLTAGE*2) #convert back to digital value
            # result_bit, _ , digital_voltage = sar_adc(result_bit_real, Vref=[50, 850], n_bits=8)

            
            outputs.append(result_bit_adc)
        result = torch.cat(outputs, dim=1)

        out_chunks.append(result)

    # adding up all input blocks' contributions
    summed = torch.stack(out_chunks, dim=0).sum(dim=0)
    
        

    # adding bias if provided
    if bias is not None:
        bias_block = bias[:].view(1, -1, 1, 1)
        summed = summed + bias_block

        # outputs.append(summed)
    
    # result = torch.cat(outputs, dim=1)
    return summed


def add_disturbance(input, variation, total_len):
  
    if input > 0:
        base = torch.cat([torch.full((input,),1.0),torch.zeros(total_len-input)])
    else: base = torch.cat([torch.full((abs(input),), -1.0),torch.zeros(total_len-input)])

    base = base[torch.randperm(total_len)]
    noise = torch.empty(total_len).uniform_(-variation/100, variation/100)

    result= base+noise
    result = result*P.SUPPLY_VOLTAGE/2+P.SUPPLY_VOLTAGE/2
    real = input/P.CELL_NUM*P.SUPPLY_VOLTAGE/2+P.SUPPLY_VOLTAGE/2
    mean_value = result.mean()
    # print("input:",input,"mean_value:",mean_value-real)

    return mean_value-real

def generate_variation():
    list = []
    for i in range(-P.CELL_NUM,P.CELL_NUM+1,1):
        value= add_disturbance(input = i, variation = P.CAP_VARIATION, total_len = P.CELL_NUM)  
        list.append(value)
    y = [t.item() for t in list]


    return y

def generate_parastic():
    list = []
    cap = array.cap_computing
    for i in range(-P.CELL_NUM,P.CELL_NUM+1,1):
        input = i
        value = (cap*(P.CELL_NUM+input)*P.SUPPLY_VOLTAGE/2+P.CAP_PARASITIC*P.SUPPLY_VOLTAGE/2)/(P.CELL_NUM*cap+P.CAP_PARASITIC)
            
        list.append(value)
    # y = [t.item() for t in list]

    return list




if __name__ == "__main__":
    
    # variation_list = generate_variation()
    variation_list = generate_parastic()
    variation_tensor = torch.tensor(variation_list)

    x = torch.randn(2, 3, 4, 4)

    random_values = variation_tensor[torch.randint(0, len(variation_tensor), x.shape)]
    

    

    