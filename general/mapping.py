from hardware.buffer import *
from hardware.analog.array import CIM_Array
from hardware.sensor.CIP import CIP_Array
from hardware.digital import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx
# from torchvision.models import vgg16, resnet18
import configparser
import random
import numpy as np
from model.resnet_quant_inf import resnet18
from general.launch import compute_cycle
from general.param_runtime import P



import pdb
from general.logger import log_print


import general.energy_scheduler

class mapping():
    def __init__(self, network, Hw_config):
        self.network = network

        self.hw_config = Hw_config



    def mapping_function(self):

        hw_flow = []

        depth = 34*3
        
        for i in range(len(self.network.layers)):
            
            fifo =  FIFO(capacity=depth)
            hw_flow.append(fifo)

            if self.network.layers[i].layer_type == "Conv" :
                hw_flow.append((CIM_Array(self.network, self.hw_config, 1),self.network.layers[i]))

            global_buf = Global_buffer(self.hw_config, buffer_level=1 )

            hw_flow.append(global_buf)

        return hw_flow
    

def build_hardware_flow(model, Hw_config,network_type="VGG"):
    hw_flow = []
    
    image_size = P.INPUT_SIZE

    now = time.time()

    count_CIM = 0
    for name, layer in model.named_modules():

        # log_print("name:",name)
        # log_print("layer:", layer)

        if isinstance(layer, nn.Conv2d):
            # Conv layer need padding and input buffer
            
            if P.SENSOR & (count_CIM==0):
                hw_flow.append(CIP_Array(layer, Hw_config))
                channel_size = layer.out_channels
                hw_flow.append(Global_buffer(Hw_config,channel_size, buffer_level=2))
                
                count_CIM +=1 
            else:

                channel_size = layer.out_channels
                if layer.padding == 1 or layer.padding==(1,1) :
                    hw_flow.append(padding(layer, Hw_config, image_size))

                if P.INPUT_BUFFER == 1:
                    hw_flow.append(FIFO(depth=(image_size[0]+layer.padding[0]*2),width =layer.in_channels,num = layer.kernel_size[1]))
                elif P.INPUT_BUFFER == 2:
                    hw_flow.append(REG_array(depth=layer.kernel_size[0], width=layer.kernel_size[1], num= layer.in_channels))
                elif P.INPUT_BUFFER == 3:
                    hw_flow.append(InputBuf(depth=layer.kernel_size[0], width=layer.kernel_size[1], num= layer.in_channels))
                hw_flow.append(CIM_Array(layer, Hw_config,type_i=1,count=count_CIM+1))
                
                image_size = [
                    (image_size[0]+layer.padding[0]*2-layer.kernel_size[0]+1)// layer.stride[0],
                    (image_size[1]+layer.padding[1]*2-layer.kernel_size[1]+1)// layer.stride[1]
                ]
                count_CIM +=1
                
                if P.BIT_SERIAL:
                    hw_flow.append(ShiftAdd(Hw_config, layer.out_channels))


        elif isinstance(layer, nn.BatchNorm2d):
            # BN 层
            channel_size = layer.num_features
            hw_flow.append(BatchNormalization(layer, Hw_config))
            
            

        elif (network_type == "ResNet" and "shortcut" in name):
        # Resnet module
            hw_flow.append(Shortcut(Hw_config,channel_size))

        elif isinstance(layer, nn.ReLU):
            # ReLU 
            hw_flow.append(ReLU())
            hw_flow.append(Global_buffer(Hw_config, channel_size,buffer_level=1 ))

        elif isinstance(layer, nn.Hardtanh) or layer == F.hardtanh:
            # HardTanh 
            hw_flow.append(HardTanh(Hw_config,channel_size))
            hw_flow.append(Global_buffer(Hw_config,channel_size, buffer_level=1 ))

    log_print("—————————————————————————————————————————————————————Hardware Flow——————————————————————————————————————————————————————————————")  
    log_print(hw_flow)   
    log_print("————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————") 

    return hw_flow

def set_seed(seed=42):
    random.seed(seed)                   
    np.random.seed(seed)                 
    torch.manual_seed(seed)               
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)      

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    set_seed(42)
    # vgg_model = vgg16(pretrained=False)
    resnet_model = resnet18(pretrained=False)

    input_data = torch.randn(1,3,32,32)
    output, activation_list = resnet_model(input_data)

    
    hw_flow = build_hardware_flow(resnet_model, Hw_config, network_type="ResNet")
    cycle_run, layer_cnt= compute_cycle(hw_flow, Hw_config)

    layer_cnt_list = [layer_cnt[0]] + [layer_cnt[i] - layer_cnt[i - 1] for i in range(1, len(layer_cnt))]



    cnt = 0
    total_energy = 0
    total_result = {} 
    first_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, type(hw_flow[0]))]
    last_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, type(hw_flow[-1]))]

    # pdb.set_trace()
    for i,hw in enumerate(hw_flow):
        if i in first_indices:
            energy_layer = 0 
            index = first_indices.index(i)
            log_print()
            log_print(f"------—————————————————————————————————-----layer {index} -------——————————————————————————————————-")

        if isinstance(hw,(padding, ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
            log_print(hw)
            hw.calculate_energy()
            total_energy += hw.energy
            energy_layer += hw.energy
            
        if isinstance(hw,(FIFO, Global_buffer)):
            hw.show_buffer = False
            log_print(hw)
            hw.calculate_energy()
            total_energy += hw.energy
            energy_layer += hw.energy    

        # if isinstance(hw,(CIM_Array)):
        #     log_print(hw)
        #     hw.calculate_energy_CIM(activation_list[cnt])
        #     total_energy += hw.Total_energy
        #     energy_layer +=hw.Total_energy
        #     cnt += 1

        if i in last_indices:
            log_print("\nTotal_energy_layer:",energy_layer, "J")
            log_print("cycle:", layer_cnt_list[index+1])
            log_print("Power_layer:", energy_layer/layer_cnt_list[index+1]*P.FREQUENCY, "W")
            total_result[index] ={
                'energy': energy_layer,
                'time': layer_cnt_list[index+1]/P.FREQUENCY,
                'power': energy_layer/layer_cnt_list[index+1]*P.FREQUENCY
            }
            

    log_print("———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")        
            
    total_power = total_energy/cycle_run * P.FREQUENCY
    log_print("\n[Processor Summary]")
    log_print("Total_energy(J):", total_energy)
    log_print("Time(s)",cycle_run/P.FREQUENCY, "  fps:", P.FREQUENCY/cycle_run)
    log_print("Total_power(W):",total_power)

