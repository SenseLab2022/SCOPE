import os
import math
import sys
import pdb
from general.logger import log_print
import random
import torch
import torch.nn as nn
import time
import warnings

import configparser
from model.resnet_quant_inf import resnet18

from general.param_runtime import P

# from general.find_min_banks import min_banks
from hardware.ADC import ADC
from hardware.DAC import *

now = time.time()
cap_computing = 900e-18 #f
cap_buf = 2*(3.511e-17+3.962e-17)+(1.507e-16+1.602e-16+3.8369e-17+1.403e-16) #energy consumption of controller 

class SRAMCell_8T:
    def __init__(self, Hw_config, 
                 device_type=1
                 ):
        self.Hw_config = Hw_config
        self.stored_value = 0  
        self.device_type = device_type
        self.cell_area = 0  
        self.computingunit_area = 0 
        
        
        self.cycle_per_compute = P.CYCLE_CIM
        self.supply_voltage = P.SUPPLY_VOLTAGE
        self.cap = cap_computing
        
        #weight write mode
        try:
            self.energy_write_sram = float(self.Hw_config.get('Computing Unit level', 'energy_per_write'))
        except Exception as e:
            raise ValueError("Energy consumption per write operation in CIM SRAM cell must be specified") from e

        self.calculate_area()
    
    def calculate_area(self):
        
        if self.device_type == 1: 
            self.computingunit_area= float(self.Hw_config.get('Computing Unit level', 'unit_area',fallback=0))
            num_unit = int(self.Hw_config.get('Computing Unit level', 'num_unit',fallback=1))

            config_area = float(self.Hw_config.get('Computing Unit level', 'cell_area',fallback=0))
            self.cell_area = config_area if config_area != 0 else (self.computingunit_area * num_unit *1.10) #dig_ear_coefficient
        else:
            self.cell_area = 0
            raise IndexError("Units other than SRAM are not supported yet") 
            
    
    def calculate_energy(self,alpha):

        self.energy_per_access = float(self.Hw_config.get('Computing Unit level', 'energy_per_operation',fallback =0))  # J
        if self.energy_per_access == 0:

            #charge/discharge energy of capacitor during computing
            if alpha is None:
                self.power_compute_switch = 0
            else:   
                vdd = self.supply_voltage
                gnd = 0  
                
                self.power_compute_switch = alpha * P.FREQUENCY *((self.cap *(vdd/2)**2)+ (cap_buf*(vdd**2)))
                
                self.energy_per_access = self.power_compute_switch/P.FREQUENCY

        else: self.power_compute_switch = self.energy_per_access *P.FREQUENCY
        
        # return self.energy_per_access     

    def calculate_latency(self):
        self.compute_latency = self.cycle_per_compute / P.FREQUENCY

        return self.compute_latency


        
class Neuron():
    def __init__(self,Hw_config, count):        
        self.Hw_config = Hw_config
        self.adc = ADC(Hw_config)

        self.sensor_level = int(Hw_config.get('Sensor level', 'Sensor'))
        if self.sensor_level == 1:
            num_sram_cells = int(Hw_config.get('Cell level', f'Cell_num_{count}', fallback=0))
            # self.num_comp = int(Hw_config.get('Cell level', f'comp_num_{count}', fallback=0))
            # self.comp_area = float(self.Hw_config.get('Cell level', 'comp_area', fallback=0))
        else : 
            num_sram_cells = int(Hw_config.get('Cell level', 'Cell_num',fallback=0))


        self.srams = [SRAMCell_8T(Hw_config) for i in range(num_sram_cells)]
       
        
        
    def calculate_alpha(self,activation_input):
        #Switching activity ratio alpha calculation
        if activation_input is not None:
            activation_input = activation_input.view(activation_input.shape[0],-1)
            mask = (activation_input == 1) | (activation_input == -1) 
    

            active_count = mask.sum(dim=1)  
            self.alpha = (active_count / activation_input.shape[1]).tolist() 

        else: self.alpha =[None] *P.CELL_NUM



    def energy_compute_neuron(self,activation_input):
        
        self.calculate_alpha(activation_input) 
        vdd = P.SUPPLY_VOLTAGE
        gnd = 0
        
        power_switch = 0
        self.energy_write_in = 0
        for i in range(len(self.srams)):
            self.srams[i].calculate_energy(self.alpha[i])
            self.energy_write_in += self.srams[i].energy_write_sram
            power_switch += self.srams[i].power_compute_switch


        activation_input = activation_input.view(activation_input.shape[0],-1) 
        input_dict = {1: vdd,
                      0: vdd/2,
                      -1: gnd
        }
        self.energy_compute = 0
        
        input_weight = activation_input.sum(dim=0)
        V_share = (input_weight * vdd / 2) / P.CELL_NUM + vdd / 2

        input_voltage = torch.tensor([input_dict[int(x)] for x in activation_input.view(-1)]).view(activation_input.shape).to(activation_input.device)  #digital to analog
        V_share_expanded = V_share.unsqueeze(0).expand(activation_input.shape) 
        energy_matrix = 0.5 * cap_computing * ((vdd / 2 - input_voltage) ** 2 + (vdd / 2 - V_share_expanded) ** 2) 

        energy_charge_reset_per_col = energy_matrix.sum(dim=0) 
        total_energy_per_col  = power_switch / P.FREQUENCY +energy_charge_reset_per_col+ self.adc.energy_per_conversion * (2 ** self.adc.precision)
        self.energy_compute += total_energy_per_col.sum().item()

    def compute_latency(self,activation_input):

        return (self.srams[0].calculate_latency()+self.adc.latency) *(activation_input.shape[2]*activation_input.shape[3]) #times
                            


class Bank:
    def __init__(self, Hw_config,count):
        self.Hw_config = Hw_config
        self.Device_Type = int(Hw_config.get('Computing Unit level', 'unit_type'))

        self.num_neuron = P.COLUMN_NUM
        self.neurons = [Neuron(Hw_config,count) for i in range(self.num_neuron)]

        
        
        self.column_area = 0
        self.neuron_area = 0
        self.area = 0
        self.Total_energy = 0  #init
        self.latency = 0
        # self.calculate_bank_area()
        

    def calculate_bank_area(self):
        config_area = float(self.Hw_config.get('CIM Array level', 'Bank_area',fallback=0))

       

        if config_area == 0:

            config_column_area = float(self.Hw_config.get('CIM Array level', 'Column_area',fallback=0))
            if config_column_area == 0 :

                config_neuron_area = float(self.Hw_config.get('CIM Array level', 'Neuron_area',fallback=0))
                if config_neuron_area > 0:
                    self.neuron_area = config_neuron_area
                elif len(self.srams) > 0:
                    co_PR = float(self.Hw_config.get('CIM Array level', 'Co_Place_Route',fallback=0)) #coefficient
                    self.neuron_area = (sum(cell.cell_area for cell in self.srams)+ (self.comp_area *self.num_comp)) *(1+co_PR)
                else: raise SystemExit("Missing Neuron_area or number of SRAMs")
            
          
            try:
                column_num = int(self.Hw_config.get('CIM Array level', 'Column_num'))
            except Exception as e:
                raise ValueError("Column_num must be specified") from e
            
            self.column_area = config_column_area if config_column_area != 0 else (self.neuron_area + self.adc.area)
    

        self.area = config_area if config_area != 0 else (self.column_area * column_num)   # Bank total area

    def calculate_bank_energy(self,activation):
        
        self.neurons[0].energy_compute_neuron(activation[0])
        self.energy_compute = self.neurons[0].energy_compute * len(self.neurons)
        self.energy_write_in = self.neurons[0].energy_write_in * len(self.neurons)
      
        # for i,neuron in enumerate(self.neurons):
        #     neuron.energy_compute_neuron(activation[i])
        # self.energy_compute = sum(neuron.energy_compute for neuron in self.neurons)
        # self.energy_write_in = sum(neuron.energy_write_in for neuron in self.neurons)

        

        # print("bank energy:",self.energy_compute,self.energy_write_in)

    def calculate_bank_latency(self,activation):
        
        self.latency = self.neurons[0].compute_latency(activation)

        return self.latency
            


class CIM_Array:
    def __init__(self, Sw_config, Hw_config, type_i , count):
        self.network = Sw_config
        self.Hw_config = Hw_config
        self.type_i = type_i

        self.compute_banks(type_i)

        self.banks = [Bank(Hw_config,count) for i in range(self.num_banks[0]*self.num_banks[1])]
        config_area = float(Hw_config.get('CIM Array level', 'Array_area', fallback=0))
        self.area = config_area if config_area != 0 else sum(bank.area for bank in self.banks)  
        self.total_energy = 0  
        self.latency = 0

        self.times = 0
        self.data = 0

    def compute_banks(self,type_i):

        if type_i == 1:  # Architecture determined by the network. Compute energy and latency. 
                        # Each CIM_Array may not fully utilize all Banks.
            bank = [0, 0]
            num_computingunit = self.network.kernel_size[0] * self.network.kernel_size[1] * self.network.in_channels
            bank[0] = math.ceil(num_computingunit / P.CELL_NUM)  # row
            if num_computingunit % P.CELL_NUM != 0:
                warnings.warn(
                    f"Cell_num {P.CELL_NUM} in Bank cannot evenly divide network parameters {num_computingunit}, hardware resources have remaining capacity", UserWarning)
            bank[1] = math.ceil(self.network.out_channels / P.COLUMN_NUM)  # column
            if self.network.out_channels % P.COLUMN_NUM != 0:
                warnings.warn(
                    f"Column_num {P.COLUMN_NUM} in Bank cannot evenly divide network output channels {self.network.out_channels}, hardware resources have remaining capacity", UserWarning)
            num_bank = list(map(int, self.Hw_config.get('CIM Array level', 'Bank_num').split(',')))
            assert bank[0] <= num_bank[0], "Network parameters exceed CIM array size (rows)"
            assert bank[1] <= num_bank[1], "Network parameters exceed CIM array size (columns)"
            self.num_banks = bank


        elif type_i == 2: ##For compute area
            self.num_banks = list(map(int, self.Hw_config.get('CIM Array level', 'Bank_num').split(',')))

    def split_activation(self,type_i, activation):

        if type_i == 1:
            self.activation_be = []
            self.activation = []
            activation_dim0 = torch.split(activation, P.COLUMN_NUM ,dim=0)
            activation_split = [torch.chunk(chunk, self.num_banks[0], dim =1 ) for chunk in activation_dim0]

            for i in range(len(activation_dim0)):
                for j in range(len(activation_split[i])):
                    self.activation_be.append(activation_split[i][j])

            
            for i in range(len(self.activation_be)): 
                c = self.activation_be[i].shape
                if c[0] < P.COLUMN_NUM or c[1] < P.CELL_NUM:
                    new_t = torch.zeros((P.COLUMN_NUM,P.CELL_NUM,c[2],c[3]))
                    new_t[:c[0],:c[1],:,:] =self.activation_be[i]

                else: new_t =  self.activation_be[i]
                self.activation.append(new_t)
                       

        elif type_i == 2:
            self.activation = [None] * self.num_banks[0] * self.num_banks[1]

    def calculate_energy_CIM(self,activation):

        log_print("    Start calculate CIM_energy   ")
        self.split_activation(self.type_i,activation)
        
        self.calculate_latency()
        
        for i,bank in enumerate(self.banks):
            bank.calculate_bank_energy(self.activation[i])
        self.Total_energy_compute = sum(bank.energy_compute for bank in self.banks)
        self.Total_energy_write_in = sum(bank.energy_write_in for bank in self.banks)
        self.Total_energy = self.Total_energy_compute + self.Total_energy_write_in
        log_print('  --Total Energy', self.Total_energy)
        log_print('     --energy_write_in', self.Total_energy_write_in)
        log_print('     --energy_compute', self.Total_energy_compute)
        
        self.power = self.Total_energy/self.latency
        log_print("  --Power:",self.power)
        
        

    def calculate_latency(self):
        for i,bank in enumerate(self.banks):
            latency = bank.calculate_bank_latency(self.activation[i])
            
            if latency > self.latency :
                self.latency = latency

        log_print('  --Latency:',self.latency)
               
        
    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def __str__(self):
        return "CIM_Array({})".format(self.data)
    

if __name__ == "__main__":

    shape = (16,27,4,1024)
    shape1 = (16,144,4,1024)
    shape2 = (16,144,4,256)
    shape3 = (16,144,4,256)
    indices = torch.randint(0,3,shape)
    indices1 = torch.randint(0,3,shape1)
    indices2 = torch.randint(0,3,shape2)
    indices3 = torch.randint(0,3,shape3)
    mapping = torch.tensor([-1, 0, 1])
    

    result = [mapping[indices],mapping[indices1],mapping[indices2],mapping[indices3]]

    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    # vgg_model = vgg16(pretrained=False)
    model = resnet18(pretrained=False)

    i = 0
    array_list = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            array = CIM_Array(layer,Hw_config,1,count= 0)
            i += 1
            array_list.append(array)

    for i,hw in enumerate(array_list):
        array.calculate_energy_CIM(result[i])
        array.calculate_latency()
        log_print("time:", time.time()- now)
        now = time.time()

            
    



