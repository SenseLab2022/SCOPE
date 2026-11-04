import torch
import time
import configparser as cp
import os
import math
from general.param_runtime import P


import pdb
from general.logger import log_print



class ShiftAdd():
    def __init__(self, ShiftAdd_config, channel_size):

        self.shiftadd_switch = int(ShiftAdd_config.get('Digital module', 'shiftadd_switch', fallback=0))
        self.ShiftAdd_Tech = P.DEVICE_TECH

        self.power = float((ShiftAdd_config.get('Digital module', 'SA_power', fallback=0)))
        self.area = float((ShiftAdd_config.get('Digital module', 'SA_area', fallback=0))) 

        self.input_bitwidth = int(ShiftAdd_config.get('Interface level', 'ADC_precision')) #shift_add
        self.shiftadd_channelsize = channel_size

        self.times = 0

        self.calculate_ShiftAdd_area()
        self.calculate_ShiftAdd_power()

        self.ShiftAdd_latency = 1/P.FREQUENCY


    def calculate_ShiftAdd_area(self):
        
        if self.area == 0: 
            shiftadd_area_dict = {
                16: {
                    8: 12929.238223,
                    10:  12163.133,
                    12: 11308.827,
                },
                32: {
                    8: 15721.297466
                },
                64: {
                    8: 21516.037
                }

            }

            if self.shiftadd_channelsize in shiftadd_area_dict:
                if self.input_bitwidth in shiftadd_area_dict[self.shiftadd_channelsize]:
                
                    self.area = shiftadd_area_dict[self.shiftadd_channelsize][self.input_bitwidth]
                else: self.area = shiftadd_area_dict[self.shiftadd_channelsize][8] - 405.1*(self.input_bitwidth-8)
            else: 
                self.area = 179.21 * self.shiftadd_channelsize +10032

            self.area = self.area * pow((self.ShiftAdd_Tech/40),2) 


    def calculate_ShiftAdd_power(self):
        
        if self.power == 0 :
            shiftadd_power_dict = {
                16: {
                    8: 25.539e-6,
                    10:  23.648e-6,
                    12: 21.378e-6,
                },
                32: {
                    8: 26.871e-6
                },
                64: {
                    8: 30.57e-6
                }

            }
            
            if self.shiftadd_channelsize in shiftadd_power_dict:
                if  self.input_bitwidth in shiftadd_power_dict[self.shiftadd_channelsize]:
                
                    self.power = shiftadd_power_dict[self.shiftadd_channelsize][self.input_bitwidth]
                    
                else: 
                    self.power = shiftadd_power_dict[self.shiftadd_channelsize][8] -1.0403*(self.input_bitwidth-8)
            else: 
                self.power = (shiftadd_power_dict[16][8]-shiftadd_power_dict[32][8])/16*(self.shiftadd_channelsize)+shiftadd_power_dict[16][8]
                self.power = self.power -1.0403*(self.input_bitwidth-8)

            self.power = self.power * pow((self.ShiftAdd_Tech/40),2) *((P.FREQUENCY/10e6)**1.3)

    def calculate_energy(self):
        #unit mW
        self.energy = self.power * self.ShiftAdd_latency * self.times
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)

    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def ShiftAdd_output(self):
        log_print("ShiftAdd_shape: %d x %d" %(self.ShiftAdd_shape[0], self.ShiftAdd_shape[1]))
        log_print("ShiftAdd_area:", self.ShiftAdd_area, "um^2")
        log_print("ShiftAdd_power:", self.ShiftAdd_power, "W")
        log_print("ShiftAdd_latency:", self.ShiftAdd_latency, "ns")
        log_print("ShiftAdd_energy:", self.ShiftAdd_energy, "nJ")

    def __str__(self):
        return "ShiftAdd({})".format(self.data)