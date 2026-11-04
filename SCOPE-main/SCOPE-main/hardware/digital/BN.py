import torch
import time
import pdb
from general.logger import log_print

# from general.parameters import *
from general.param_runtime import P



class BatchNormalization():
    def __init__(self,sw_config, BN_config, bn_channel_size=16):

        self.bn = int(BN_config.get('Digital module', 'bn_switch', fallback=0))
        self.bn_tech = P.DEVICE_TECH

        self.power = float((BN_config.get('Digital module', 'bn_power', fallback=0)))
        self.area = float((BN_config.get('Digital module', 'bn_area', fallback=0)))
        self.latency = float((BN_config.get('Digital module', 'bn_latency', fallback=0)))
        if P.BIT_SERIAL:
            self.input_bitwidth = (P.ACTIVATION_BIT-1) + P.ADC_BIT
        else: self.input_bitwidth =  P.ADC_BIT
        self.bn_channel_size = sw_config.num_features

        self.times = 0

        self.calculate_BN_area()
        self.calculate_BN_power()



    def calculate_BN_area(self):

        if self.area == 0: 
            bn_area_dict = {
                11: {
                    8: 3183.4908,
                    16:  6335.758919
                },
                16: {
                    16: 8692.11016
                }        
            }

            if self.input_bitwidth in bn_area_dict:
                if self.bn_channel_size in bn_area_dict[self.input_bitwidth]:
                
                    self.area = bn_area_dict[self.input_bitwidth][self.bn_channel_size]
                else: 
                    self.area = bn_area_dict[self.input_bitwidth][16] * self.bn_channel_size / 16
            else: 
                self.area = (bn_area_dict[16][16]-bn_area_dict[11][16])/5 * (self.input_bitwidth-11)+bn_area_dict[11][16]
                self.area = self.area *self.bn_channel_size /16
            self.area = self.area * pow((self.bn_tech/40),2)* self.bn_channel_size/16

    def calculate_BN_power(self):

        if self.power == 0 :
            bn_power_dict = {
                11: {
                    8: 2.8661e-6,
                    16:  5.7212e-6
                },
                16: {
                    16: 7.813e-16
                }        
            }
            
            if self.input_bitwidth in bn_power_dict:
                if self.bn_channel_size in bn_power_dict[self.input_bitwidth]:
                
                    self.power = bn_power_dict[self.input_bitwidth][self.bn_channel_size]
                    
                else: 
                    self.power = bn_power_dict[self.input_bitwidth][16] * self.bn_channel_size / 16
            else: 
                self.power = (bn_power_dict[16][16]-bn_power_dict[11][16])/5*(self.input_bitwidth-11)+bn_power_dict[11][16]
                self.power = self.power * self.bn_channel_size / 16

            self.power = self.power * pow((self.bn_tech/40),2) *((P.FREQUENCY/10e6)**1.3)

             

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY *self.times
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)


    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def __str__(self):
        return "Batch Normalization({})".format(self.data)

