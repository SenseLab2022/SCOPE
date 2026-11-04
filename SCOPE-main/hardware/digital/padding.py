import configparser as cp
import os
import math
import numpy as np
import pdb
from general.logger import log_print

from general.param_runtime import P



class padding():
    def __init__(self, sw_config, padding_config, img_size = [32,32]):
        self.padding = 1

        self.power = float((padding_config.get('Digital module', 'padding_power', fallback=0)))
        self.area = float((padding_config.get('Digital module', 'padding_area', fallback=0)))

        self.latency = 0
        self.padding_tech = P.DEVICE_TECH

        self.times = 0
        self.data = 0 

        self.img_size = np.array(img_size)
        self.padding_channel_size = math.ceil(sw_config.in_channels/P.COLUMN_NUM) * P.COLUMN_NUM
        self.activation_bitwidth = P.ACTIVATION_BIT
        self.input_bitwidth  = self.activation_bitwidth*self.padding_channel_size

        self.calculate_padding_area()
        self.calculate_padding_power()



    def calculate_padding_area(self):

        if self.area == 0: 
            padding_area_dict = {
                16: {
                    64: 8692.11016
                },
                32: {
                    64: 19227.423311,
                    128:  37446.367804,
                    256: 73781.062794
                },
                64: {
                    64: 37784.526
                },
                    
                   
            }
            if (self.img_size[0] in padding_area_dict) and (self.input_bitwidth in padding_area_dict[self.img_size[0]]):
                self.area = padding_area_dict[self.img_size[0]][self.input_bitwidth]

            else: 
                self.area =  ((602.35*self.img_size[0] + 586.441) * self.input_bitwidth/64)


            self.area = self.area * pow((self.padding_tech/40),2)* self.padding_channel_size/16

    def calculate_padding_power(self):

        if self.power == 0 :
            padding_power_dict = {
                16: {
                    64: 6.0715e-6
                },
                32: {
                    64: 11.561e-6,
                    128:  22.274e-6,
                    256: 43.643e-6
                },
                64: {
                    64: 22.568e-6
                },
                    
                   
            }
            
            
            if (self.img_size[0] in padding_power_dict) and (self.input_bitwidth in padding_power_dict[self.img_size[0]]):

                self.power = padding_power_dict[self.img_size[0]][self.input_bitwidth]

            else: 
                self.power =  ((0.3437*self.input_bitwidth + 0.568) * self.img_size[0]/32)*1e-6


            self.power = self.power * pow((self.padding_tech/40),2)* self.padding_channel_size/16 *((P.FREQUENCY/10e6)**1.3)


    def calculate_energy(self):
        
        self.energy = self.power*1/P.FREQUENCY *self.times
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)

    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def __str__(self):
        return "Padding({})".format(self.data)