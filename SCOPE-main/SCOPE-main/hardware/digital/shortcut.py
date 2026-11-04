import torch
import time
import pdb
from general.logger import log_print

from general.param_runtime import P




class Shortcut():
    def __init__(self,shortcut_config, shortcut_channel_size=16, input_bitwidth=16 ):

        self.shortcut = int(shortcut_config.get('Digital module', 'shortcut_switch', fallback=0))
        self.shortcut_tech = P.DEVICE_TECH

        self.power = float((shortcut_config.get('Digital module', 'shortcut_power', fallback=0)))
        self.area = float((shortcut_config.get('Digital module', 'shortcut_area', fallback=0)))
        self.latency = 1/P.FREQUENCY

        self.input_bitwidth = input_bitwidth
        self.shortcut_channel_size = shortcut_channel_size

        self.calculate_shortcut_area()
        self.calculate_shortcut_power()

        self.times = 0



    def calculate_shortcut_area(self):

        if self.area == 0: 
            shortcut_area_dict = {
                16: {
                    16:  11696.907648
                },
                17: {
                    16: 12136.143648
                },
                18: {
                    16: 12624.242450
                } ,
                19: {
                    16: 13182.5485
                },   
                20: {
                    16: 13572.039653
                }, 
                21: {
                    16: 14119.408858
                }    
            }

        if self.input_bitwidth in shortcut_area_dict:
            if self.shortcut_channel_size in shortcut_area_dict[self.input_bitwidth]:
            
                self.area = shortcut_area_dict[self.input_bitwidth][self.shortcut_channel_size]
            else: 
                self.area = shortcut_area_dict[self.input_bitwidth][16] * self.shortcut_channel_size / 16
        else: self.area =  (485.1*self.input_bitwidth+ 3914.2)  *self.shortcut_channel_size / 16

        
        self.area = self.area * pow((self.shortcut_tech/40),2)* self.shortcut_channel_size/16

    def calculate_shortcut_power(self):

        if self.power == 0 :
            shortcut_power_dict = {
                16: {
                    16:  14.516e-6
                },
                17: {
                    16: 15.891e-6
                },
                19: {
                    16: 16.66e-6
                },   
                21: {
                    16: 18.54e-6
                }    
            }
            
        if self.input_bitwidth in shortcut_power_dict: 
            if self.shortcut_channel_size in shortcut_power_dict[self.input_bitwidth]:
            
                self.power = shortcut_power_dict[self.input_bitwidth][self.shortcut_channel_size]
                
            else: 
                self.power = shortcut_power_dict[self.input_bitwidth][16] * self.shortcut_channel_size/ 16
        else: self.power = (0.7427*self.input_bitwidth+2.8469) *self.shortcut_channel_size/16 * 1e-6

        self.power = self.power * pow((self.shortcut_tech/40),2) *((P.FREQUENCY/10e6)**1.3)

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY *self.times
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)

    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def __str__(self):
        return "Shortcut({})".format(self.data)