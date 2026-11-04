import torch
import time
import pdb
from general.logger import log_print
import configparser
# from general.parameters import *
from general.param_runtime import P
        


class Sigmoid():
    def compute(self, x):
        return torch.sigmoid(x)


class HardTanh():
    def __init__(self, hardtanh_config, hardtanh_channel_size=16,  input_bitwidth=17 ):

        self.hardtanh = int(hardtanh_config.get('Digital module', 'hardtanh_switch', fallback=0))
        self.hardtanh_tech = int(hardtanh_config.get('Device level', 'Device_Tech'))

        self.power = float((hardtanh_config.get('Digital module', 'hardtanh_power', fallback=0)))
        self.area = float((hardtanh_config.get('Digital module', 'hardtanh_area', fallback=0)))
        self.latency = 1/P.FREQUENCY

        self.input_bitwidth = input_bitwidth
        self.hardtanh_channel_size = hardtanh_channel_size

        self.calculate_hardtanh_area()
        self.calculate_hardtanh_power()

        self.times = 0

    def calculate_hardtanh_area(self):

        if self.area == 0: 
            hardtanh_area_dict = {
                17: {
                    16:  10137.884683
                },
                18: {
                    16: 10547.132668
                },    
                19: {
                    16: 10414.303488
                },
                20: {
                    16: 10267.185895
                },
                21: {
                    16: 10484.334279
                },
                22: {
                    16: 10581.001482
                }  
            }

            if self.hardtanh_channel_size in hardtanh_area_dict[self.input_bitwidth]:
            
                self.area = hardtanh_area_dict[self.input_bitwidth][self.hardtanh_channel_size]
            else: 
                self.area = hardtanh_area_dict[self.input_bitwidth][16] * self.hardtanh_channel_size / 16
            self.area = self.area * pow((self.hardtanh_tech/40),2)* self.hardtanh_channel_size/16

    def calculate_hardtanh_power(self):

        if self.power == 0 :
            hardtanh_power_dict = {
                17: {
                    16:  8.6386e-6
                },
                18: {
                    16: 8.7654-6
                },    
                19: {
                    16: 8.9154-6
                },
                20: {
                    16: 8.7896-6
                },
                21: {
                    16: 8.9845-6
                },
                22: {
                    16: 8.8894-6
                }  
            }
            
            if self.input_bitwidth in hardtanh_power_dict:    
                if self.hardtanh_channel_size in hardtanh_power_dict[self.input_bitwidth]:
                
                    self.power = hardtanh_power_dict[self.input_bitwidth][self.hardtanh_channel_size]
                    
                else: 
                    self.power = hardtanh_power_dict[self.input_bitwidth][16] * self.hardtanh_channel_size / 16
            else: self.power = (0.051*self.input_bitwidth +7.8357) * self.hardtanh_channel_size / 16

            self.power = self.power * pow((self.hardtanh_tech/40),2) *((P.FREQUENCY/10e6)**1.3)
            

    def calculate_energy(self):

        self.energy = self.power*1/P.FREQUENCY *self.times 
        log_print("  --power:",self.power)
        log_print("  --energy:",self.energy)


    def compute(self, x):
        return torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)
    
    def processing_one_times(self, data):
        self.times +=1
        self.data = data

    def __str__(self):
        return "Hardtanh({})".format(self.data)


class ReLU():
    def __init__(self):
        super().__init__()
        self.times = 0


    def compute(self, x):
        return torch.nn.functional.relu(x)
    
    def processing_one_times(self, data):
        self.times +=1
        self.data = data


# 主激活类，用于选择并调用具体的激活函数
class Activation():
    def __init__(self, activation_config, activation_type= "hardtanh"):

        self.activation = int(activation_config.get('Digital module', 'activation_switch', fallback=0))
        self.activation_type = (activation_config.get('Digital module', 'Activation_type')).lower()
        self.activation_fn = self._get_activation_function(self.activation_type)

    def _get_activation_function(self, activation_type):
        if activation_type == 'sigmoid':
            return Sigmoid()
        elif activation_type == 'hardtanh':
            return HardTanh()
        elif activation_type == 'relu':
            return ReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def compute_performance(self):
        return self.activation_fn.power *self.activation_fn.latency

    def set_activation(self, activation_type):
        self.activation_type = activation_type.lower()
        self.activation_fn = self._get_activation_function(self.activation_type)


if __name__ == "__main__":
    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    activation = Activation(Hw_config)

    # Change Activation
    activation.set_activation("hardtanh")
    hardtanh_performance = activation.compute_performance()
    log_print("HardTanh:", hardtanh_performance)