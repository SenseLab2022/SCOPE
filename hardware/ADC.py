import os
import math
import pdb
from general.logger import log_print

import configparser
# from general.parameters import *
from general.param_runtime import P



class ADC:
    def __init__(self, Hw_config):
        self.adc_tech = int(Hw_config.get('Device level', 'Device_Tech'))
        self.adc_option = int(Hw_config.get('Interface level', 'ADC_option',fallback=0))
        self.precision = int(Hw_config.get('Interface level', 'ADC_precision'))  # ADC Resolution
        self.area = float(Hw_config.get('Interface level', 'ADC_Area', fallback=0)) 
        self.voltage = float(Hw_config.get('Interface level', 'ADC_supply_voltage', fallback=0)) 
        self.sampling_rate = float(Hw_config.get('Interface level', 'ADC_sampling_rate', fallback=0)) 
        # assert self.sampling_rate <= P.FREQUENCY/self.precision, f"Error: f_sampling ({self.sampling_rate} Hz) must be less than f_clock/N ({FREQUENCY/self.precision} Hz)"        
        
        self.latency = 1/self.sampling_rate

        get_energy = float(Hw_config.get('Interface level', 'energy_per_conversion',fallback = 0)) 
        if get_energy == 0:
            self.energy_per_conversion = self.calculate_energy()  # J
        else: self.energy_per_conversion = get_energy
    
    def calculate_energy(self):
        
        list_adc = [self.voltage, self.sampling_rate, self.precision]
        weights = [1,0.5e-9,0.1] #1:1:1 

        option_dict = {1: [0.90, 2.4e9, 10, 28], #V,f,N,processing_Node
                       2: [0.85, 2.0e9, 10, 16], 
                       3: [1.10, 2.6e9, 10, 40], 
                       4: [1.20, 1.0e9, 12, 65], 
                       5: [0.90, 0.5e9, 11, 28], 
                       6: [1.00, 0.5e9, 10, 28], 
                       7: [0.90, 0.1e9, 12, 28], 
                       8: [1.00, 0.1e9, 12, 55], 

		}

        self.adc_option= find_min_key(option_dict,list_adc, weights, 3) #find to choose which adc

        energy_dict = {1: 17.7, #fJ/conversion
                       2: 20, 
                       3: 25.6, 
                       4: 59.7, 
                       5: 5.58, 
                       6: 21.7, 
                       7: 20.13, 
                       8: 24.26, 

		}

        energy_FOM = energy_dict[self.adc_option]*1e-15
        energy_FOM *= (self.voltage/option_dict[self.adc_option][0])**2 * \
                        (option_dict[self.adc_option][1]/self.sampling_rate*0.25) *\
                        (self.adc_tech/option_dict[self.adc_option][3])
                       

        
        return energy_FOM
    
    
    
def find_min_key(dict, params,weights, N):
    """
    To find the key in dict whose corresponding list has the minimum weighted sum of absolute differences

    """
    min_key = None  
    min_sum = float('inf')  # 初始化最小和为无穷大
    
    for key, values in dict.items():
        # 计算 params 与当前列表的差值之和
        value_list = (abs(p - v) for p, v in zip(params[:N], values[:N]))
        current_sum = sum(p*w for p,w in zip(value_list,weights))
        
        if current_sum < min_sum:
            min_sum = current_sum
            min_key = key

    return min_key

    
if __name__ == "__main__":


    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    adc = ADC(Hw_config)

