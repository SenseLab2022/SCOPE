import argparse
import os
import sys
import time
import configparser
import pdb
import warnings
from general.logger import log_print


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hardware.buffer import *
from hardware.analog.array import CIM_Array
from hardware.digital import *
from hardware.sensor import CIP

from model.resnet import resnet16
from general.param_runtime import P


from general.adc_noise import *
from model.modules.replace_quant import *
from general.mapping import mapping, build_hardware_flow
from general.launch import compute_cycle
from general.logger import log_print
from general.array_count import minimal_arrays

class ChipPerformanceCalculator:
    def __init__(self, sw_config, hw_config, network_type):
        self.network = sw_config
        self.network_type = network_type
        self.hardware = hw_config
        self.warnings = [] 
        
        
        self.cycle_calculate()


    # def CIM_array_energy(self):
    #     return sum(array.energy for array in self.CIM_array)
    
    def cycle_calculate(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.hw_flow = build_hardware_flow(self.network, self.hardware, self.network_type)

        self.warnings = [str(item.message) for item in w]

        if P.SENSOR == 1:
            indices = sorted([2, 3, 4], reverse=True)
            for index in indices:
                del self.hw_flow[index]

        self.cycle_run, self.layer_cnt= compute_cycle(self.hw_flow, self.hardware)


    def total_chip_energy(self,activation_list):
        

        layer_cnt = self.layer_cnt
        hw_flow = self.hw_flow
        cycle_run = self.cycle_run
        layer_cnt_list = [layer_cnt[0]] + [layer_cnt[i] - layer_cnt[i - 1] for i in range(1, len(layer_cnt))]
        
        cnt = 0
        total_energy = 0
        total_result = {} 
        first_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, type(hw_flow[0]))]
        last_indices = [index for index, item in enumerate(hw_flow) if isinstance(item, type(hw_flow[-1]))]

        array_list = []
        layer_array = []
        layer_digital = []
        layer_buf = []

        now = time.time()
       
        for i,hw in enumerate(hw_flow):
            if i in first_indices:
                energy_layer = 0 
                power_array = 0
                power_digital = 0
                power_adc = 0
                power_buf = 0
                index = first_indices.index(i)
                log_print()
                log_print(f"----------------------------------------------layer {index} --------------------------------------------")

            if isinstance(hw,(padding, ShiftAdd, BatchNormalization, Shortcut, HardTanh)):
                log_print(hw)
                hw.calculate_energy()
                power_digital += hw.power
                total_energy += hw.energy
                energy_layer += hw.energy
                
            if isinstance(hw,(FIFO, Global_buffer)):
                hw.show_buffer = False
                log_print(hw)
                hw.calculate_energy()
                power_buf += hw.power
                total_energy += hw.energy
                energy_layer += hw.energy    

            if isinstance(hw,(CIM_Array)):
                log_print(hw)
                hw.calculate_energy_CIM(activation_list[cnt])
                power_array += hw.power
                total_energy += hw.Total_energy
                energy_layer +=hw.Total_energy
                cnt += 1
                
                array_list.append(hw.num_banks)
              

            if i in last_indices:
                log_print("\nTotal_energy_layer:",energy_layer, "J")
                log_print("cycle:", layer_cnt_list[index+1])
                log_print("Power_layer:", energy_layer/layer_cnt_list[index+1]*P.FREQUENCY, "W")
                total_result[index] ={
                    'energy': energy_layer,
                    'time': layer_cnt_list[index+1]/P.FREQUENCY,
                    'power': energy_layer/layer_cnt_list[index+1]*P.FREQUENCY
                }
                log_print("Time:", time.time()-now)

                layer_array.append(power_array)
                layer_digital.append(power_digital)
                layer_buf.append(power_buf)

        log_print("———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")        
                
        total_power = total_energy/cycle_run * P.FREQUENCY
        log_print("\n[Processor Summary]")
        log_print("Total_energy(J):", total_energy)
        log_print("Time(s)",cycle_run/P.FREQUENCY, "  fps:", P.FREQUENCY/cycle_run)
        log_print("Total_power(W):",total_power)

        array_shape = list(map(int, Hw_config.get('CIM Array level', 'Bank_num').split(',')))
        n, places = minimal_arrays(array_shape, array_list,  allow_sorted=False, Hw_config= self.hardware)

        
        # # # Plotting the results
        # x_labels = [f'layer{i}' for i in range(len(layer_array))]
        # plt.figure(figsize=(10, 5))
        # plt.plot(layer_cnt[1:], layer_buf[:], label='layer_buf', marker='o')
        # plt.plot(layer_cnt[1:], layer_array[:], label='layer_array', marker='s')
        # plt.plot(layer_cnt[1:], layer_digital[:], label='layer_digital', marker='^')
        # plt.xlabel('Layer')
        # plt.ylabel('Value')
        # plt.title('Layer-wise Comparison of Buffer, Array, and Digital')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()


        return total_result






if __name__ == "__main__":
    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    # Example Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet16().to(device)
    state_dict = torch.load("model/resnet_44000.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    calculator = ChipPerformanceCalculator(sw_config=model, hw_config=Hw_config, network_type= "ResNet")
    
    if P.INFERENCE:
        model = replace_forward_function(model) 
        model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
        print("replaced model with IRConv2d_CIM")
    else:
        model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)
    

    model = replace_forward(model, target_performance=1)

    variation_list = generate_variation()
    parastic_list = generate_parastic()
    P.cap_variation_tensor = torch.tensor(variation_list).to(device)
    P.cap_parasitic_tensor = torch.tensor(parastic_list).to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model(dummy_input)

    performance_result = calculator.total_chip_energy(tensor_list)