import argparse
import os
import sys
import time
import configparser
import pdb
from general.logger import log_print

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hardware.buffer import *
from hardware.analog.array import CIM_Array
from hardware.digital import *
from hardware.sensor import CIP

from model.resnet_quant_inf import resnet18
from general.mapping import mapping, build_hardware_flow
from general.logger import log_print

parser = argparse.ArgumentParser(description='Tool_Chain')
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', type=float, default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', type=float, default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--parallelRead', type=float, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
parser.add_argument('--ADCprecision', type=float, default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', type=float, default=1, help='cell precision (e.g. 4-bit/cell)')

# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', type=float, default=0., help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', type=float, default=0, help='retention time')
parser.add_argument('--v', type=float, default=0, help='drift coefficient')
parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', type=float, default=0, help='drift target for fixed-direction drift, range 0-1')

class ChipAreaCalculator:
    def __init__(self, sw_config, hw_config, network_type):
        self.network = sw_config
        self.network_type = network_type
        self.hardware = hw_config

        self.sensor_level = int(Hw_config.get('Sensor level', 'Sensor'))     
        self.digital_area = float(hw_config.get('Digital module', 'Digital_area', fallback =0))

        # self.controller_area = float(hw_config.get('Architecture level', 'Controller_area', fallback =0))
        self.bus_area = float(hw_config.get('Architecture level', 'bus_area', fallback =0))
        # self.pmu_area = float(hw_config.get('Sensor level', 'PMU_area', fallback =0))
        self.CIP_array = CIP.CIP_Array(sw_config,hw_config)

        self.hardware_flow_generate()



    def hardware_flow_generate(self):
        self.hw_flow = build_hardware_flow(self.network, self.hardware, self.network_type)


    def CIM_array_area(self):
        if self.array_num > 1 :
            for i in len(self.CIM_array):
                log_print("area_CIMarray_",i,":", self.CIM_array[i].area)
        return sum(array.area for array in self.CIM_array)

    def control_area(self):
        # Example: control logic grows with control signals and decoder size
        if self.controller_area == -1:
            controller_area = 0
        elif self.controller_area > 0:
            controller_area = self.controller_area
        return controller_area
    
    def CIP_array_area(self):
        return sum(array.area for array in self.CIP_array)
    
    def PMU_area(self):
        # Example: control logic grows with control signals and decoder size
        if self.pmu_area == -1:
            pmu_area = 0
        elif self.pmu_area > 0:
            pmu_area = self.pmu_area
        return pmu_area

    def interconnect_bus_area(self):
        if self.bus_area == -1 :
            bus_area = 0         
        elif self.bus_area >= 0:
            bus_area = self.bus_area
        # return self.num_buses * self.bus_width * self.routing_distance * 0.05
        return bus_area
    
    def digital_module_area(self):
        if self.digital_area == -1:
            digital_module_area = 0
        elif self.digital_area > 0 :
            digital_module_area = self.digital_area
        else:
            # self.pooling.calculate_Pooling_area()


            # pooling_area = self.pooling.Pooling_area if self.pooling.pooling_switch != 0 else 0
            # shiftadd_area = self.shiftadd.ShiftAdd_area if self.shiftadd.shiftadd_switch !=0 else 0


            # digital_module_area = pooling_area+ shiftadd_area, BN, shortcut, activation
            digital_module_area =0
        return digital_module_area

    def total_chip_area(self):       

        hw_flow = self.hw_flow
        area_digital = 0 
        area_buf = 0
        padding_ = []
        shiftadd_ = []
        batchnorm_ = []
        shortcut_ = []
        hardtanh_ =[]
        buffer_ = []
        fifo_ = []

        for i,hw in enumerate(hw_flow):
            if isinstance(hw,(padding)):
                # log_print(hw)
                padding_.append(hw.area) 
            if isinstance(hw,(ShiftAdd)):
                # log_print(hw)
                shiftadd_.append(hw.area) 
            if isinstance(hw,(BatchNormalization)):
                # log_print(hw)
                batchnorm_.append(hw.area) 
            if isinstance(hw,(Shortcut)):
                # log_print(hw)
                shortcut_.append(hw.area) 
            if isinstance(hw,(HardTanh)):
                # log_print(hw)
                hardtanh_.append(hw.area) 

            if isinstance(hw,(Global_buffer)):
                buffer_.append(hw.area)
            if isinstance(hw,(FIFO)):
                fifo_.append(hw.area)

        area_digital = max(padding_)+ max(shiftadd_)+ max(batchnorm_)+ max(shortcut_)+ max(hardtanh_)
        area_buf = max(buffer_)+max(fifo_)
        print("  --area_digital:",area_digital)
        print("  --area_buffer:",area_buf)

        # area_CIM = self.CIM_array_area()
        # area_bus = self.interconnect_bus_area()
        # area_control = self.control_area()
        
        # log_print("area_CIM:",area_CIM)
        # log_print("area_control:",area_control)
        # log_print("area_bus:",area_bus)

        if self.sensor_level == 1:
            area_CIP = self.CIP_array_area()
            area_pmu = self.PMU_area()
            
            log_print("area_CIP:",area_CIP)
            log_print("area_pmu:",area_pmu) 
        else:
            area_CIP = 0
            area_pmu = 0
        

        # total_area = (area_CIM + area_digital + area_buf +area_CIP + area_pmu)
        # return total_area


if __name__ == "__main__":

    HwConfig_path = './config.ini'
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')

    model = resnet18(pretrained=False)

    calculator = ChipAreaCalculator(sw_config=model, hw_config=Hw_config, network_type= "ResNet")
    calculator.total_chip_area()
    # log_print("Total Chip Area:", calculator.total_chip_area()*1e6, "mm^2")