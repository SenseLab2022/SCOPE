import argparse
import os
import sys
import time
import configparser
import pdb
from general.logger import log_print
# from general.parameters import *
from general.param_runtime import P


from hardware import buffer

class ActivePixelSensor:
    def __init__(self,
        # performance parameters
        pd_capacitance = 54e-15, # [F] 100fF
        pd_supply = P.SUPPLY_VOLTAGE, # [V]
        dynamic_sf = True,
        output_vs = 1.2,  # output voltage swing [V]
        num_transistor = 3,
        fd_capacitance = 10e-15,  # [F]
        load_capacitance = 1e-12,  # [F]
        tech_node = P.DEVICE_TECH,  # [um]
        pitch = 20,  # [um]
        array_vsize = 224,
        enable_cds = False,):
        
        # set input/output driver
        self.input_need_driver = False
        self.output_driver = True

        self.pd_capacitance = pd_capacitance
        self.pd_supply = pd_supply
        self.dynamic_sf = dynamic_sf
        self.output_vs = output_vs
        if output_vs is None:
            self.output_vs = self.pd_supply - get_nominal_supply(self.tech_node)
        else:
            self.output_vs = output_vs
        self.num_transistor = num_transistor
        self.fd_capacitance = fd_capacitance
        self.load_capacitance = load_capacitance
        self.tech_node = tech_node
        self.pitch = pitch
        self.array_vsize = array_vsize
        self.enable_cds = enable_cds
        if enable_cds:
            self.num_readout = 2
        else:
            self.num_readout = 1


    def calculate_aps_energy(self):
        """Calculate Sensing Energy

        Returns:
            float: the energy consumption of this analog compoenent in unit of ``J``.
        """
        if self.num_transistor == 3:
            energy_fd = 0
        elif self.num_transistor == 4:
            energy_fd = self.fd_capacitance * (self.pd_supply ** 2)  # E = Cfd * Vdd^2 
        else:
            raise Exception("Defined APS is not supported.")

        if self.dynamic_sf:
            energy_sf = (self.load_capacitance +
                         get_pixel_parasitic(self.array_vsize, self.tech_node, self.pitch)) * self.output_vs ** 2
                        # E = (CLoad + Cpixel_para)*Vsw^2
        else:
            energy_sf = (self.load_capacitance +
                         get_pixel_parasitic(self.array_vsize, self.tech_node, self.pitch)) * self.pd_supply * self.output_vs
                        # E = (CLoad + Cpixel_para)*Vdd*Vsw

        energy_pd = self.pd_capacitance * (self.pd_supply ** 2)  
        energy = energy_pd + energy_fd + self.num_readout * energy_sf

        return energy
    def calculate_pd_energy(self):
        """Calculation mode, only return pd energy consumption"""
        return self.pd_capacitance * (self.pd_supply ** 2)
    
    def calculate_pd_load_energy(self):
        """Calculation mode, return pd + load energy consumption"""
        pd_supply = 1.8
        load_capacitance = 100e-12  # [F]
        return load_capacitance * (pd_supply ** 2)


class Comparator(object):
    """ Dynamic voltage comparator.

    Args:
        supply (float): [unit: V] supply voltage.
        i_bias (float): [unit: A] bias current of the circuit.
        t_readout (float): [unit: s] readout time, during which the comparison is finished.
    """

    def __init__(
            self,
            supply=1.8,  # [V]
            c_parasitic=217e-15,  # [F]
            i_bias=1e-6,  # [A]
            t_readout=1e-7  # [s]
    ):
        self.supply = supply
        self.c_parasitic = c_parasitic
        self.i_bias = i_bias
        self.t_readout = t_readout

    def calculate_comparator_energy(self):
        """Calculate Energy

        Returns:
            float: the energy consumption of this analog compoenent in unit of ``J``.
        """
        energy = self.c_parasitic * (self.supply ** 2)
        # energy = self.supply * self.i_bias * self.t_readout  
        return energy
        


class CIP_Array(ActivePixelSensor,Comparator):
    def __init__(self,  Sw_config, Hw_config):
        ActivePixelSensor.__init__(self)
        Comparator.__init__(self)
        self.network = Sw_config
        self.hw_config = Hw_config
        self.area = 0
        self.total_energy = 0  # init
        # self.buffer = buffer.buffer(Sw_config,Hw_config, buffer_level = 2)
        self.times = 0
        self.latency =0
       
        self.pixel_img_size = list(map(int, Hw_config.get('Device level', 'Input_size').split(',')))
        self.pixel_img_width = self.pixel_img_size[1]
        self.pixel_img_height =self.pixel_img_size[0]
        self.kernel_width =  self.network.conv1.kernel_size[0]
        self.kernel_height =  self.network.conv1.kernel_size[1]
        self.kernel_size =  self.network.conv1.kernel_size[0]* self.network.conv1.kernel_size[1]
        self.channel_num = self.network.conv1.out_channels
        self.Hidden_layer_num = int(self.hw_config.get('Sensor level', 'Hidden_layer_num'))

    def calculate_area(self):
        config_area = float(self.hw_config.get('Sensor level', 'CIP_area', fallback=0))

        if config_area > 0 :
            self.area = config_area
        else:
            pixel_img_num = 1
            pixel_img_area = float(self.hw_config.get('Sensor level', 'pixel_img_area'))
            
            for num in self.pixel_img_size:
                pixel_img_num *= num 

            pixel_harvest_area = 0
            for i in range (1,6):
                pixel_hv_area = float(self.hw_config.get('Sensor level', f'pixel_harvest_area_{i}', fallback = 0))
                num = int(self.hw_config.get('Sensor level', f'pixel_harvest_num_{i}', fallback = 0 ))
                pixel_harvest_area += (pixel_hv_area*num)

            PE_area = float(self.hw_config.get('Sensor level', 'PE_area'))
            PE_num = int(self.hw_config.get('Sensor level', 'PE_num'))

            self.area = (pixel_img_num*pixel_img_area) + pixel_harvest_area + (PE_area*PE_num)
    
    def calculate_total_energy(self, NNtype='CNN'):
        # 成像模式
        self.enable_cds = True
        APS_energy = self.calculate_aps_energy()
        total_APS_energy =  self.pixel_img_size * APS_energy
        # 计算模式，E = PD + PE
        if NNtype == 'CNN':
            
            total_pd_energy = self.pixel_img_size  * self.calculate_pd_energy()
            Num_comp = (self.pixel_img_width / self.kernel_width ) * (self.pixel_img_height / self.kernel_width ) 
            total_pe_energy = Num_comp * self.calculate_comparator_energy()
            total_CIP_energy = total_pd_energy + total_pe_energy
        elif NNtype == 'MLP':
            total_pd_energy = self.calculate_pd_load_energy()* self.Hidden_layer_num
            total_pe_energy = self.calculate_comparator_energy() * self.Hidden_layer_num
            total_CIP_energy = total_pd_energy + total_pe_energy

        return total_APS_energy, total_pd_energy, total_pe_energy, total_CIP_energy
    
    def calculate_energy_CIM(self):
        log_print("    Start calculate CIM_energy   ")

        self.Total_energy_compute = (total_CIP_energy) * self.times 
        self.Total_energy_sense = total_APS_energy

        self.Total_energy = self.Total_energy_compute + self.Total_energy_sense

        log_print('  --Total Energy', self.Total_energy)
        log_print('     --energy_APS_sense', self.Total_energy_sense)
        log_print('     --energy_compute', self.Total_energy_compute)


        self.power = self.Total_energy/self.latency *P.FREQUENCY
        log_print("  --Power:",self.power)




    def processing_one_times(self,data):
        self.data = data
        self.times +=1

def get_nominal_supply(tech_node):
    """ Compute nominal supply voltage under different process nodes.

    Args:
        tech_node (int): [unit: nm] pixel's process node.

    Returns:
        float: supply voltage [unit: V]
    """

    if 130 < tech_node <= 180:
        supply = 1.8
    elif 65 < tech_node <= 130:
        supply = 1.5
    elif tech_node <= 65:
        supply = 1.1
    else:
        raise Exception("Defined tech_node is not supported.")
    return supply

def get_pixel_parasitic(
        array_v,
        tech_node,  # [nm]
        pitch  # [um]
):
    """ Compute column parasitic capacitance (Farad per column) on the pixel's output node.

    Args:
        array_v (int): vertical size of the pixel array.
        tech_node (int): [unit: nm] pixel's process node.
        pitch (float): [unit: um] pixel's pitch size.

    Returns:
        float: parasitic capacitance [unit: F]
    """

    C_p = 9e-15 / 130 / 5 * tech_node * pitch * array_v
    return C_p

if __name__ == '__main__':
    
    hw_Config_path = './config.ini'
    hw_config = configparser.ConfigParser()
    hw_config.read(hw_Config_path, encoding = 'UTF-8')

    total_APS_energy, total_pd_energy, total_pe_energy, total_CIP_energy = CIP_Array(hw_config).calculate_total_energy(NNtype='MLP')
    print("total_pixel_energy:", total_APS_energy)
    print("total_pd_energy:", total_pd_energy)
    print("total_PE_energy:", total_pe_energy)
    print("total_CIP_energy:", total_CIP_energy)       