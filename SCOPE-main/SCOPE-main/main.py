import argparse
import os
import sys
import time
import configparser
import torch
import pickle
import torchvision.transforms as transforms
import numpy as np
import pdb
from general.logger import log_print
import argparse

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from general.mapping import mapping
from main_performance import ChipPerformanceCalculator
from main_accuracy import accuracy_inference
from hardware.PMU import PMU
from torchvision.models import vgg16
from model.modules import ir_1w8a
from model.resnet import resnet16
from general import parameters
from general.parameters import *
from general.adc_noise import *
from model.modules.replace_quant import *
from general.energy_scheduler import scheduling, timing_plot

def parse_args():
    parser = argparse.ArgumentParser(description='Tool_Chain')
    parser.add_argument("--config",type=str,default="./config.ini",help="Path to hardware configuration file (e.g., config.ini)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    log_print("--------------------------Start Simulation!----------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet16().to(device)
    # state_dict = torch.load("model/resnet_44000.pth", map_location=device)
    # model.load_state_dict(state_dict)
    model.eval()

    calculator = ChipPerformanceCalculator(sw_config=model, hw_config=Hw_config, network_type= "ResNet")
    if INFERENCE:
        model = replace_forward_function(model) 
        model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
        print("replaced model with IRConv2d_CIM")
    else:
        model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)
    

    model = replace_forward(model, target_performance=1)

    variation_list = generate_variation()
    parastic_list = generate_parastic()
    parameters.cap_variation_tensor = torch.tensor(variation_list).to(device)
    parameters.cap_parasitic_tensor = torch.tensor(parastic_list).to(device)

    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    model(dummy_input)


    # HwConfig_path = './config.ini'
    args  = parse_args()
    HwConfig_path = args.config
    Hw_config = configparser.ConfigParser()
    Hw_config.read(HwConfig_path, encoding = 'UTF-8')


    
    performance_result = calculator.total_chip_energy(tensor_list)  
    
    Cap, V_Hi = scheduling(performance_result, V_max= SUPPLY_VOLTAGE*1.1, V_Li= SUPPLY_VOLTAGE*0.9)

    timing = PMU.PMU_run(Cap, V_Hi, v_li= V_Hi[0])
    print(timing)
    timing_plot(performance_result,V_Hi, timing)

    # area = ChipAreaCalculator(sw_config=network, hw_config= Hw_config)
    # log_print("Total Chip Area:", area.total_chip_area()*1e6, "mm^2")

    dataset_path = "dataset/test_batch"
    accuracy_inference(model, dataset_path)


