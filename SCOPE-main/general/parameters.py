from enum import Enum
import configparser
import pdb
from dataclasses import dataclass
from typing import Optional
import sys 
import os
import torch

DEFAULT_CFG_PATH = os.path.join(os.path.dirname(__file__), "../config.ini")
Hw_config = configparser.ConfigParser()
Hw_config.read(DEFAULT_CFG_PATH, encoding='UTF-8')

DEVICE_TECH = int(Hw_config.get('Device level', 'Device_Tech'))
FREQUENCY = float(Hw_config.get('Device level', 'Frequency'))
SUPPLY_VOLTAGE = float(Hw_config.get('Device level', 'Supply_voltage'))

INPUT_SIZE = list(map(int, Hw_config.get('Device level', 'Input_size').split(',')))
SENSOR = int(Hw_config.get('Sensor level', 'Sensor'))
CYCLE_CIP = int(Hw_config.get('Sensor level', 'Cycle_per_CIP'))
if SENSOR != 0 and SENSOR!= 1 :
    raise ValueError(f"传感器需为0或1")

INPUT_BUFFER = int(Hw_config.get('CIM Array level', 'buffer_choice'))
BANK_NUM =list(map(int, Hw_config.get('CIM Array level', 'Bank_num').split(','))) #4,4
COLUMN_NUM = int(Hw_config.get('CIM Array level', 'Column_num')) #16
CELL_NUM = int(Hw_config.get('Cell level', 'Cell_num')) #144
NUM_UNIT = int(Hw_config.get('Computing Unit level', 'num_unit')) #16

CYCLE_CIM = int(Hw_config.get('CIM Array level', 'Cycle_CIM_per_MAC'))
BIT_SERIAL = int(Hw_config.get('CIM Array level', 'Bit_serial'))

ACTIVATION_BIT = int(Hw_config.get('Device level', 'Activation_bit'))
WEIGHT_BIT = int(Hw_config.get('Device level', 'Weight_bit'))
ADC_BIT = int(Hw_config.get('Interface level', 'ADC_precision', fallback=0))

INFERENCE = int(Hw_config.get('Algorithm Configuration', 'Inference'))
NOISE = float(Hw_config.get('Algorithm Configuration', 'noise',fallback=0))
CAP_VARIATION = float(Hw_config.get('Algorithm Configuration', 'cap_variation',fallback=0))
CAP_PARASITIC = float(Hw_config.get('Algorithm Configuration', 'Cap_parasitic',fallback=0))
cap_variation_tensor = torch.tensor([])
cap_parasitic_tensor = torch.tensor([])

MAX_CYCLE_CNT = 20000000

@dataclass(frozen=True)
class Params:
    DEVICE_TECH: int
    FREQUENCY: float
    SUPPLY_VOLTAGE: float
    INPUT_SIZE: list
    SENSOR: int
    CYCLE_CIP: int
    INPUT_BUFFER: int
    BANK_NUM: list
    COLUMN_NUM: int
    CELL_NUM: int
    NUM_UNIT: int
    CYCLE_CIM: int
    BIT_SERIAL: int
    ACTIVATION_BIT: int
    WEIGHT_BIT: int
    ADC_BIT: int
    INFERENCE: int
    NOISE: float
    CAP_VARIATION: float
    CAP_PARASITIC: float
    cap_variation_tensor: torch.Tensor
    cap_parasitic_tensor: torch.Tensor
    MAX_CYCLE_CNT: int = 20_000_000

def _read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg

def load_params(path: Optional[str] = None, build_tensors: bool = True) -> Params:
    cfg = _read_config(path or DEFAULT_CFG_PATH)
    DEVICE_TECH = int(cfg.get('Device level', 'device_tech'))
    FREQUENCY   = float(cfg.get('Device level', 'Frequency'))
    SUPPLY_VOLTAGE = float(cfg.get('Device level', 'Supply_voltage'))

    INPUT_SIZE  = list(map(int, cfg.get('Device level', 'Input_size').split(',')))
    SENSOR      = int(cfg.get('Sensor level', 'Sensor'))
    CYCLE_CIP   = int(cfg.get('Sensor level', 'Cycle_per_CIP'))
    if SENSOR not in (0, 1):
        raise ValueError("Sensor should be 0 or 1")

    INPUT_BUFFER= int(cfg.get('CIM Array level', 'buffer_choice'))
    BANK_NUM    = list(map(int, cfg.get('CIM Array level', 'Bank_num').split(',')))
    COLUMN_NUM  = int(cfg.get('CIM Array level', 'Column_num'))
    CELL_NUM    = int(cfg.get('Cell level', 'Cell_num'))
    NUM_UNIT    = int(cfg.get('Computing Unit level', 'num_unit'))

    CYCLE_CIM   = int(cfg.get('CIM Array level', 'Cycle_CIM_per_MAC'))
    BIT_SERIAL  = int(cfg.get('CIM Array level', 'Bit_serial'))

    ACTIVATION_BIT = int(cfg.get('Device level', 'Activation_bit'))
    WEIGHT_BIT     = int(cfg.get('Device level', 'Weight_bit'))
    ADC_BIT        = int(cfg.get('Interface level', 'ADC_precision', fallback=0))

    INFERENCE   = int(cfg.get('Algorithm Configuration', 'Inference'))
    NOISE       = float(cfg.get('Algorithm Configuration', 'noise', fallback=0))
    CAP_VARIATION  = float(cfg.get('Algorithm Configuration', 'cap_variation', fallback=0))
    CAP_PARASITIC  = float(cfg.get('Algorithm Configuration', 'Cap_parasitic', fallback=0))

    if build_tensors:
        cap_variation_tensor = torch.tensor([])  
        cap_parasitic_tensor = torch.tensor([])
    else:
        cap_variation_tensor = torch.empty(0)
        cap_parasitic_tensor = torch.empty(0)

    return Params(
        DEVICE_TECH, FREQUENCY, SUPPLY_VOLTAGE,
        INPUT_SIZE, SENSOR, CYCLE_CIP,
        INPUT_BUFFER, BANK_NUM, COLUMN_NUM, CELL_NUM, NUM_UNIT,
        CYCLE_CIM, BIT_SERIAL, ACTIVATION_BIT, WEIGHT_BIT, ADC_BIT,
        INFERENCE, NOISE, CAP_VARIATION, CAP_PARASITIC,
        cap_variation_tensor, cap_parasitic_tensor
    )








