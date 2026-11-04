import streamlit as st
import configparser
from io import StringIO
import torch
import time
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import gc
import importlib

import main_performance 
from main_accuracy import accuracy_inference, unpickle
from hardware.PMU import PMU
from model.resnet import resnet16

import general.parameters as parameters
from general.adc_noise import *
from general.logger import log_print
from model.modules.replace_quant import *
from general.energy_scheduler import scheduling, timing_plot_streamlit
from hardware.PMU.pmu_visualizer import eval_pmu

from general.param_runtime import P, bump_version, set_cfg_path

set_cfg_path("./config.ini")



HELP_TEXTS = {
    "Device level.device_type": "1: SRAM, other values can be left blank",
    "Device level.device_tech": "Process technology node, unit: nm",
    "Device level.input_size": "Image size: height & width",
    "Device level.activation_bit": "Activation bit width",
    "Device level.weight_bit": "Weight bit width",
    "Device level.frequency": "Operating frequency, unit: Hz",
    "Device level.supply_voltage": "Supply voltage, unit: V",
    "CIM Array level.bank_num": "Number of rows, columns",
    "CIM Array level.column_num": "Number of output channels per bank",
    "CIM Array level.cycle_cim_per_mac": "Cycles per MAC operation",
    "CIM Array level.bit_serial": "CIM computation, 1 = True (serial), 0 = parallel",
    "CIM Array level.buffer_choice": "Input mode to CIM — 1: FIFO; 2: REG_array; 3: Buffer",
    "Cell level.cell_num": "Number of cells forming one column before ADC conversion, equals kernel size × minimum input channels",
    "Computing Unit level.unit_type": "Type of compute-in-memory, e.g., 1: SRAM",
    "Computing Unit level.num_unit": "How many units are multiplexed into one",
    "Computing Unit level.energy_per_operation": "Energy per MAC operation, unit: J",
    "Computing Unit level.energy_per_write": "Energy per single memory write, unit: J",
    "Interface level.adc_precision": "ADC resolution (bits)",
    "Interface level.energy_per_conversion": "ADC figure of merit (J/conversion)",
    "Interface level.adc_sampling_rate": "ADC sampling rate (Hz)",
    "Interface level.adc_supply_voltages": "ADC supply voltage (V)",
    "Interface level.adc_area": "ADC area (µm²)",
    "Digital module.digital_area": "Digital module area (µm²)",
    "Digital module.energy": "Energy, unit: J",
    "Architecture level.array_num": "Number of arrays, whether reused",
    "Architecture level.buffer_capacity": "Buffer capacity, unit: kB",
    "Architecture level.buffer_wordwidth": "Buffer word width, unit: b",
    "Architecture level.buffer_type": "1 = SRAM",
    "Algorithm Configuration.cap_variation": "Capacitance variation (%)",
    "Algorithm Configuration.noise": "Noise level (%)",
    "Algorithm Configuration.cap_parasitic": "Parasitic capacitance (F)",


    # ... More to added
}

OPTIONS = {
    "Device level.device_tech": [22, 28, 40, 65, 180],
    "CIM Array level.buffer_choice": [1,2,3],
    "Computing Unit level.unit_type": [1],
    "Digital module.activation_type":["hardtanh"],
    "Architecture level.buffer_type":[1],    
}

EXTRA_FIELDS = {
    "dac_option": [
        {"label": "DAC precision(bits", "type": "int", "min": 1, "max": 16, "value": 8, "target": ("Interface level", "DAC_bits")}, 
    ],
    "adc_option": [
        {"label": "adc_precision", "type": "int", "min": 1, "max": 16, "value": 8, "target": ("Interface level", "adc_precision")},
        {"label": "energy_per_conversion", "type": "float", "value": 0.0, "step": 0.01, "fmt": "%.3f", "target": ("Interface level", "energy_per_conversion")},
        {"label": "adc_sampling_rate", "type": "int", "value": 800000000, "target": ("Interface level", "adc_sampling_rate")},
        {"label": "adc_supply_voltages", "type": "float", "value": 0.9, "step": 0.1, "fmt": "%.2f", "target": ("Interface level", "adc_supply_voltages")},
        {"label": "adc_area", "type": "float", "value": 7.4e-09, "step": 0.01, "fmt": "%.3f", "target": ("Interface level", "adc_area")},
    ],
}

def upsert(cfg, sec, opt, val): #多加config的section
    if not cfg.has_section(sec):
        cfg.add_section(sec)
    cfg[sec][opt] = str(val)

def load_config(path ="./config.ini"):
    config = configparser.ConfigParser()
    config.read(path, encoding='UTF-8')
    return config

def save_config(config, config_path):
    with open(config_path, 'w', encoding='UTF-8') as f:
        config.write(f)

if __name__ == "__main__":
    path = "./config.ini"
    Hw_config = load_config(path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "running_inference" not in st.session_state:
        st.session_state.running_inference = False
    
    if "last_network_choice" not in st.session_state:
        st.session_state.last_network_choice = ""

    with st.sidebar:
        st.title("🔧 Senselab ToolChain")

        page = st.radio(
            "Select Function",
            ["Configuration Editor", "Performance Evaluation Simulation", "Accuracy Evaluation Simulation"]
        )

        ## Select configuration section
        if page == "Configuration Editor":
            section = st.selectbox(
                "Select Configuration Section",
                options=Hw_config.sections(),
                index=0
            )

    if page == "Configuration Editor":
        st.header("⚙️ Hardware Configuration Editor")
        if st.session_state.last_network_choice != "Configuration Editor":
            st.session_state.last_network_choice = "Configuration Editor"
        
        st.subheader(f"Current Configuration Block: **[{section}]**")

        # Determined by True/False, usually not displayed
        for key in Hw_config[section]:
            if key.lower() in("adc_precision", "energy_per_conversion", "adc_sampling_rate", "adc_supply_voltages","adc_area",
                              "cycle_per_cip","pe_size", "time_exposure", "cip_area", "pixel_img_area", "pe_area", "pe_num", "buffer_area",
                              "pixel_harvest_area_1","pixel_harvest_num_1","pixel_harvest_area_2","pixel_harvest_num_2","pixel_harvest_area_3","pixel_harvest_num_3"):
                continue

            current_value = Hw_config[section][key].strip()

            help_text = HELP_TEXTS.get(f"{section}.{key}", "")
            options = OPTIONS.get(f"{section}.{key}", None)

            # if section == "Device level" and key.lower() == "device_tech".lower():
            if key.lower() in ["device_tech", "buffer_choice","unit_type","activation_type","buffer_type"]:  #selectbox
                current_val = str(current_value)
                new_value = st.selectbox(
                    f"{key}",
                    options=[str(opt) for opt in options],
                    index=options.index(current_val) if current_val in options else 0,
                    help=help_text
                )

            elif key.lower() in ["bit_serial", "switch", "dac_option","adc_option","inference", "sensor"]: #True/False
                current_val = True if str(current_value) == "1" else False
                new_bool = st.selectbox(
                    f"{key} (True/False)",
                    options=[True, False],
                    index=0 if current_val else 1,
                    help=help_text
                )
                new_value = "1" if new_bool else "0"


                Hw_config[section][key] = str(new_value)
                trigger_key = f"{section}.{key}"
                match = next((t for t in EXTRA_FIELDS.keys() if t.lower() == trigger_key.lower()), None)

                if new_bool and key.lower() in (k.lower() for k in EXTRA_FIELDS.keys()): #shown only true
                    st.markdown(f"**🔧 {key.lower()} Extra Configuration**")

                    for trigger, fields in EXTRA_FIELDS.items():
                        if trigger.lower() != key.lower(): 
                            continue
                        for f in fields:
                            target_sec, target_key = f["target"]
                            if not Hw_config.has_section(target_sec):
                                Hw_config.add_section(target_sec)
                            wkey = f"{target_sec}.{target_key}"  

                            if f["type"] == "select":
                                val = st.selectbox(f["label"], f["options"],help=help_text, key=wkey)
                            elif f["type"] == "int":
                                val = st.number_input(f["label"], min_value=f.get("min", 0), max_value=f.get("max", 10**9),
                                                    value=f.get("value", 0), step=1,help=help_text, key=wkey)
                            elif f["type"] == "float":
                                val = st.number_input(f["label"], value=f.get("value", 0.0), step=f.get("step", 0.1),
                                                    format=f.get("fmt", "%.4f"),help=help_text, key=wkey)
                            else:
                                val = st.text_input(f["label"], value=f.get("value",""),help=help_text, key=wkey)

                            Hw_config[target_sec][target_key] = str(val)
                    st.markdown("---")
                    
                else:
                    if key.lower() in (k.lower() for k in EXTRA_FIELDS.keys()):
                        for trigger, fields in EXTRA_FIELDS.items():
                            if trigger.lower() != key.lower(): 
                                continue
                            for f in fields:
                                target_sec, target_key = f["target"]
                                if Hw_config.has_option(target_sec, target_key):
                                    Hw_config.remove_option(target_sec, target_key)
                

            elif ',' in current_value: #two values 
                parts = [p.strip() for p in current_value.split(',')]
                new_parts = []
                st.write(f"**{key}(Length, Width)**")
                for i, p in enumerate(parts):
                    try:
                        p_val = p
                    except:
                        p_val = float(p)
                    new_p = st.number_input(
                        f"{key} - {i+1}",
                        value=int(p_val),
                        step=1,
                        help=help_text
                    )
                    new_parts.append(str(new_p))
                new_value = ','.join(new_parts)

            else:
                try:
                    num_val = float(current_value.strip())
                    if current_value.strip() == "0":
                        new_value = st.number_input(
                            key,
                            value=int(num_val),
                            step=1,
                            help=help_text
                        )
                    elif current_value.strip() == "0.0":
                        new_value = st.number_input(
                            key,
                            value=0.0,
                            step=0.1,
                            help=help_text,
                            format="%.6e" if abs(num_val) < 1e-3 or abs(num_val) > 1e3 else "%.6f"  
                        )

                    elif '.' in current_value or 'e' in current_value.lower():
                        # float
                        new_value = st.number_input(
                            key,
                            value=num_val,
                            step=0.1,
                            help=help_text,
                            format="%.4e" if abs(num_val) < 1e-3 or abs(num_val) > 1e3 else "%.4f"
                        )
                    else:
                        # int
                        new_value = st.number_input(
                            key,
                            value=int(num_val),
                            step=1,
                            help=help_text
                        )
        
                except:
                    # 否则用文本输入
                    new_value = st.text_input(
                        key,
                        value=current_value,
                        help=help_text
                    )

            Hw_config[section][key] = str(new_value)

        if section =="Sensor level":
            if Hw_config[section]["sensor"]=="1":
                
                st.markdown("---")
                AREA_KEY_PREFIX = "pixel_harvest_area_"
                NUM_KEY_PREFIX  = "pixel_harvest_num_"
                ss_count_key    = f"{section}.pixel_img_area_count"


                indices = set()
                for k in Hw_config[section]:
                    kl = k.lower()
                    if kl.startswith(AREA_KEY_PREFIX) or kl.startswith(NUM_KEY_PREFIX):
                        suf = k.split("_")[-1]
                        if suf.isdigit():
                            indices.add(int(suf))

                if ss_count_key not in st.session_state:
                    st.session_state[ss_count_key] = max(indices) if indices else 0
                st.session_state[ss_count_key] = max(st.session_state[ss_count_key], max(indices) if indices else 0)

                to_delete = []
                N = st.session_state[ss_count_key]
                for i in range(1, N + 1):
                    area_key = f"{AREA_KEY_PREFIX}{i}"
                    num_key  = f"{NUM_KEY_PREFIX}{i}"
    
                    w_def = float(Hw_config[section].get(area_key, "0.0")) if Hw_config.has_option(section, area_key) else 0.0
                    num_def = int(float(Hw_config[section].get(num_key, "0"))) if Hw_config.has_option(section, num_key) else 0

                    cols = st.columns([4, 2, 2])
                    with cols[0]:
                        w_val = st.number_input(
                            f"{area_key}",
                            value=float(w_def),        
                            min_value=0.0,             
                            step=0.1,                 
                            format="%.4e",
                            key=f"{section}.{area_key}.w",
                        )
                    with cols[1]:
                        num_val = st.number_input(f"{num_key}", value=num_def, step=1, min_value=0, key=f"{section}.{num_key}")

                    # write it to config.ini
                    Hw_config[section][area_key] = f"{w_val}"
                    Hw_config[section][num_key]  = str(num_val)

                    del_cols = st.columns([1, 4])
                    with del_cols[0]:
                        if st.button("🗑 Delete ", key=f"del.{section}.{i}"):
                            to_delete.append(i)

                    st.markdown("---")
                if to_delete:
                    for i in to_delete:
                        for k in (f"{AREA_KEY_PREFIX}{i}", f"{NUM_KEY_PREFIX}{i}"):
                            if Hw_config.has_option(section, k):
                                Hw_config.remove_option(section, k)
                    st.session_state[ss_count_key] = max(0, N - len(to_delete))
                    st.experimental_rerun()

                # Adding group
                if st.button("➕ Add New Set"):
                    st.session_state[ss_count_key] += 1
                    i = st.session_state[ss_count_key]
                    Hw_config[section][f"{AREA_KEY_PREFIX}{i}"] = Hw_config[section].get(f"{AREA_KEY_PREFIX}{i}", "0,0")
                    Hw_config[section][f"{NUM_KEY_PREFIX}{i}"]  = Hw_config[section].get(f"{NUM_KEY_PREFIX}{i}", "0")
                    st.experimental_rerun()

        # Save button
        st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)
        if st.button("💾 Save Configuration"):
            save_config(Hw_config, path)
            bump_version() # Reload params (important)
            st.success("Configuration saved successfully!")
            time.sleep(1)

            st.experimental_rerun()

    elif page == "Performance Evaluation Simulation":
        if  st.session_state.last_network_choice != "Performance Evaluation Network":
            Hw_config = load_config(path)
            for key in ["processor_sim", "pmu_sim", "calculator", "performance"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.last_network_choice = "Performance Evaluation Network"

        subpage = st.sidebar.selectbox(
            "Select Simulation Type",
            ["System-level Simulation","PMU Only Simulation"]
        )

        if subpage == "System-level Simulation":
            st.header("⚙️ Performance Evaluation Simulation")
             
            name = st.selectbox(
                "Select Network Type", 
                ["-- Please Select --", "ResNet16"],
                help="Currently only ResNet is supported",
                key="Performance Evaluation Network"
                )

            if name == "ResNet16":
                with st.expander("📦 ResNet Configuration"):
                    st.write("Loading ResNet16 network...")
                    model = resnet16().to(device)
                    model_path = st.text_input("Enter Pretrained Model Parameters", 
                                            value="model/resnet_44000.pth", 
                                            help="Path to the pretrained model"
                                            )

                if model_path.strip():
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                model.eval()


                if "processor_sim" not in st.session_state:
                    st.session_state.processor_sim = False

                if "pmu_sim" not in st.session_state:
                    st.session_state.pmu_sim = False

                if "calculator" not in st.session_state:
                    st.session_state.calculator = None

                if "performance" not in st.session_state:
                    st.session_state.performance = None
                
                if not st.session_state.processor_sim:
                    if st.button("🚀 Confirm Network & Generate Hardware Flow"):
                        st.session_state.log_buffer = []

                        calculator = main_performance.ChipPerformanceCalculator(sw_config=model, hw_config=Hw_config, network_type= "ResNet")
                        for msg in getattr(calculator, "warnings", []):
                            st.warning(f"⚠️ {msg}")
                        
                        st.session_state.calculator = calculator

                        if INFERENCE:
                            model = replace_forward_function(model) 
                            model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
                            print("replaced model with IRConv2d_CIM")
                        else:
                            model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)

                        model = replace_forward(model, target_performance=1)

                        h,w = INPUT_SIZE[0], INPUT_SIZE[1]

                        dummy_input = torch.randn(1, 3, h, w).to(device)
                        model(dummy_input)

                        st.session_state.processor_sim = True  

                        st.info("✅ Hardware flow and cycle calculation generated successfully! Please proceed with the simulation.")
                
                if st.session_state.processor_sim and not st.session_state.pmu_sim:
            
                    if st.button("🚀 Run Processor Simulation"):
                        st.write("------------------Processor Simulation Started------------------")    
                        calculator = st.session_state.calculator               
                        performance_result = calculator.total_chip_energy(tensor_list)
                        st.session_state.performance = performance_result

                        st.success("✅ Processor simulation completed!")
                
                        # Cap, V_Hi = scheduling(performance_result, V_max= SUPPLY_VOLTAGE*1.1, V_Li= SUPPLY_VOLTAGE*0.9)
                        st.session_state.pmu_sim = True
                
                if st.session_state.pmu_sim:
                    if st.button("🚀 Run PMU Simulation"):
                        performance_result = st.session_state.performance

                        Cap, V_Hi = scheduling(performance_result, V_max= SUPPLY_VOLTAGE*1.1, V_Li= SUPPLY_VOLTAGE*0.9)
                        
                        st.write("------------------PMU Simulation Started------------------") 
                        timing = PMU.PMU_run(Cap, V_Hi, v_li= V_Hi[0])

                        st.success("✅ PMU simulation completed!")

                        st.write("-----------Generated Execution Plot under Actual Energy Harvesting------------") 
                        timing_plot_streamlit(performance_result, V_Hi, timing)

        elif subpage == "PMU Only Simulation":
            eval_pmu()

    elif page == "Accuracy Evaluation Simulation":
        st.header("⚙️ Accuracy Evaluation Simulation")

        if  st.session_state.last_network_choice != "Accuracy Evaluation Simulation":
            st.session_state.running_inference = False
            st.session_state.last_network_choice = "Accuracy Evaluation Simulation"

        st.session_state.log_buffer = [] 


        if "model" not in st.session_state:
            st.session_state.model = None
        if "model_final" not in st.session_state:
            st.session_state.model_final = None
        if "model_path" not in st.session_state:
            st.session_state.model_path = None
        if "dataset_path" not in st.session_state:
            st.session_state.dataset_path = None
        if "LOG_AREA" not in st.session_state:
            st.session_state.LOG_AREA = st.empty()

            

        name = st.selectbox(
            "Select Network Type", 
            ["-- Please Select --", "ResNet16"],
            help="Currently only ResNet is supported",
            key="Accuracy Evaluation Network"
        )   
        if name == "ResNet16":
            with st.expander("📦 ResNet Configuration"):
                
                model_path = st.text_input("Enter Pretrained Model Path", value="model/resnet_44000.pth", help= "Path to the pretrained model")
                st.session_state.model_path = model_path

        task, dataset = st.columns(2)
        with task:
            task = st.selectbox(
                "Select Task Type",
                ["-- Please Select --", "Classification Task"],
                key="task_type"
            )
        with dataset:
            if task == "Classification Task":
                dataset = st.selectbox(
                    "Select Dataset",
                    ["-- Please Select --", "CIFAR-10", "Pedestrian Binary Classification"],
                    key="dataset"
                )

                if dataset == "CIFAR-10":
                    dataset_path = "dataset/test_batch"
                    if "model_inf" not in st.session_state:
                        st.session_state.model_inf = resnet16(num_classes=10).to(device)
                    # st.session_state.model = model
                    st.session_state.dataset_path = dataset_path
                elif dataset == "Pedestrian 2-Classification":
                    st.warning("⚠️ Currently only CIFAR-10 is supported!")


        if "visualize" not in st.session_state:
            st.session_state.visualize = True
        

        if name == "ResNet16" and task =="Classification Task" and dataset == "CIFAR-10":
 


            if st.button("Accuracy Evaluation"):
                st.session_state.running_inference = True
                st.session_state.step = 0
                st.session_state.correct_num = 0
                model = st.session_state.model_inf
                model_path = st.session_state.model_path
                

                if model_path.strip():
                    state_dict = torch.load(model_path, map_location=device)
                    model.load_state_dict(state_dict)
                model.eval()

                if INFERENCE:
                    model = replace_forward_function(model) 
                    model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d_cim)
                    print("replaced model with IRConv2d_CIM")
                else:
                    model = replace_conv_with_irconv(model, ir_1w8a.IRConv2d)

                model = replace_forward(model, target_performance=1)
                st.session_state.model_final = model
                variation_list = generate_variation()
                parastic_list = generate_parastic()
                P.cap_variation_tensor = torch.tensor(variation_list)
                P.cap_parasitic_tensor = torch.tensor(parastic_list)

                
                
          
                

                
            if st.session_state.running_inference:
                model = st.session_state.model_final
                dataset_path = st.session_state.dataset_path

                st.write("--------Start accuracy evaluation--------") 


                if st.button("🔄 Image Visualization"):
                    st.session_state.visualize = not st.session_state.visualize

                status = "ON" if st.session_state.visualize else "OFF"
                st.write(f"Visualization is currently: {status}")
     
            

                image_area = st.empty()
                i = st.session_state.step

                #cycle
                testXtr = unpickle(dataset_path)

                num_test = len(testXtr['labels'])

                if "max_steps" not in st.session_state:
                    st.session_state.max_steps = num_test
                if "y_out" not in st.session_state:
                    st.session_state.y_out = np.zeros(1) 
                if "y_label" not in st.session_state:
                    st.session_state.y_label = np.zeros(1)

                if st.session_state.step == 0:
                    st.session_state.y_out = np.zeros(num_test)
                    st.session_state.y_label = np.zeros(num_test)

                label_map = {
                    0: "airplane",
                    1: "car",
                    2: "bird",
                    3: "cat",
                    4: "deer",
                    5: "dog",
                    6: "frog",
                    7: "horse",
                    8: "ship",
                    9: "truck",
                }
                
                img = np.reshape(testXtr['data'][i], (3, 32, 32))
                img = img.transpose(1, 2, 0)
                img_show = img.astype(np.uint8)

                img_data = transforms.ToTensor()(img)
                img_data = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_data)
                img_data = torch.reshape(img_data, (1,3,32,32)).to(device)

                with torch.no_grad():
                    output = model(img_data)
                    output = output.detach().cpu().numpy()
                
                    st.session_state.y_label[i] = testXtr['labels'][i]
                    st.session_state.y_out[i] = np.argmax(output, axis = 1)

                    del output
                    del img_data
                torch.cuda.empty_cache()
                gc.collect()

                st.session_state.step += 1
                if st.session_state.y_label[i]== st.session_state.y_out[i]:
                    st.session_state.correct_num += 1
                st.info(f"Accuracy:{st.session_state.correct_num/st.session_state.step*100} %")

                if st.session_state.visualize:
                    col1, col2 = image_area.columns(2)

                    with col1:
                        st.image(img_show, caption=f"Frame {i}", channels="RGB", use_column_width=True)

                    with col2:
                        st.markdown(
                            f"""
                            <div style='font-size:20px; font-weight:bold; color:#333;'>
                                ✅ Pred: <span style='color:green'>{label_map[st.session_state.y_out[i]]}</span><br>
                                🔮 True: <span style='color:blue'>{label_map[st.session_state.y_label[i]]}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    time.sleep(1.5)  

                log_print(str(i) + '_' + str(testXtr['labels'][i]) + '\tpredicted class:', st.session_state.y_out[i]) 
                
                

                if st.session_state.step < st.session_state.max_steps:
                    
                    st.experimental_rerun()
                else:
                    st.success("✅ Inference Completed!")
                    st.session_state.running_inference = False

                    cm = confusion_matrix(st.session_state.y_label, st.session_state.y_out)

                    sum_correct = 0
                    for k in range(10):
                        sum_correct += cm[k, k]
                    log_print('sum_correct: ',sum_correct)
                    log_print('Accuracy(%):', sum_correct/num_test)


                




        

            
    






        


