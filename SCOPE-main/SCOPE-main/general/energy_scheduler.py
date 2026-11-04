import math
import pdb
from general.logger import log_print
from hardware.PMU import PMU
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def scheduling(total_result, V_max=1.2, V_Li= 0.8):
    Cap = 0
    V_Hi = [V_Li]
    for i in range(len(total_result)):
        energy = total_result[i]['energy']

        C_st = 2* energy/(V_max**2 - V_Li**2)
        if C_st >Cap: Cap = C_st
    
    log_print("Minimum Capacitance Value:", Cap)

    for i in range(len(total_result)):
        energy = total_result[i]['energy']
        V_Hi.append(math.sqrt(2*energy/Cap + V_Li**2))

    log_print("Each activation VoltageV_Hi:",V_Hi)

    return Cap, V_Hi

def timing_plot(result, V_h, timing):
    t_total = []
    V_total = []
    for i in range(1,len(timing)):

        if i == 1:
            t_total.append(timing[0])
        
        t_total.append(timing[i])
        V_total.append(V_h[0])
        t_total.append(result[i-1]["time"])
        V_total.append(V_h[i])

        if i == len(timing)-1 :
            V_total.append(V_h[0])
            

    cumulative_time = np.cumsum(t_total)
    pdb.set_trace()
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_time, V_total, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title("Voltage vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Frame", 1/cumulative_time[-1])


def timing_plot_streamlit(result, V_h, timing):
    t_total = []
    V_total = []
    
    for i in range(1, len(timing)):
        if i == 1:
            t_total.append(timing[0])
        
        t_total.append(timing[i])
        V_total.append(V_h[0])
        t_total.append(result[i - 1]["time"])
        V_total.append(V_h[i])

        if i == len(timing) - 1:
            V_total.append(V_h[0])

    cumulative_time = np.cumsum(t_total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cumulative_time, V_total, marker='o')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Voltage vs Time")
    ax.grid(True)
    fig.tight_layout()

    st.pyplot(fig)



if __name__ == "__main__":
    total_energy = {0: {'energy': 2.1884925184499985e-08, 'time': 0.0003243, 'power': 6.748358058741902e-05}, 1: {'energy': 2.9039133502250028e-08, 'time': 0.0003244, 'power': 8.951644112900748e-05}, 2: {'energy': 2.775550092900003e-08, 'time': 0.0003243, 'power': 8.55858801387605e-05}, 3: {'energy': 2.934988457299993e-08, 'time': 0.0003244, 'power': 9.047436674784195e-05}, 4: {'energy': 2.8080124592999958e-08, 'time': 0.0003243, 'power': 8.65868781776132e-05}, 5: {'energy': 1.3019302513250002e-08, 'time': 0.0001708, 'power': 7.622542455064404e-05}, 6: {'energy': 6.2215757552499845e-09, 'time': 8.59e-05, 'power': 7.242812287834673e-05}, 7: {'energy': 6.600568187999986e-09, 'time': 8.6e-05, 'power': 7.675079288372078e-05}, 8: {'energy': 6.24576095350002e-09, 'time': 8.59e-05, 'power': 7.2709673498254e-05}, 9: {'energy': 2.4864327435000012e-09, 'time': 4.76e-05, 'power': 5.2235982006302545e-05}, 10: {'energy': 1.5226556747499998e-09, 'time': 2.43e-05, 'power': 6.266072735596707e-05}, 11: {'energy': 1.6165446884999999e-09, 'time': 2.44e-05, 'power': 6.625183149590163e-05}, 12: {'energy': 1.5259748072500004e-09, 'time': 2.43e-05, 'power': 6.279731717078191e-05}}

    Cap, V_Hi = scheduling(total_energy)
    log_print(V_Hi)

    timing = PMU.PMU_run(Cap, V_Hi, v_li= V_Hi[0])

    timing_plot(V_Hi, timing)

    
