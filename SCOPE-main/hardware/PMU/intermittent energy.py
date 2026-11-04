import numpy as np
import matplotlib.pyplot as plt
import pdb
from OSC import Oscillator
from RMCP import ReconfigurableMainCharge_Pump

# 能量供给间断，输入电压间断
def generate_intermittent_v_eh_bus(t):
    # 生成间断的v_eh_bus的值
    v_eh_bus_list = np.zeros_like(t)  # 初始化电压值为 0
    for i, time_point in enumerate(t):
        if 0 <= time_point < 1 or 2 <= time_point < 3 or 4 <= time_point <= 5:  # 在这些时间段内生成正弦波
            v_eh_bus_list[i] = 0.32 + 0.02 * np.sin(2 * np.pi * time_point)
        else:  # 其他时间段电压为 0
            v_eh_bus_list[i] = 0
    return v_eh_bus_list

def main():
    # 生成时间序列
    t = np.linspace(0, 5, 700)  # 时间序列

    # 生成间断的 v_eh_bus 值
    v_eh_bus_list = generate_intermittent_v_eh_bus(t)

    # OSC参数设置
    C1 = 206e-15
    C2 = 206e-15
    v_target = 1.2
    C_storage = 50e-6

    # 创建Oscillator实例
    oscillator = Oscillator(v_eh_bus_list, C1, C2, v_target, C_storage)

    # 运行模拟并处理结果
    results = oscillator.run_simulation(0)

    if isinstance(results, dict):
        t_results = results.get('t_charge', [])
    else:
        try:
            t_results = results[0]  # 假设第一个元素是t_charge
        except (TypeError, IndexError):
            t_results = []

    # 处理嵌套列表情况
    if len(t_results) > 0 and isinstance(t_results[0], (list, np.ndarray)):
        t_results = t_results[0]

    # 确保t_results是数值列表
    try:
        formatted_t_results = [round(float(t), 3) for t in t_results]
    except TypeError as e:
        print(f"Error formatting t_results: {e}")
        print(f"Sample t_results: {t_results[:5] if len(t_results) > 0 else 'empty'}")
        formatted_t_results = []

    I_values = []
    max_v_actual_values = []
    max_N_values = []

    # RMCP参数设置
    max_N = 3
    C_stage = 12e-12
    alpha = 0.0026
    beta = 0.0026
    frequency = 203333.0

    # 创建RMCP实例
    main_charge_pump = ReconfigurableMainCharge_Pump(v_eh_bus_list, C_stage, alpha, beta, v_target, max_N ,frequency)

    # 运行RMCP模拟
    rmcp_results = main_charge_pump.run()

    # 处理RMCP结果
    if isinstance(rmcp_results, tuple):
        rmcp_energy_consumptions, n_v_actual_all_results = rmcp_results
    else:
        rmcp_energy_consumptions = rmcp_results.get('energy', [])
        n_v_actual_all_results = rmcp_results.get('n_v_actual', [])

    # 处理每个v_eh_bus的结果
    for i, v_bus in enumerate(v_eh_bus_list):
        try:
            # 获取当前v_eh_bus对应的所有结果
            if i < len(n_v_actual_all_results):
                v_actual_list = [v_actual for N, v_actual in n_v_actual_all_results[i]]
                N_list = [N for N, v_actual in n_v_actual_all_results[i]]

                if v_actual_list:
                    max_v_actual = max(v_actual_list)
                    max_N = N_list[v_actual_list.index(max_v_actual)]
                    max_v_actual_values.append(round(max_v_actual, 3))
                    max_N_values.append(max_N)
                else:
                    max_v_actual_values.append(0)
                    max_N_values.append(0)

                # 获取I值
                _, I = oscillator.find_freq_and_I_interpolated(v_bus)
                I_values.append(round(I * 15360, 3))
            else:
                max_v_actual_values.append(0)
                max_N_values.append(0)
                I_values.append(0)
        except Exception as e:
            print(f"Error processing index {i}: {str(e)}")
            max_v_actual_values.append(0)
            max_N_values.append(0)
            I_values.append(0)

    # 计算功率和效率
    power_values = [round(I * v_actual * 1e-12, 8)
                    for I, v_actual in zip(I_values, max_v_actual_values)]

    efficiency_values = []
    for v_eh_bus, max_v_actual, max_N in zip(v_eh_bus_list, max_v_actual_values, max_N_values):
        if max_N > 0 and v_eh_bus > 0:
            efficiency = max_v_actual / (v_eh_bus * (max_N + 1))
            efficiency_values.append(round(efficiency * 100, 2))
        else:
            efficiency_values.append(0)

    plt.figure(figsize=(14, 10))

    # 输入电压曲线
    plt.subplot(2, 2, 1)
    plt.plot(t, v_eh_bus_list)
    plt.title('v_eh_bus')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)

    # 输出电流曲线
    plt.subplot(2, 2, 2)
    plt.plot(t, np.array(I_values) / 1e12, 'orange')  # 将电流单位从pA转换为A
    plt.title('I_out')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid(True)

    # 输出电压曲线
    plt.subplot(2, 2, 3)
    plt.plot(t, max_v_actual_values, 'green')
    plt.title('v_out')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)

    # 输出功率曲线 - 修改为W单位
    plt.subplot(2, 2, 4)
    plt.plot(t, power_values, 'red')
    plt.title('P_out')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (W)')
    plt.grid(True)

    pdb.set_trace()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()