from general.logger import log_print
from ACP import AuxiliaryCharge_Pump  # 导入ACP
from OSC import Oscillator  # 导入OSC
from RMCP import ReconfigurableMainCharge_Pump  # 导入RMCP

def PMU_run(C_storage = 50e-6, v_target_list = [0.6, 0.8, 1.0,1.1], v_li = 0.81):
    v_eh_bus_list = [0.35]  # 输入电压列表

    # 用户设置OSC的参数
    C1 = 206e-15  # C1的电容值
    C2 = 206e-15  # C2的电容值


    # 创建 Oscillator 的实例对象
    oscillator = Oscillator(v_eh_bus_list, C1, C2, v_target_list, C_storage)

    # 运行模拟
    osc_results = oscillator.run_simulation(v_li)

    # 用户设置ACP的参数
    C = 12e-12  # 电容值
    cycles = 1  # 循环次数
    alpha = 0.0026  # 电容上级板的寄生参数
    beta = 0.0026  # 电容下极板的寄生参数
    frequency = 203333.0  # 时钟频率

    # 创建 AuxiliaryCharge_Pump 的实例对象
    acp = AuxiliaryCharge_Pump(v_eh_bus_list, alpha, beta, C, frequency)

    # 运行模拟
    acp_energy_consumptions = acp.run(cycles)
    acp_power_consumptions = acp.calculate_power(cycles)

    # 用户设置RMCP的参数
    max_N = 4  # 电荷泵最高级数
    C_stage = 12e-12 # 电容值
    alpha = 0.01  # 电容上级板的寄生参数
    beta = 0.02  # 电容下极板的寄生参数

    # 初始化RMCP结果存储
    all_rmcp_energy_consumptions = []
    all_rmcp_power_consumptions = []
    all_n_v_actual_results = []
    all_formatted_t_results = []
    all_freq_results = []
    max_v_actual_list = []  # 存储每个v_target对应的最大v_actual

    # 对每个目标电压运行RMCP模拟
    for v_target in v_target_list:
        # 创建 ReconfigurableMainCharge_Pump 的实例对象
        main_charge_pump = ReconfigurableMainCharge_Pump(v_eh_bus_list, C_stage, alpha, beta, v_target, max_N, frequency)

        # 运行模拟
        rmcp_energy_consumptions, n_v_actual_all_results = main_charge_pump.run()
        rmcp_power_consumptions = main_charge_pump.calculate_power()

        # 存储结果
        all_rmcp_energy_consumptions.append(rmcp_energy_consumptions[0])
        all_rmcp_power_consumptions.append(rmcp_power_consumptions[0])
        all_n_v_actual_results.append(n_v_actual_all_results[0])

        # 计算并存储当前v_target的最大v_actual
        current_max_v_actual = max([v_actual for N, v_actual in n_v_actual_all_results[0]])
        max_v_actual_list.append(round(current_max_v_actual, 3))

        # 获取OSC对应v_target的结果
        idx = v_target_list.index(v_target)
        all_formatted_t_results.append(osc_results['t_charge'][idx][0])
        all_freq_results.append(osc_results['frequency'][idx][0])

    # 输出ACP结果
    log_print(f"\n———————————————————————————————————————————————————PMU Result—————————————————————————————————————————————————————————") 
    log_print(f"ACP_energy_consumption (J): {acp_energy_consumptions}")
    log_print(f"ACP_power_consumption (W): {acp_power_consumptions}")

    # 输出OSC结果
    log_print("OSC_energy_consumption (J):", [e[0] for e in osc_results['energy']])
    log_print("OSC_power_consumption (W):", [e[0] for e in osc_results['power_consumption']])

    # 输出RMCP结果
    # log_print("RMCP Results:")
    # for i, v_target in enumerate(v_target_list):
    #     log_print(f"For v_target = {v_target} V")
    #     log_print("RMCP_energy_consumption (J):", all_rmcp_energy_consumptions[i])
    #     log_print("RMCP_power_consumption (W):", all_rmcp_power_consumptions[i])
    #     for N, v_actual in all_n_v_actual_results[i]:
    #         log_print(f"N{N}: v_actual = {v_actual:.3f} V")

    # 累加各个模块的 Energy
    total_energy_consumption = [
        round(acp_energy_consumptions[0] + osc_results['energy'][i][0] + all_rmcp_energy_consumptions[i], 16)
        for i in range(len(v_target_list))
    ]

    # 累加各个模块的 Power
    total_power_consumption = [
        round(acp_power_consumptions[0] + osc_results['power_consumption'][i][0] + all_rmcp_power_consumptions[i], 16)
        for i in range(len(v_target_list))
    ]

    # 输出最终结果
    log_print("\nTotal Results:")
    log_print("v_in (V):", v_eh_bus_list)
    log_print("v_target (V):", v_target_list)
    log_print("v_out (V):", max_v_actual_list)
    log_print("Total_energy_consumption(J):", total_energy_consumption)
    log_print("Total_power_consumption (W):", total_power_consumption)
    log_print("t_charge (s):", all_formatted_t_results)

    # 计算 Pumping_efficiency
    efficiency_list = []
    for i, v_target in enumerate(v_target_list):
        max_N = max([N for N, v_actual in all_n_v_actual_results[i]])
        pumping_efficiency = max_v_actual_list[i] / (v_eh_bus_list[0] * (max_N + 1))
        efficiency_list.append(round(pumping_efficiency * 100, 2))

    # 输出效率结果
    log_print("Pumping_efficiency (%):", efficiency_list)

    return all_formatted_t_results

if __name__ == "__main__":
    PMU_run()