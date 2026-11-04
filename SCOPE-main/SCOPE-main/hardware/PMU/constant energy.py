from OSC import Oscillator  # 导入 OSC
from RMCP import ReconfigurableMainCharge_Pump  # 导入RMCP


# 能量供给稳定，输入电压不变条件下
def main():
    # 用户设置OSC的参数
    v_eh_bus_list = [0.34]  # 输入电压列表
    C1 = 206e-15  # C1的电容值
    C2 = 206e-15  # C2的电容值
    v_target_list = [1.4]  # 目标电压值列表
    C_storage = 50e-6  # 片外存储电容值

    # 查找表 (v_eh_bus, Iin (μA), Pin (μW))
    lookup_table = [
        (0.29, 14.235, 4.12815),
        (0.30, 14.000, 4.20000),
        (0.31, 13.700, 4.24700),
        (0.32, 13.363, 4.27616),
        (0.33, 12.630, 4.16790),
        (0.34, 12.085, 4.10890),
        (0.35, 12.880, 4.50800)
    ]

    # 初始化结果存储字典
    results_dict = {
        'v_target': v_target_list,
        'v_in': [],  #输入电压v_eh_bus
        't_charge': [],
        'I_pA': [],
        'I_scaled': [],
        'v_actual': [],
        'power': [],
        'Iin': [],  # 输入电流
        'Pin': [],  #输入功率
        'Efficiency': []  #效率
    }

    # 对每个目标电压进行循环
    for v_target in v_target_list:
        # 创建 Oscillator 的实例对象
        oscillator = Oscillator(v_eh_bus_list, C1, C2, v_target, C_storage)

        # 运行模拟
        osc_results = oscillator.run_simulation()

        # 处理充电时间结果
        t_results = osc_results['t_charge']
        if isinstance(t_results[0], list):
            t_results = t_results[0]
        formatted_t_results = [round(float(t), 3) for t in t_results]
        results_dict['t_charge'].append(formatted_t_results)

        # 用户设置RMCP的参数
        max_N = 4  # 电荷泵最高级数
        C_stage = 12e-12  # 电容值
        alpha = 0.01  # 电容上级板的寄生参数
        beta = 0.02  # 电容下极板的寄生参数
        frequency = 203333.0  # 时钟频率

        # 创建 ReconfigurableMainCharge_Pump 的实例对象
        main_charge_pump = ReconfigurableMainCharge_Pump(v_eh_bus_list, C_stage, alpha, beta, v_target, max_N,
                                                         frequency)

        # 运行RMCP模拟
        rmcp_energy_consumptions, n_v_actual_all_results = main_charge_pump.run()

        # 处理每个v_eh_bus的结果
        current_I_values = []
        current_I_scaled_values = []  # 存储乘以15360后的I值
        current_v_actual_values = []
        current_power_values = []
        current_Iin_values = []  # 输入电流
        current_Pin_values = []   # 输入功率
        current_Efficiency_values = []  # 效率

        for i, v_bus in enumerate(v_eh_bus_list):
            # 记录输入电压
            results_dict['v_in'].append(v_bus)  # 保存输入电压

            # 获取I值
            _, I = oscillator.find_freq_and_I_interpolated(v_bus)
            current_I_values.append(round(I, 3))
            # 计算乘以15360后的I值
            scaled_I = I * 15360 * 1e-12
            current_I_scaled_values.append(round(scaled_I, 7))

            # 获取v_actual值
            v_actual_list = [v_actual for N, v_actual in n_v_actual_all_results[i]]
            max_v_actual = max(v_actual_list)
            current_v_actual_values.append(round(max_v_actual, 3))

            # 计算功率
            power = I * max_v_actual * 1e-12 * 15360
            current_power_values.append(round(power, 8))

            # 从查找表获取Iin和Pin
            Iin = None
            Pin = None
            for entry in lookup_table:
                if abs(entry[0] - v_bus) < 0.001:
                    Iin = entry[1]  # μA
                    Pin = entry[2]  # μW
                    break

            if Iin is not None and Pin is not None:
                # 将Iin从μA转换为A
                Iin_A = Iin * 1e-6  # 从μA转为A
                current_Iin_values.append(round(Iin_A, 6))  # 保留6位小数

                # 将Pin从μW转换为W
                Pin_W = Pin * 1e-6  # 从μW转为W
                current_Pin_values.append(round(Pin_W, 8))  # 保留8位小数

                # 计算效率
                efficiency = power / Pin_W if power != 0 else 0
                # 将效率转换为百分比
                efficiency_percent = efficiency * 100
                current_Efficiency_values.append(round(efficiency_percent, 2))  # 保留2位小数
            else:
                current_Iin_values.append(None)
                current_Pin_values.append(None)
                current_Efficiency_values.append(None)

        # 将当前v_target的结果存入字典
        results_dict['I_pA'].append(current_I_values)
        results_dict['I_scaled'].append(current_I_scaled_values)
        results_dict['v_actual'].append(current_v_actual_values)
        results_dict['power'].append(current_power_values)
        results_dict['Iin'].append(current_Iin_values)
        results_dict['Pin'].append(current_Pin_values)
        results_dict['Efficiency'].append(current_Efficiency_values)

    # 输出结果
    print(f"v_target (V): {results_dict['v_target']}")

    print(f"t (s): {results_dict['t_charge']}")
    print(f"v_in (V): {results_dict['v_in']}")
    print(f"I_in (A): {results_dict['Iin']}")
    print(f"P_in (W): {results_dict['Pin']}")
    print(f"v_out (V): {results_dict['v_actual']}")
    print(f"I_out (A): {results_dict['I_scaled']}")
    print(f"P_out (W): {results_dict['power']}")
    print(f"Efficiency (%): {results_dict['Efficiency']}")

if __name__ == "__main__":
    main()
