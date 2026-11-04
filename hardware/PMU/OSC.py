import numpy as np
import pdb

lookup_table = [
    {"I": "25p", "freq": "87.6K", "v_eh_bus": 0.289},
    {"I": "50p", "freq": "100.6K", "v_eh_bus": 0.304},
    {"I": "150p", "freq": "168.1K", "v_eh_bus": 0.335},
    {"I": "200p", "freq": "190.6K", "v_eh_bus": 0.348},
    {"I": "250p", "freq": "209.7K", "v_eh_bus": 0.351},
    {"I": "300p", "freq": "226.2K", "v_eh_bus": 0.354},
    {"I": "350p", "freq": "241.1K", "v_eh_bus": 0.359},
    {"I": "400p", "freq": "254.5K", "v_eh_bus": 0.362},
    {"I": "450p", "freq": "266.8K", "v_eh_bus": 0.367},
    {"I": "500p", "freq": "278.2K", "v_eh_bus": 0.370},
]


class Oscillator:
    def __init__(self, v_eh_bus_list, C1, C2, v_target_list, C_storage):
        self.v_eh_bus_list = v_eh_bus_list  # 输入电压列表
        self.v_target_list = v_target_list if isinstance(v_target_list, list) else [v_target_list]  # 输出目标电压值列表
        self.C1 = C1  # C1的电容值
        self.C2 = C2  # C2的电容值
        self.C_storage = C_storage  # 片外电容值
        self.energy_consumed = 0  # 总能量消耗

    # 插值查找
    def find_freq_and_I_interpolated(self, v_eh_bus):
        # 按 v_eh_bus 排序
        sorted_table = sorted(lookup_table, key=lambda x: x["v_eh_bus"])

        # 找到输入值所在的范围
        for i in range(len(sorted_table) - 1):
            if sorted_table[i]["v_eh_bus"] <= v_eh_bus <= sorted_table[i + 1]["v_eh_bus"]:
                # 线性插值
                x0, x1 = sorted_table[i]["v_eh_bus"], sorted_table[i + 1]["v_eh_bus"]
                y0_freq, y1_freq = float(sorted_table[i]["freq"][:-1]), float(sorted_table[i + 1]["freq"][:-1])
                y0_I, y1_I = float(sorted_table[i]["I"][:-1]), float(sorted_table[i + 1]["I"][:-1])

                # 计算插值
                freq = y0_freq + (y1_freq - y0_freq) * (v_eh_bus - x0) / (x1 - x0)
                I = y0_I + (y1_I - y0_I) * (v_eh_bus - x0) / (x1 - x0)
                return freq, I

        # 如果输入值超出范围，返回最接近的值
        closest_entry = min(lookup_table, key=lambda x: abs(x["v_eh_bus"] - v_eh_bus))
        return float(closest_entry["freq"][:-1]), float(closest_entry["I"][:-1])

    # 计算时间 t 的函数
    def calculate_t(self, v_eh_bus, v_target,v_li, C_storage):
        # 查找对应的 I
        freq, I = self.find_freq_and_I_interpolated(v_eh_bus)
        # 计算 t
        t = ((v_target-v_li) * C_storage) / (I * 1e-12 * 15360)  # 将 I 从 pA 转换为 A
        return t, freq

    def get_energy(self, v_eh_bus):
        # 电容 C1 充电
        v1 = v_eh_bus
        energy_c1 = 0.5 * self.C1 *v1**2 # 电容储能公式 E = 1/2 * C * V^2

        # 电容 C2 放电
        v2 = v_eh_bus
        energy_c2 = 0.5 * self.C2 *  v2**2

        # 总能量消耗
        total_energy = energy_c1
        self.energy_consumed += total_energy  # 累加到总能量消耗
        return self.energy_consumed

    def run_simulation(self, V_li):
        # 初始化结果列表
        all_results = {
            'v_target': [],
            't_charge': [],
            'frequency': [],
            'energy': [],
            'power_consumption': []
        }

        for v_target in self.v_target_list:
            self.energy_consumed = 0
            t_results = []
            freq_results = []
            energy_results = []

            for v_eh_bus in self.v_eh_bus_list:
                # 计算时间 t 和频率 freq
                t, freq = self.calculate_t(v_eh_bus, v_target,V_li, self.C_storage)
                t_results.append(round(t, 16))
                freq_results.append(round(freq, 3))

                # 计算能量消耗
                energy = self.get_energy(v_eh_bus)
                energy_results.append(round(energy, 16))

            # 计算功率消耗
            power_consumption = [round(energy * freq * 1000, 14) for energy, freq in zip(energy_results, freq_results)]

            # 将结果添加到总列表中
            all_results['v_target'].append(v_target)
            all_results['t_charge'].append([round(t, 3) for t in t_results])
            all_results['frequency'].append(freq_results)
            all_results['energy'].append(energy_results)
            all_results['power_consumption'].append(power_consumption)

        return all_results
