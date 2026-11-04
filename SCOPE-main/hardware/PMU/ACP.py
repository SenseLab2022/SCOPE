import numpy as np



class AuxiliaryCharge_Pump:
    def __init__(self, v_eh_bus_list, alpha, beta, C, frequency):
        self.v_eh_bus_list = v_eh_bus_list  # 输入电压列表
        self.C = C  # 电容值
        self.v_aux = 0  # 输出电压 v_aux 初始为 0
        self.energy_consumed = 0  # 总能量消耗
        self.alpha = alpha  # 电容上级板寄生参数
        self.beta = beta  # 电容下级板寄生参数
        self.frequency = frequency

    def charge(self, v_eh_bus):
        # 充电过程
        v_cap = v_eh_bus  # 电容充电到输入电压
        energy = 0.5 * self.C * (self.alpha + self.beta) * v_cap ** 2  # 电容储能公式
        self.energy_consumed += energy  # 能量消耗

    def transfer(self, v_eh_bus):
        # 转移过程
        v_cap = v_eh_bus  # 电容电压保持不变
        self.v_aux += v_cap  # 电容放电，电压转移到 v_aux
        energy = 0.5 * self.C * v_cap ** 2  # 放电过程的能量消耗
        self.energy_consumed += energy  # 能量消耗

    def run(self, cycles):
        energy_results = []  # 存储每次循环的能量消耗
        for v_eh_bus in self.v_eh_bus_list:
            self.energy_consumed = 0  # Reset energy consumed at the start of each bus cycle
            for _ in range(cycles):
                self.charge(v_eh_bus)
            energy_results.append(round(self.energy_consumed, 17))

        return energy_results

    def calculate_power(self, cycles):
        power_results = []  # 存储每次循环的功率消耗
        for v_eh_bus in self.v_eh_bus_list:
            self.energy_consumed = 0  # Reset energy consumed at the start of each bus cycle
            for _ in range(cycles):
                self.charge(v_eh_bus)
            power = round(self.frequency * self.energy_consumed / 2, 12)
            power_results.append(power)

        return power_results
