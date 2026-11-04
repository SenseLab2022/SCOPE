import numpy as np
from torchvision.datasets.utils import verify_str_arg

class ReconfigurableMainCharge_Pump:
    def __init__(self, v_eh_bus_list, C_stage, alpha, beta, v_target, max_N, frequency):
        self.v_eh_bus_list = v_eh_bus_list  # 输入电压列表
        self.C_stage = C_stage  # 电容值
        self.alpha = alpha  # 电容上级板寄生参数
        self.beta = beta  # 电容下级板寄生参数
        self.v_target = v_target  # 目标输出电压
        self.max_N = max_N  # 最大允许级数
        self.energy_consumed = 0  # 总能量消耗
        self.frequency = frequency # 时钟频率
        self.power_consumption = 0

    def charge_pump_process(self, v_eh_bus):
        # 初始化变量
        v_actual = v_eh_bus  # 初始实际电压值
        N = 0  # 初始化阶段计数
        CLK = 0
        Vout = 0
        QCLK2 = 0
        voltage_loss = 0

        # 存储每级的 N 和 v_actual
        n_v_actual_results = []
        while v_actual <= self.v_target and N < self.max_N:
            if CLK == 0:
                Qin = QCLK2
                vertical_encoding = Qin / (self.C_stage * (self.alpha + 1))
                Vout += vertical_encoding
                Qout = Vout * (self.alpha / (self.alpha + 1) + self.beta) * self.C_stage
                voltage_loss += Vout
                CLK = 1

            if CLK == 1:
                QCLK2 = Qout + v_eh_bus * (self.beta / (self.beta + 1) + self.alpha) * self.C_stage
                VCLK2 = QCLK2 / (self.C_stage * (self.alpha + 1 + self.beta + self.alpha / (1 + self.alpha)))
                Vout = VCLK2
                QCLK2 = Vout * (self.alpha / (self.alpha + 1) + self.beta) * self.C_stage
                voltage_loss += Vout
                N += 1
                CLK = 0


            # 计算实际输出电压
            v_actual = v_eh_bus * (N + 1) - voltage_loss * N

            # 存储每级的 N 和 v_actual
            n_v_actual_results.append((N, round(v_actual, 14)))

        return N, n_v_actual_results

    def charge(self, v_eh_bus,alpha, beta,N):
        # 充电过程
        energy = 0.5 * ((alpha+beta)*self.C_stage) * v_eh_bus ** 2   # 电容储能公式
        self.energy_consumed += energy

    def transfer(self, v_eh_bus,alpha, beta, N):
        # 转移过程
        energy = 0.5 * ((alpha+beta) * self.C_stage) * v_eh_bus ** 2   # 电容储能公式
        self.energy_consumed += energy

    def run(self):
        energy_results = []  # 存储每次循环的能量消耗
        n_v_actual_all_results = []  # 存储所有输入电压的每级 N 和 v_actual 结果

        for v_eh_bus in self.v_eh_bus_list:
            self.energy_consumed = 0  # 重置能量消耗
            N, n_v_actual_results = self.charge_pump_process(v_eh_bus)
            for _ in range(N):
                self.charge(v_eh_bus,self.beta,self.alpha,N)


            energy_results.append(round(self.energy_consumed, 20))  # 保留小数点后14位
            n_v_actual_all_results.append(n_v_actual_results)  # 存储当前输入电压的每级结果

        return energy_results, n_v_actual_all_results

    def calculate_power(self):
        power_results = []  # 存储每次循环的功率消耗
        n_v_actual_all_results = []  # 存储所有输入电压的每级 N 和 v_actual 结果
        for v_eh_bus in self.v_eh_bus_list:
            self.energy_consumed = 0  # 重置能量消耗
            N, n_v_actual_results = self.charge_pump_process(v_eh_bus)
            for _ in range(N):
                self.charge(v_eh_bus, self.beta, self.alpha, N)


            power_results.append(round(self.energy_consumed * self.frequency / 2, 16))  # 保留小数点后14位
            n_v_actual_all_results.append(n_v_actual_results)  # 存储当前输入电压的每级结果

        return power_results


