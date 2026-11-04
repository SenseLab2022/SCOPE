import streamlit as st
import numpy as np
from ACP import AuxiliaryCharge_Pump
from OSC import Oscillator
from RMCP import ReconfigurableMainCharge_Pump


HELP_TEXTS = {
    "Storage Capacitance": "Off-chip storage capacitance",
    "ACP Capacitance": "Defaults to a 3-stage Pelliconi charge pump",
    "RMCP Capacitance":"Defaults to a Dickson charge pump",
    "ACP Alpha":"Upper plate parasitic parameters",
    "ACP Beta":"Lower plate parasitic parameters",
    "RMCP Alpha":"Upper plate parasitic parameters",
    "RMCP Beta":"Lower plate parasitic parameters",
    "Max Stages":"Maximum number of stages in the main charge pump",
}


def eval_pmu():
    st.title("⚙️ PMU Simulator ⚙️ ")
    st.markdown("""
    This application simulates a Power Management Unit (PMU) consisting of:
    - Auxiliary Charge Pump (ACP)
    - Oscillator (OSC)
    - Reconfigurable Main Charge Pump (RMCP)
    """)

    # Sidebar for input parameters
    with st.sidebar:
        st.header("💾 Simulation Parameters")

        # Common parameters
        v_eh_bus = st.number_input("Input Voltage (V)", min_value=0.1, max_value=5.0, value=0.35, step=0.01)
        v_target = st.number_input("Target Voltage (V)", min_value=0.5, max_value=5.0, value=1.2, step=0.1)
        frequency = st.number_input("Clock Frequency (Hz)", min_value=1000, max_value=1000000, value=203333, step=1000)
        osc_C_storage = st.number_input("Storage Capacitance (F)", format="%.2e", min_value=1e-6, max_value=100e-6,
                                        value=5e-6, step=1e-6,help = HELP_TEXTS["Storage Capacitance"])


        # ACP parameters
        st.subheader("ACP Parameters")
        acp_C = st.number_input("ACP Capacitance (F)", format="%.2e", min_value=1e-15, max_value=1e-6, value=12e-12,
                                step=1e-12, help = HELP_TEXTS["ACP Capacitance"])
        acp_alpha = st.number_input("ACP Alpha", min_value=0.0, max_value=1.0, value=0.0026, step=0.0001,format="%.4f",help = HELP_TEXTS["ACP Alpha"])
        acp_beta = st.number_input("ACP Beta", min_value=0.0, max_value=1.0, value=0.0026, step=0.0001,format="%.4f",help = HELP_TEXTS["ACP Beta"])
        acp_cycles = st.number_input("ACP Cycles", min_value=1, max_value=3, value=1, step=1)

        # OSC parameters
        st.subheader("OSC Parameters")
        osc_C1 = st.number_input("OSC C1 (F)", format="%.2e", min_value=1e-15, max_value=1e-6, value=206e-15,
                                 step=1e-15)
        osc_C2 = st.number_input("OSC C2 (F)", format="%.2e", min_value=1e-15, max_value=1e-6, value=206e-15,
                                 step=1e-15)


        # RMCP parameters
        st.subheader("RMCP Parameters")
        rmcp_C_stage = st.number_input("RMCP Capacitance (F)", format="%.2e", min_value=1e-15, max_value=1e-6,
                                       value=12e-12, step=1e-12,help = HELP_TEXTS["RMCP Capacitance"])
        rmcp_alpha = st.number_input("RMCP Alpha", min_value=0.0, max_value=1.0, value=0.01, step=0.001,help = HELP_TEXTS["RMCP Alpha"])
        rmcp_beta = st.number_input("RMCP Beta", min_value=0.0, max_value=1.0, value=0.02, step=0.001,help = HELP_TEXTS["RMCP Beta"])
        rmcp_max_N = st.number_input("Max Stages", min_value=1, max_value=6, value=6, step=1,help = HELP_TEXTS["Max Stages"])

    # Prepare inputs for simulation
    v_eh_bus_list = [v_eh_bus]
    v_target_list = [v_target]

    # Run OSC simulation
    oscillator = Oscillator(v_eh_bus_list, osc_C1, osc_C2, v_target_list, osc_C_storage)
    osc_results = oscillator.run_simulation(0)

    # Run ACP simulation
    acp = AuxiliaryCharge_Pump(v_eh_bus_list, acp_alpha, acp_beta, acp_C, frequency)
    acp_energy_consumptions = acp.run(acp_cycles)
    acp_power_consumptions = acp.calculate_power(acp_cycles)

    # Run RMCP simulation
    all_rmcp_energy_consumptions = []
    all_rmcp_power_consumptions = []
    all_n_v_actual_results = []
    max_v_actual_list = []

    for v_target in v_target_list:
        main_charge_pump = ReconfigurableMainCharge_Pump(
            v_eh_bus_list, rmcp_C_stage, rmcp_alpha, rmcp_beta,
            v_target, rmcp_max_N, frequency
        )
        rmcp_energy_consumptions, n_v_actual_all_results = main_charge_pump.run()
        rmcp_power_consumptions = main_charge_pump.calculate_power()

        all_rmcp_energy_consumptions.append(rmcp_energy_consumptions[0])
        all_rmcp_power_consumptions.append(rmcp_power_consumptions[0])
        all_n_v_actual_results.append(n_v_actual_all_results[0])

        current_max_v_actual = max([v_actual for N, v_actual in n_v_actual_all_results[0]])
        max_v_actual_list.append(round(current_max_v_actual, 3))

    # Calculate totals
    total_energy_consumption = [
        round(acp_energy_consumptions[0] + osc_results['energy'][i][0] + all_rmcp_energy_consumptions[i], 16)
        for i in range(len(v_target_list))
    ]

    total_power_consumption = [
        round(acp_power_consumptions[0] + osc_results['power_consumption'][i][0] + all_rmcp_power_consumptions[i], 16)
        for i in range(len(v_target_list))
    ]

    # Calculate efficiency
    efficiency_list = []
    for i, v_target in enumerate(v_target_list):
        max_N = max([N for N, v_actual in all_n_v_actual_results[i]])
        pumping_efficiency = max_v_actual_list[i] / (v_eh_bus_list[0] * (max_N + 1))
        efficiency_list.append(round(pumping_efficiency * 100, 2))

    # Display results
    st.header("Simulation Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Input Voltage", f"{v_eh_bus} V")
        st.metric("Target Voltage", f"{v_target} V")
        st.metric("Output Voltage", f"{max_v_actual_list[0]} V")

    with col2:
        st.metric("Total Energy Consumption", f"{total_energy_consumption[0]:.2e} J")
        st.metric("Total Power Consumption", f"{total_power_consumption[0]:.2e} W")
        st.metric("Pumping Efficiency", f"{efficiency_list[0]}%")

    # Detailed results in expanders
    with st.expander("ACP Results"):
        st.write(f"Energy Consumption: {acp_energy_consumptions[0]:.2e} J")
        st.write(f"Power Consumption: {acp_power_consumptions[0]:.2e} W")

    with st.expander("OSC Results"):
        st.write(f"Energy Consumption: {osc_results['energy'][0][0]:.2e} J")
        st.write(f"Power Consumption: {osc_results['power_consumption'][0][0]:.2e} W")
        st.write(f"Charge Time: {osc_results['t_charge'][0][0]:.3f} s")

    with st.expander("RMCP Results"):
        st.write(f"Energy Consumption: {all_rmcp_energy_consumptions[0]:.2e} J")
        st.write(f"Power Consumption: {all_rmcp_power_consumptions[0]:.2e} W")

        st.write("Stage-wise Voltage Boost:")
        for N, v_actual in all_n_v_actual_results[0]:
            st.write(f"Stage {N}: {v_actual:.3f} V")

    # Plotting voltage progression through RMCP stages
    if all_n_v_actual_results:
        import matplotlib.pyplot as plt

        stages = [N for N, v_actual in all_n_v_actual_results[0]]
        voltages = [v_actual for N, v_actual in all_n_v_actual_results[0]]

        fig, ax = plt.subplots()
        ax.plot(stages, voltages, marker='o')
        ax.set_xlabel("Stages (N)")
        ax.set_ylabel("Output Voltage (V)")
        ax.set_title("Voltage Boost Through Charge Pump Stages")
        ax.grid(True)
        ax.set_xticks(stages)

        st.pyplot(fig)


if __name__ == "__main__":
    eval_pmu()