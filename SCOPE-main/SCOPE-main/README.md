# SCOPE
PMU+ACIM System-level Performance Modeling Tool

## Overview
This repository provides a system-level simulator for simulating PMU behaviour and evaluating compute-in-memory based CNN accelerators, focusing on energy, latency, and accuracy trade-offs under quantization and hardware non-idealities, bridging power management and computational architecture through cross-stack co-design.

## Environment

- Python 3.13
- PyTorch 2.8

## Installation
```bash
git clone https://github.com/tayyiexian/SCOPE.git
cd SCOPE
pip install -r requirements.txt
```

### Dataset download
download Cifar-10 test-batch.
```bash
bash download.sh
```

## Quick Start
Run a ResNet-16 inference simulation with 1-bit weights and 4-bit activations:

```bash
python main.py --config ./config.ini
```

## Streamlit Interactive Interface （Recommended） 

We also provides an interactive web-based interface powered by Streamlit, which allows users to explore different hardware configurations and visualize simulation results more conveniently.

### Launch Streamlit App

```bash
streamlit run ./visualize.py
```

 Once launched, the application will open in your browser, where you can:

 - Adjust key simulation parameters
 - Run simulations interactively
 - Visualize latency, energy, and accuracy results
