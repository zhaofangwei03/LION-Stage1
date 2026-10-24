# LION Stage 1 — 基础学习与实验

## 1. 项目简介

本项目用于整理 LION-AI4Fusion 基础学习手册 Stage 1 的代码练习及实验结果。

项目共包含五个任务，主要涉及 PyTorch 基础训练、自动微分、FFT 频域分析、有限差分法和物理信息神经网络（PINN）。

## 2. 任务介绍

### T1：MNIST 手写数字识别

- 使用 PyTorch 构建两层全连接神经网络。
- 完成数据加载、训练、验证和模型参数保存。
- 绘制训练损失和准确率曲线。

代码：`T1_MNIST/train_mnist.py`

### T2：PyTorch 自动微分

- 使用 PyTorch autograd 计算函数的一阶和二阶导数。
- 与解析导数比较，计算误差。
- 理解 `create_graph=True` 的作用。

代码：`T2_autograd/T2_autograd.py`

### T3：FFT 频域分析与信号重建

- 构造包含多个频率分量的一维信号。
- 利用 FFT 分析频谱，通过频率截断和 IFFT 重建信号。
- 比较递减幅值和等幅值两组实验的重建误差。
- 分析不同频率成分及其能量对 MSE 的影响。

代码：`T3_FFT/T3_FFT.py`

### T4：有限差分法求解一维 Poisson 方程

- 使用中心差分法离散 Poisson 方程。
- 构造线性方程组并使用 NumPy 求解。
- 比较数值解与精确解。
- 研究网格密度对数值误差的影响。

代码：`T4_Poisson/T4_Poisson.py`

### T5：PINN 求解一维 Poisson 方程

- 使用 PyTorch 从零搭建物理信息神经网络。
- 利用自动微分计算 PDE residual。
- 构造 PDE loss 和边界条件损失 BC loss。
- 绘制采样点分布、训练损失曲线和预测误差图。

代码：`T5_PINN_Poisson/T5_PINN_Poisson.py`

## 3. 运行环境

本项目开发与测试环境：

- 操作系统：Windows
- Python：3.11.16
- NumPy：2.4.6
- Matplotlib：3.11.2
- PyTorch：2.14.1+cpu
- torchvision：0.29.1+cpu
- Conda 环境名称：research

本项目使用 PyTorch CPU 版本完成实验，无需 GPU。

主要依赖见 `requirements.txt`。PyTorch 安装时应根据目标计算机的 CPU/CUDA 环境选择对应版本。

## 4. 运行方法

安装所需依赖：

```bash
python -m pip install -r requirements.txt
```

在项目根目录打开终端，进入相应任务文件夹并运行代码。

例如运行 T4：

```bash
cd T4_Poisson
python T4_Poisson.py
```

运行 T5：

```bash
cd T5_PINN_Poisson
python T5_PINN_Poisson.py
```

其他任务也可以按照相同方式运行对应的 Python 文件。

## 5. 实验结果

各任务的实验结果保存在相应的结果文件夹中。

主要包括：

- T1：训练损失曲线、准确率曲线。
- T2：一阶、二阶导数计算结果与误差。
- T3：频谱图、信号重建图、MSE 对比图。
- T4：数值解与精确解对比图、误差曲线。
- T5：采样点分布图、PINN 训练曲线、预测解对比图、绝对误差图。

## 6. 学习总结

通过 Stage 1 的五个任务，初步掌握了 PyTorch 神经网络训练、自动微分、FFT 频域分析和有限差分求解的基本方法，并完成了一个最小 PINN 实验。

通过 T4 与 T5，进一步理解了传统数值方法和物理信息神经网络求解微分方程的不同思路。