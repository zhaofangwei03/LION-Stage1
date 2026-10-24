
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 基本参数
fs = 500
T = 1
t = np.arange(0, T, 1 / fs)
N = len(t)

frequencies = [5, 12, 25, 45, 70, 100]
cutoffs = [10, 20, 35, 55, 80, 120]

# 两组实验的幅值
amplitudes_A = [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]
amplitudes_B = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

os.makedirs("results_comparison", exist_ok=True)


# 2. 生成信号
def make_signal(amplitudes):
    signal = np.zeros_like(t)

    for f, a in zip(frequencies, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)

    return signal


# 3. FFT截断与重建
def reconstruct(fft_data, freq, cutoff, mode):
    fft_cut = fft_data.copy()

    if mode == "low":
        # 只保留低于等于截止频率的分量
        fft_cut[np.abs(freq) > cutoff] = 0

    elif mode == "high":
        # 只保留高于截止频率的分量
        fft_cut[np.abs(freq) <= cutoff] = 0

    return np.fft.ifft(fft_cut).real


# 4. 单组实验
def run_experiment(name, amplitudes):
    signal = make_signal(amplitudes)

    fft_data = np.fft.fft(signal)
    freq = np.fft.fftfreq(N, d=1 / fs)

    # 单边幅值谱
    positive = freq >= 0
    spectrum = 2 * np.abs(fft_data[positive]) / N
    spectrum[0] /= 2

    # 频谱图
    plt.figure(figsize=(9, 4))
    plt.plot(freq[positive], spectrum)
    plt.xlim(0, 120)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"{name}: Frequency Spectrum")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results_comparison/{name}_spectrum.png")
    plt.close()

    # 不同截止频率下的低通重建
    reconstructions = []
    errors = []

    for cutoff in cutoffs:
        rec = reconstruct(fft_data, freq, cutoff, "low")
        mse = np.mean((signal - rec) ** 2)

        reconstructions.append(rec)
        errors.append(mse)

        print(f"{name}: cutoff={cutoff} Hz, MSE={mse:.6f}")

    # 重建波形图
    fig, axes = plt.subplots(7, 1, figsize=(10, 14))

    axes[0].plot(t, signal)
    axes[0].set_title(f"{name}: Original Signal")

    for i, cutoff in enumerate(cutoffs):
        axes[i + 1].plot(t, reconstructions[i])
        axes[i + 1].set_title(
            f"Cutoff={cutoff} Hz, MSE={errors[i]:.4f}"
        )

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-6.5, 6.5)
        ax.grid()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"results_comparison/{name}_reconstruction.png")
    plt.close()

    # 低频与高频对照：55 Hz
    low = reconstruct(fft_data, freq, 55, "low")
    high = reconstruct(fft_data, freq, 55, "high")

    low_mse = np.mean((signal - low) ** 2)
    high_mse = np.mean((signal - high) ** 2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    for ax, data, title in zip(
        axes,
        [signal, low, high],
        [
            "Original Signal",
            f"Keep <=55 Hz, MSE={low_mse:.4f}",
            f"Keep >55 Hz, MSE={high_mse:.4f}"
        ]
    ):
        ax.plot(t, data)
        ax.set_title(f"{name}: {title}")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-6.5, 6.5)
        ax.grid()

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"results_comparison/{name}_low_high.png")
    plt.close()

    return errors, low_mse, high_mse


# 5. 运行两组实验
errors_A, low_A, high_A = run_experiment(
    "A_Decreasing", amplitudes_A
)

errors_B, low_B, high_B = run_experiment(
    "B_Equal", amplitudes_B
)


# 6. 两组MSE曲线放在一起比较
plt.figure(figsize=(9, 5))

plt.plot(cutoffs, errors_A, "o-", label="A: Decreasing Amplitude")
plt.plot(cutoffs, errors_B, "s-", label="B: Equal Amplitude")

plt.xlabel("Low-pass Cutoff Frequency (Hz)")
plt.ylabel("MSE")
plt.title("Reconstruction Error Comparison")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("results_comparison/MSE_comparison.png")
plt.close()


# 7. 输出对比结果
print("\n========== Comparison ==========")
print("Cutoff | Decreasing MSE | Equal MSE")

for cutoff, ea, eb in zip(cutoffs, errors_A, errors_B):
    print(f"{cutoff:>6} | {ea:.6f}       | {eb:.6f}")

print("\n55 Hz low/high comparison:")
print(f"Group A: low MSE={low_A:.4f}, high MSE={high_A:.4f}")
print(f"Group B: low MSE={low_B:.4f}, high MSE={high_B:.4f}")

print("\nT3 comparison completed!")
