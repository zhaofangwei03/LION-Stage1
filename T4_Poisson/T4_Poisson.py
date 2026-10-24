import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 创建结果文件夹
os.makedirs("results", exist_ok=True)

# 2. 定义精确解和方程右端项
def exact_solution(x):
    return np.sin(np.pi * x)

def source_function(x):
    return np.pi**2 * np.sin(np.pi * x)


# 3. 有限差分求解 Poisson 方程
def solve_poisson(n):
    # 将 [0,1] 划分成 n 个区间
    h = 1.0 / n
    x = np.linspace(0, 1, n + 1)

    # 内部网格点
    x_inner = x[1:-1]
    m = n - 1

    # 构造系数矩阵 A
    A = (
        2 * np.eye(m)
        - np.eye(m, k=1)
        - np.eye(m, k=-1)
    )

    # 构造右端向量 b
    b = h**2 * source_function(x_inner)

    # 求解内部网格点的数值解
    u_inner = np.linalg.solve(A, b)

    # 加上两端的边界条件
    u = np.zeros(n + 1)
    u[1:-1] = u_inner

    return x, u


# 4. 使用不同网格进行求解
grid_sizes = [10, 20, 40, 80]
errors = []

plt.figure(figsize=(9, 5))

# 绘制精确解
x_exact = np.linspace(0, 1, 500)
plt.plot(
    x_exact,
    exact_solution(x_exact),
    "k-",
    linewidth=2,
    label="Exact Solution"
)

for n in grid_sizes:
    x, u = solve_poisson(n)

    # 计算精确解
    u_exact = exact_solution(x)

    # 最大绝对误差
    max_error = np.max(np.abs(u - u_exact))
    errors.append(max_error)

    print(f"n={n}, h={1/n:.5f}, Max Error={max_error:.8f}")

    # 绘制数值解
    plt.plot(x, u, "o--", markersize=3, label=f"n={n}")

plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Poisson Equation: Numerical vs Exact Solution")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/solution_comparison.png", dpi=300)
plt.close()



# 5. 绘制网格数量与最大绝对误差的关系

plt.figure(figsize=(9, 5))

plt.plot(
    grid_sizes,
    errors,
    "o-",
    color="royalblue",
    linewidth=2,
    markersize=6,
    label="Max Error"
)

# 横轴：网格区间数
plt.xticks(grid_sizes)
plt.xlabel("Number of Grid Intervals (n)")

# 纵轴：最大绝对误差
plt.ylabel("Maximum Absolute Error")

plt.title("Grid Density vs Maximum Error")
plt.grid(True, linestyle="--", alpha=0.3)

# 标注各点对应的误差值
for n, error in zip(grid_sizes, errors):
    plt.annotate(
        f"{error:.6f}",
        (n, error),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.legend()
plt.tight_layout()

# 保存图片
plt.savefig("results/error_comparison.png", dpi=300)
plt.close()



# 6. 绘制最密网格的误差分布
x, u = solve_poisson(80)
point_errors = np.abs(u - exact_solution(x))

plt.figure(figsize=(8, 5))
plt.plot(x, point_errors)
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("Pointwise Error (n=80)")
plt.grid()
plt.tight_layout()
plt.savefig("results/error_distribution.png", dpi=300)
plt.close()

print("T4 completed! Figures saved in results/")
