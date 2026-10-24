
import torch

# ==================================================
# T2：PyTorch 自动微分
# ==================================================

# 1. 定义变量 x
x = torch.tensor(2.0, requires_grad=True)

# 2. 定义函数 y = x³ + 2x²
y = x**3 + 2*x**2

# 3. 使用 autograd 计算一阶导数
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]

# 4. 使用 autograd 计算二阶导数
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]

# 5. 根据数学公式计算解析导数
first_true = 3*x.item()**2 + 4*x.item()
second_true = 6*x.item() + 4

# 6. 计算自动微分与解析导数的误差
error1 = abs(dy_dx.item() - first_true)
error2 = abs(d2y_dx2.item() - second_true)

# 7. 输出实验结果
print("========== T2 自动微分实验 ==========")

print("x =", x.item())
print("y =", y.item())

print("\n--- 一阶导数 ---")
print("Autograd 结果：", dy_dx.item())
print("解析导数结果：", first_true)
print("绝对误差：", error1)

print("\n--- 二阶导数 ---")
print("Autograd 结果：", d2y_dx2.item())
print("解析导数结果：", second_true)
print("绝对误差：", error2)

print("\ncreate_graph=True 用于建立支持高阶求导的计算图。")
print("T2 实验完成！")

