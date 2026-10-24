import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 基本设置
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("results", exist_ok=True)

# 求解方程：-u''(x) = pi^2*sin(pi*x), x in (0, 1)
# 边界条件：u(0) = u(1) = 0
# 精确解：u(x) = sin(pi*x)，仅用于训练结束后评估

def exact_solution(x):
    return np.sin(np.pi * x)


# 2. 定义 PINN 神经网络：输入 x，输出预测值 u(x)
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 生成采样点：100 个内部点 + 2 个边界点
x_interior = torch.linspace(0, 1, 102)[1:-1].reshape(-1, 1)
x_interior.requires_grad_(True)
x_boundary = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

# 绘制采样点分布
plt.figure(figsize=(9, 2.5))
plt.scatter(x_interior.detach().numpy(), np.zeros(100), s=14, label="Interior points")
plt.scatter([0, 1], [0, 0], color="red", s=70, zorder=3, label="Boundary points")
plt.yticks([])
plt.xlabel("x")
plt.title("PINN Collocation Points")
plt.legend()
plt.tight_layout()
plt.savefig("results/sampling_points.png", dpi=300)
plt.close()


# 4. 计算 PDE loss 和 BC loss
def calculate_losses():
    u = model(x_interior)

    # 自动微分：一阶导数、二阶导数
    u_x = torch.autograd.grad(
        u, x_interior,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x_interior,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # PDE residual: r(x) = -u''(x) - pi^2*sin(pi*x)
    residual = -u_xx - (torch.pi**2) * torch.sin(torch.pi * x_interior)
    loss_pde = torch.mean(residual**2)

    # Dirichlet 边界条件：u(0)=u(1)=0
    u_boundary = model(x_boundary)
    loss_bc = torch.mean(u_boundary**2)

    return loss_pde, loss_bc


# 5. 训练 PINN
EPOCHS = 3000
history_total = []
history_pde = []
history_bc = []

for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()
    loss_pde, loss_bc = calculate_losses()
    loss = loss_pde + loss_bc
    loss.backward()
    optimizer.step()

    history_total.append(loss.item())
    history_pde.append(loss_pde.item())
    history_bc.append(loss_bc.item())

    if epoch == 1 or epoch % 500 == 0:
        print(
            f"Epoch {epoch:4d}/{EPOCHS} | "
            f"Total={loss.item():.6f} | "
            f"PDE={loss_pde.item():.6f} | "
            f"BC={loss_bc.item():.6f}"
        )

# 6. 绘制训练损失曲线
plt.figure(figsize=(9, 5))
plt.semilogy(history_total, label="Total loss")
plt.semilogy(history_pde, label="PDE loss")
plt.semilogy(history_bc, label="BC loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("PINN Training Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)
plt.close()

# 7. 预测并计算误差
x_test = np.linspace(0, 1, 200).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(torch.tensor(x_test, dtype=torch.float32)).numpy()
u_exact = exact_solution(x_test)
abs_error = np.abs(u_pred - u_exact)
mse = np.mean((u_pred - u_exact)**2)
max_error = np.max(abs_error)

print(f"Test MSE: {mse:.8f}")
print(f"Test Max Absolute Error: {max_error:.8f}")

# 8. 绘制预测解与精确解对比图
plt.figure(figsize=(9, 5))
plt.plot(x_test, u_exact, "k-", linewidth=2, label="Exact solution")
plt.plot(x_test, u_pred, "r--", linewidth=2, label="PINN prediction")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("PINN vs Exact Solution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/solution_comparison.png", dpi=300)
plt.close()

# 9. 绘制绝对误差分布图
plt.figure(figsize=(9, 5))
plt.plot(x_test, abs_error, color="royalblue")
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.title("PINN Absolute Error")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/absolute_error.png", dpi=300)
plt.close()

print("T5 completed! Figures saved in results/")
