import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本设置
# ============================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5


# 创建保存模型和结果的文件夹
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

  
# ============================================================
# 2. 下载和读取 MNIST
# ============================================================

# 把图片转换成 PyTorch Tensor
transform = transforms.ToTensor()

# MNIST 原始训练集，共 60000 张
full_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("MNIST 原始训练集数量：", len(full_dataset))


# ============================================================
# 3. 划分训练集和验证集
# ============================================================

# 50000 张用于训练
# 10000 张用于验证
train_dataset, val_dataset = random_split(
    full_dataset,
    [50000, 10000]
)

print("训练集数量：", len(train_dataset))
print("验证集数量：", len(val_dataset))


# ============================================================
# 4. DataLoader
# ============================================================

train_loader = DataLoader( 
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. 建立简单神经网络
# ============================================================

class SimpleMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # 28 × 28 图片展开成 784 个数字
            nn.Flatten(),

            # 第一层：784 → 128
            nn.Linear(28 * 28, 128),

            # 激活函数
            nn.ReLU(),

            # 第二层：128 → 10
            # 10 对应数字 0~9
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


model = SimpleMLP()

print("\n模型结构：")
print(model)


# ============================================================
# 6. 定义 Loss 和优化器
# ============================================================

# 多分类任务使用交叉熵损失
loss_fn = nn.CrossEntropyLoss()

# Adam 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 7. 创建变量保存每轮结果
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# 8. 开始训练
# ============================================================

for epoch in range(EPOCHS):

    # ------------------------
    # 训练阶段
    # ------------------------

    model.train()

    total_train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

       
        # 清空上一轮梯度
        optimizer.zero_grad()

        # forward：模型预测
        outputs = model(images)

        # 计算 Loss
        loss = loss_fn(outputs, labels)

        # backward：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        total_train_loss += loss.item()

        # 找预测结果
        predicted = outputs.argmax(dim=1)

        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = total_train_loss / len(train_loader)

    train_accuracy = train_correct / train_total


    # ------------------------
    # 验证阶段
    # ------------------------

    model.eval()

    total_val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

      
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            total_val_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = total_val_loss / len(val_loader)

    val_accuracy = val_correct / val_total


    # ------------------------
    # 保存这一轮结果
    # ------------------------

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


# ============================================================
# 9. 保存模型
# ============================================================

torch.save(
    model.state_dict(),
    "model/mnist_model.pth"
)

print("\n模型已保存到：model/mnist_model.pth")


# ============================================================
# 10. 绘制 Loss 曲线
# ============================================================

epochs = range(1, EPOCHS + 1)

plt.figure()

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MNIST Loss Curve")

plt.legend()
plt.grid()

plt.savefig(
    "results/loss_curve.png",
    dpi=200
)

plt.show()


# ============================================================
# 11. 绘制 Accuracy 曲线
# ============================================================

plt.figure()

plt.plot(
    epochs,
    [x * 100 for x in train_accuracies],
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    [x * 100 for x in val_accuracies],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("MNIST Accuracy Curve")

plt.legend()
plt.grid()

plt.savefig(
    "results/accuracy_curve.png",
    dpi=200
)

plt.show()


print("\nT1 MNIST 训练完成！")