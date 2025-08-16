import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from data_process.my_dataset import MyDataset
from models.ResNet import resnet50
import matplotlib.pyplot as plt

# 定义数据预处理
transform = transforms.Compose([  # 定义一个图像预处理的组合操作
    transforms.ToTensor(),        # 将图像转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 根据均值和标准差进行归一化
])

# 使用 torchvision 的 ImageFolder 加载自定义格式的数据集
train_dataset = MyDataset(root='data/train', transform=transform)
val_dataset = MyDataset(root='data/val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=32)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化模型，并将模型移动到 device 上
model = resnet50().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用 Adam 优化器，学习率为 0.001

# 用于保存训练过程中的损失和准确率
train_losses = []      # 记录每个 epoch 的平均损失
train_accuracies = []  # 记录每个 epoch 的训练集准确率
val_accuracies = []   # 记录每个 epoch 的测试集准确率

# 训练模型
epochs = 10          # 训练轮数
best_accuracy = 0.0  # 初始化最佳验证集准确率
best_model_path = 'results/best_model.pth'  # 最佳模型保存路径 or 最佳权重文件路径

for epoch in range(epochs):
    running_loss = 0.0  # 初始化当前 epoch 的损失
    correct_train = 0   # 当前 epoch 的正确预测样本数
    total_train = 0     # 当前 epoch 的样本总数

    # 训练阶段
    model.train()  # 设置模型为训练模式
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        images, labels = images.to(device), labels.to(device)  # 将数据移动到 device 上
        optimizer.zero_grad()              # 梯度清零
        outputs = model(images)            # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()                    # 反向传播
        optimizer.step()                   # 更新参数
        running_loss += loss.item()        # 累加损失，得到这个 epoch 的总损失

        #  计算训练集准确率
        _, predicted = torch.max(outputs, 1)                 # 获取预测结果 eg:[0, 0.1, 0.2, 0.5, 0.1, 0.1] -> 3
        total_train += labels.size(0)                        # 累加样本数，得到当前 epoch 的样本总数
        correct_train += (predicted == labels).sum().item()  # 累加正确预测的样本数，得到当前 epoch 的正确预测样本数

    # 计算训练集上的准确率
    train_loss = running_loss / len(train_loader)  # 计算当前 epoch 的训练损失
    train_accuracy = correct_train / total_train   # 计算当前 epoch 的训练集准确率
    train_losses.append(train_loss)                # 记录每个 epoch 的平均损失（每 batch）
    train_accuracies.append(train_accuracy)        # 记录每个 epoch 的训练集准确率
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2%}")

    # 验证阶段
    model.eval()  # 设置模型为测试模式
    correct = 0   # 当前 epoch 的正确预测样本数
    totale = 0    # 当前 epoch 的样本总数
    with torch.no_grad():  # 不计算梯度，节省内存
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                        # 前向传播
            _, predicted = torch.max(outputs, 1)           # 获取预测结果
            totale += labels.size(0)                       # 累加样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数

    # 计算测试集上的准确率
    val_accuracy = correct / totale      # 计算当前 epoch 的测试集准确率
    val_accuracies.append(val_accuracy)  # 记录每个 epoch 的测试集准确率
    print(f"Epoch {epoch+1}/{epochs}, validate Accuracy: {val_accuracy:.2%}")

    # 如果测试集准确率提高，保存当前模型的权重
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_accuracy,
        }, best_model_path)
        print(f"Best model saved with accuracy: {best_accuracy:.2%}")

print(f"Best Accuracy on val set:{best_accuracy:.2%}")

# 绘制并保存损失和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制训练集损失曲线
plt.subplot(1,2,1)
plt.plot(train_losses, label='Training Loss')  # 传入数据，设置标签为 Training Loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()    # 显示图例
plt.grid(True)  # 显示网格

# 绘制训练集和测试集准确率曲线
plt.subplot(1,2,2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation Accuracy over Epochs')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('results/loss_and_accuracy_curves.png')
plt.show()

