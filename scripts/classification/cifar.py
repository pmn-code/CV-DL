import swanlab
import torch
from model import *
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time

# 添加当前目录到Python路径，确保可以导入train模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 准备数据集
# 定义数据转换，包括调整图像大小为224x224以匹配Tudui模型的输入要求
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

# 准备CIFAR10数据集
dataset_root = "datasets/classification"
train_data = torchvision.datasets.CIFAR10(dataset_root, train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(dataset_root, train=False, transform=transform, download=True)

# len()获取数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# 由于train.py中的Trainer期望有val_loader，我们从train_data中分割一部分作为验证集
val_size = int(0.1 * len(train_data))
train_size = len(train_data) - val_size
train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=True)

# 导入train模块中的Trainer类
from train import Trainer

# 设置训练参数
model_name = 'Tudui'
optimizer_name = 'Adam'
criterion_name = 'CrossEntropyLoss'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 3
learning_rate = 1e-3

# 初始化swanlab实验
swanlab.init(
    project="cifar10-training",
    experiment_name=f"cifar10-run-{int(time.time())}",
    description="使用Tudui模型训练CIFAR10数据集",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "optimizer": optimizer_name,
        "model": model_name
    }
)

# 初始化训练器
trainer = Trainer(
    model=model_name,
    optimizer=optimizer_name,
    criterion=criterion_name,
    device=device
)

# 使用Trainer进行训练
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')
    # 训练一个epoch
    train_acc, train_loss = trainer._train(
        epoch=epoch,
        train_loader=train_dataloader,
    )
    print(f"Train Accuracy: {train_acc:.3f}%, Train Loss: {train_loss:.3f}")
    
    # 在验证集上评估模型
    val_acc, val_loss = trainer._val(
        epoch=epoch,
        val_loader=val_dataloader,
    )
    print(f"Val Accuracy: {val_acc:.3f}%, Val Loss: {val_loss:.3f}")
    
    # 保存模型
    trainer.save_model(
        model_name=model_name,
        dataset_name="CIFAR10",
        split_ratio="0.9_0.1",  # 训练集:验证集的比例
        epoch=epoch,
        train_acc=train_acc,
        val_acc=val_acc
    )

# 在测试集上进行最终评估
print("\n-------------最终测试开始------------")
trainer.model.eval()
test_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        outputs = trainer.model(imgs)
        loss = trainer.criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

# 计算测试准确率和平均损失
test_acc = 100 * test_correct / test_total
avg_test_loss = test_loss / len(test_dataloader)

print(f"Test Accuracy: {test_acc:.3f}%")
print(f"Test Loss: {avg_test_loss:.3f}")

# 在swanlab中记录测试结果
swanlab.log({
    "test/accuracy": test_acc,
    "test/loss": avg_test_loss
})

print("\n训练和测试完成！")
# swanlab会自动关闭，不需要显式调用close

