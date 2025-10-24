import torch
from torch import nn
 
# 搭建神经网络（10分类网络）- 适应224x224输入
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 把网络放到序列中 - 调整为适合224x224输入的架构
        self.model = nn.Sequential(
            # 第一层卷积+池化: 224x224 -> 112x112
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 第二层卷积+池化: 112x112 -> 56x56
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 第三层卷积+池化: 56x56 -> 28x28
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 第四层卷积+池化: 28x28 -> 14x14
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 第五层卷积+池化: 14x14 -> 7x7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            # 展平: 256*7*7 = 12544
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(in_features=256*7*7, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=10)  # 10个类别
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    


if __name__ == '__main__':
    # 测试网络的验证正确性
    tudui = Tudui()
    input = torch.ones((20, 3, 224, 224))  # batch_size=20, 3通道，224x224
    output = tudui(input)
    print(f"输入形状: {input.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 [20, 10]