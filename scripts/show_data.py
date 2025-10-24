import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义与之前相同的预处理转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 训练集（假设已下载到 "datasets" 目录）
train_set = torchvision.datasets.CIFAR10(
    root="datasets\classification",
    train=True,
    download=False,
    transform=transform
)

# --------------------------
# 1. 查看数据集基本信息
# --------------------------
print(f"训练集样本数量：{len(train_set)}")  # CIFAR-10 训练集共 50000 个样本
print(f"数据集类别：{train_set.classes}")   # 10 个类别名称
print(f"类别到索引的映射：{train_set.class_to_idx}")  # 类别名称对应的数字标签


# --------------------------
# 2. 查看单个样本的内容
# --------------------------
# 取第 0 个样本（(图像张量, 标签) 的元组）
sample_image, sample_label = train_set[0]

print("\n单个样本信息：")
print(f"图像张量形状：{sample_image.shape}")  # 输出：(3, 32, 32) → (通道数, 高度, 宽度)
print(f"标签（数字）：{sample_label}")         # 输出：0-9 之间的整数（对应某个类别）
print(f"标签（类别名称）：{train_set.classes[sample_label]}")  # 转换为类别名称


# --------------------------
# 3. 可视化图像（需要反归一化）
# --------------------------
def imshow(img):
    # 反归一化：将 [-1, 1] 范围转回 [0, 1]（因为之前的 Normalize 是 (x-0.5)/0.5 = 2x-1）
    img = img / 2 + 0.5  
    # 转换为 numpy 数组，并调整形状为 (H, W, C)（matplotlib 要求的格式）
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # 从 (C, H, W) 转为 (H, W, C)
    plt.imshow(npimg)
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# # 显示第 0 个样本的图像
# print("\n显示第 0 个样本的图像：")
# imshow(sample_image)


# --------------------------
# 4. 批量展示多个样本 - 将num_samples个图像拼接在一幅图中展示
# --------------------------
# 设置要显示的样本数量
num_samples = 5

# 从数据集获取样本
images = []
labels = []
for i in range(num_samples):
    img, label = train_set[i]
    images.append(img)
    labels.append(train_set.classes[label])

# 计算合适的子图布局（自动调整行数和列数）
cols = min(5, num_samples)  # 最多5列
rows = (num_samples + cols - 1) // cols  # 计算需要的行数

# 创建一个大图来拼接所有图像
plt.figure(figsize=(3 * cols, 3 * rows))
plt.suptitle(f'CIFAR-10 样本展示 ({num_samples}个图像)', fontsize=16)

# 显示每个样本
for i in range(num_samples):
    plt.subplot(rows, cols, i + 1)
    
    # 使用imshow函数显示图像
    img = images[i] / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # 从 (C, H, W) 转为 (H, W, C)
    plt.imshow(npimg)
    
    # 显示类别标签作为标题
    plt.title(labels[i], fontsize=10)
    plt.axis('off')  # 关闭坐标轴

# 调整子图间距，使布局更紧凑
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # 为suptitle留出空间
plt.show()

# 可选：创建一个大的拼接图像，将所有样本直接拼接成一个大图像
from PIL import Image
import torch

def tensor_to_pil(img_tensor):
    # 将张量转换为PIL图像
    img = img_tensor / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = (npimg * 255).astype(np.uint8)
    return Image.fromarray(npimg)

# 创建大的拼接图像
if num_samples > 0:
    # 将第一个图像作为基准
    first_img = tensor_to_pil(images[0])
    width, height = first_img.size
    
    # 计算拼接后的尺寸
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    result_width = width * cols
    result_height = height * rows
    
    # 创建空白画布
    result_image = Image.new('RGB', (result_width, result_height), color='white')
    
    # 将每个图像粘贴到画布上
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        img_pil = tensor_to_pil(images[i])
        result_image.paste(img_pil, (col * width, row * height))
    
    # 显示拼接后的大图
    plt.figure(figsize=(10, 10 * rows / cols))
    plt.imshow(result_image)
    plt.title(f'CIFAR-10 拼接图像展示 ({num_samples}个图像)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()