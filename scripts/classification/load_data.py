import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class FLSMDDDataset(Dataset):
    """FLSMDD数据集加载类"""
    def __init__(self, dataset_root, split_ratio="0.80_0.10_0.10", data_type="train", transform=None):
        """
        初始化数据集
        Args:
            dataset_root: 数据集根目录路径，如 "e:/PMN_WS/torch_test/datasets/classification/FLSMDD"
            split_ratio: 数据划分比例文件夹名称，如 "0.80_0.10_0.10"
            data_type: 数据类型，可选 "train", "test", "val"
            transform: 数据预处理转换
        """
        self.dataset_root = dataset_root
        self.split_ratio = split_ratio
        self.data_type = data_type
        self.transform = transform if transform else self._get_default_transform()
        
        # 构建数据文件路径
        self.data_file = os.path.join(dataset_root, split_ratio, f"{data_type}.txt")
        self.image_root = dataset_root  # 直接使用数据集根目录，因为相对路径已包含init_image
        
        # 加载数据和标签
        self.data = self._load_data()
        
        # 加载类别映射
        self.class_to_idx = self._load_class_map()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
    
    def _get_default_transform(self):
        """获取默认的数据转换"""
        if self.data_type == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_class_map(self):
        """从class_map.txt文件加载类别映射"""
        class_map = {}
        # 构建class_map.txt文件路径
        class_map_file = os.path.join(self.dataset_root, self.split_ratio, "class_map.txt")
        
        if not os.path.exists(class_map_file):
            raise FileNotFoundError(f"类别映射文件不存在: {class_map_file}")
        
        try:
            with open(class_map_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 解析行内容，格式为: 索引 类别名
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            idx, class_name = parts
                            # 构建类别名到索引的映射
                            class_map[class_name] = int(idx)
        except Exception as e:
            raise ValueError(f"读取类别映射文件失败: {e}")
        
        if not class_map:
            raise ValueError(f"类别映射文件为空或格式错误: {class_map_file}")
        
        return class_map
    
    def _load_data(self):
        """从txt文件加载数据和标签"""
        data = []
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 解析行内容，格式为: 相对路径 标签
                    parts = line.split()
                    if len(parts) >= 2:
                        rel_path = parts[0]
                        label = parts[1]
                        data.append((rel_path, label))
        
        if not data:
            raise ValueError(f"数据文件为空或格式错误: {self.data_file}")
        
        print(f"加载了 {len(data)} 个 {self.data_type} 样本")
        return data
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        rel_path, label_str = self.data[idx]
        
        # 构建完整的图像路径
        # 由于split_data.py中生成的相对路径已经包含init_image，直接拼接数据集根目录即可
        img_path = os.path.join(self.image_root, rel_path)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"读取图像失败 {img_path}: {e}")
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        # 直接将字符串标签转换为整数索引
        label = int(label_str)
        
        return image, label

    def get_class_distribution(self):
        """获取类别分布"""
        distribution = {}
        for _, label in self.data:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def show_sample(self, idx=None):
        """显示样本图像、文件名和类别"""
        if idx is None:
            idx = np.random.randint(0, len(self))
        
        # 获取图像路径和标签
        rel_path, label_str = self.data[idx]
        label = int(label_str)
        
        # 获取图像文件名
        image_name = os.path.basename(rel_path)
        
        # 获取类别名称
        class_name = self.idx_to_class.get(label, f"未知类别({label})")
        
        # 读取并处理图像
        img_path = os.path.join(self.image_root, rel_path)
        try:
            image = Image.open(img_path).convert('RGB')
            
            # 应用转换以保持一致性
            if self.transform:
                # 创建临时转换，不包括归一化，以便正确显示
                display_transform = transforms.Compose([
                    t for t in self.transform.transforms 
                    if not isinstance(t, transforms.Normalize)
                ])
                image = display_transform(image)
            
            # 如果是张量，转换为numpy数组
            if isinstance(image, torch.Tensor):
                image = image.numpy().transpose((1, 2, 0))
        except Exception as e:
            print(f"读取图像失败 {img_path}: {e}")
            return
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"文件名: {image_name}\n类别: {class_name} (索引: {label})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def get_data_loaders(dataset_root, split_ratio="0.80_0.10_0.10", batch_size=32, num_workers=4):
    """
    获取数据加载器
    
    Args:
        dataset_root: 数据集根目录
        split_ratio: 数据划分比例
        batch_size: 批次大小
        num_workers: 工作线程数
    
    Returns:
        train_loader, val_loader, test_loader: 训练、验证、测试数据加载器
    """
    # 创建数据集实例
    train_dataset = FLSMDDDataset(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        data_type="train"
    )
    val_dataset = FLSMDDDataset(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        data_type="val"
    )
    test_dataset = FLSMDDDataset(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        data_type="test"
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 打印数据集信息
    print(f"类别映射: {train_dataset.class_to_idx}")
    print(f"类别分布: {train_dataset.get_class_distribution()}")
    
    # 获取一个样本
    image, label = train_dataset[0]
    # train_dataset.show_sample(0)
    print(f"样本形状: {image.shape}")
    print(f"样本标签: {label} ({train_dataset.idx_to_class[label]})")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 示例使用
    dataset_root = "E:/PMN_WS/torch_test/datasets/classification/FLSMDD"
    split_ratio = "0.80_0.10_0.10"
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        batch_size=8
    )
    
    print(f"训练数据加载器批次数量: {len(train_loader)}")
    print(f"验证数据加载器批次数量: {len(val_loader)}")
    print(f"测试数据加载器批次数量: {len(test_loader)}")