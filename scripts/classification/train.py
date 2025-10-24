import load_data
import torch
import torch.nn as nn
from model import Tudui, Net
from PIL import Image
from torchvision import transforms
import swanlab
import os
import time

class Trainer():
    def __init__(self,model,optimizer,criterion,device):
        self.model = self._model(model).to(device)
        self.optimizer = self._optimizer(optimizer)
        self.criterion = self._criterion(criterion)  
        self.device = device
        print(f'''———————————————————— device ————————————————————:\n{self.device},
———————————————————— Model ————————————————————:\n{self.model},
———————————————————— Optimizer ————————————————————: \n{self.optimizer},
———————————————————— Loss Function ————————————————————: \n{self.criterion}''')

    def _model(self, model):
        if model == 'Tudui':
            return Tudui()  # 正确传递age参数
        elif model == 'Net':
            return Net()  # 正确传递age参数
        else:
            raise ValueError(f"未知模型名称: {model}")

    def _optimizer(self, optimizer):
        if optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=0.001)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        else:
            raise ValueError(f"未知优化器名称: {optimizer}")

    def _criterion(self, criterion):
        if criterion == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"未知损失函数名称: {criterion}")

    # 训练一个epoch
    def _train(self,epoch,train_loader):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            # print(inputs.shape,labels)
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播 + 反向传播 + 优化
            outputs = self.model(inputs)
            # 计算损失
            loss = self.criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()

            # 统计准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 计算当前batch的准确率和平均损失
            batch_acc = 100 * predicted.eq(labels).sum().item() / labels.size(0)
            batch_loss = loss.item()

            # 每10个batch打印一次
            if i % 10 == 0:
                print(f'[{epoch + 1}, {i + 1}],batch_loss: {batch_loss:.3f}, batch_accuracy: {batch_acc:.3f}%')
                
                # 每个batch记录一次指标
                swanlab.log({
                    "train/batch_accuracy": batch_acc,
                    "train/batch_loss": batch_loss,
                    "train/epoch": epoch + 1,
                    "train/batch": i + 1
                })

        # 记录每个epoch的最终训练准确率和平均损失
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        swanlab.log({
            "train/epoch_accuracy": train_acc,
            "train/epoch_loss": train_loss
        })
        return train_acc, train_loss

    # 测试函数
    def _val(self,epoch,val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device) 
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # 记录测试准确率和平均损失
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        swanlab.log({
            "val/accuracy": val_acc,
            "val/loss": avg_val_loss
        })
    
        return val_acc, avg_val_loss
    
    # 保存模型
    def save_model(self, model_name, dataset_name, split_ratio, epoch, train_acc, val_acc):
        """
        保存模型，命名格式为：model_{model_name}_{dataset_name}_{split_ratio}_epoch{epoch}.pth
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            split_ratio: 数据集分割比例
            epoch: 当前训练轮次
            train_acc: 训练准确率
            val_acc: 测试准确率
        """
        import os
        
        # 创建保存模型的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建模型文件名
        filename = f"{model_name}-{dataset_name}-{split_ratio}-{epoch+1}.pth"
        filepath = os.path.join(save_dir, filename)
        
        # 保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
        }, filepath)
        
        print(f"模型已保存: {filepath}")
        return filepath
  
    


def main():
    swanlab.init(
        project="NKSID",
        experiment_name=f"cifar10-run-{int(time.time())}",
        description="使用简单CNN训练CIFAR10数据集",
        config={
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 10,
            "optimizer": "SGD",
            "momentum": 0.9
        }
    )
    # 1-加载数据集
    dataset_root = "E:/PMN_WS/torch_test/datasets/classification/NKSID"
    split_ratio = "0.80_0.10_0.10"
    train_loader, val_loader, test_loader = load_data.get_data_loaders(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        batch_size=32
    )

    # 2-初始化模型、优化器、损失函数
    model_name = 'Tudui'
    optimizer_name = 'Adam'
    criterion_name = 'CrossEntropyLoss'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 3-初始化训练器
    trainer = Trainer(
        model = model_name,
        optimizer = optimizer_name,
        criterion = criterion_name,
        device = device,
    )
    # 4-训练模型
    num_epochs = 10
    # 从dataset_root中提取数据集名称
    dataset_name = os.path.basename(dataset_root)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        # 训练一个epoch
        train_acc, train_loss = trainer._train(
            epoch=epoch,
            train_loader=train_loader,
        )
        print(f"Train Accuracy: {train_acc:.3f}%, Train Loss: {train_loss:.3f}")
        # 测试模型
        val_acc, val_loss = trainer._val(
            epoch=epoch,
            val_loader=val_loader,
        )
        print(f"Val Accuracy: {val_acc:.3f}%, Val Loss: {val_loss:.3f}")
        
        # 保存模型
        trainer.save_model(
            model_name=model_name,
            dataset_name=dataset_name,
            split_ratio=split_ratio,
            epoch=epoch,
            train_acc=train_acc,
            val_acc=val_acc
        )

if __name__ == "__main__":
    main()