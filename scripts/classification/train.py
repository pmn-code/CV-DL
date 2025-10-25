
import torch
import torch.nn as nn
from model import Tudui, Net

import swanlab


class Trainer():
    def __init__(self,model,optimizer,criterion,device,lr=0.001):
        self.model = self._model(model).to(device)
        self.device = device
        self.lr=lr
        self.optimizer = self._optimizer(optimizer)
        self.criterion = self._criterion(criterion)
        print(f'''———————————————————— device ————————————————————:\n{self.device},
———————————————————— Model ————————————————————:\n{self.model},
———————————————————— Optimizer ————————————————————: \n{self.optimizer},
———————————————————— Loss Function ————————————————————: \n{self.criterion},
———————————————————— Learning Rate ————————————————————: \n{self.lr}''')

    def _model(self, model):
        if model == 'Tudui':
            return Tudui()  # 正确传递age参数
        elif model == 'Net':
            return Net()  # 正确传递age参数
        else:
            raise ValueError(f"未知模型名称: {model}")

    def _optimizer(self, optimizer):
        if optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
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
            # print(f"image_shape: {inputs.shape}, labels[0]: {labels[0]}, labels_shape: {labels.shape}")
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播 + 反向传播 + 优化
            outputs = self.model(inputs)
            # 计算损失
            loss = self.criterion(outputs, labels)
            # print(f"outputs[0]: {outputs[0]}, outputs_shape: {outputs.shape}, loss: {loss.item()}")
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
            running_loss += batch_loss

            # 每10个batch打印一次
            if i % 10 == 0:
                print(f'[{epoch + 1}, {i + 1}],batch_loss: {batch_loss:.3f}, batch_accuracy: {batch_acc:.3f}%')

        # 记录每个epoch的最终训练准确率和平均损失
        train_acc = 100 * correct / total
        train_loss = running_loss
        swanlab.log({
            "train/accuracy": train_acc,
            "train/loss": train_loss
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
        val_loss = val_loss
        swanlab.log({
            "val/accuracy": val_acc,
            "val/loss": val_loss
        })
    
        return val_acc, val_loss
    
    # 保存模型
    def save_model(self, model_name, dataset_name, epoch, train_acc, val_acc):
        """
        保存模型，命名格式为：{model_name}_{dataset_name}_epoch{epoch}.pth
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称   
            epoch: 当前训练轮次
            train_acc: 训练准确率
            val_acc: 测试准确率
        """
        import os
        
        # 创建保存模型的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, f"{model_name}_{dataset_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 构建模型文件名
        filename = f"{model_name}-{dataset_name}-epoch{epoch+1}.pth"
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