import load_data
import swanlab
import os
import time
from train import Trainer
import torch
import torch.nn as nn
from model import Tudui, Net


import swanlab
from PIL import Image
from torchvision import transforms

Learning_Rate = 0.001
Batch_Size = 32
Epochs = 10

Momentum = 0.9
Split_Ratio = "0.80_0.10_0.10"

Model = 'Tudui'
Optimizer = "Adam"
Criterion = 'CrossEntropyLoss'

def main():
    swanlab.init(
        project="FLSMDD",
        experiment_name=f"FLSMDD-run-{int(time.time())}",
        description="简单CNN训练FLSMDD数据集",
        config={
            "learning_rate": Learning_Rate,
            "batch_size": Batch_Size,
            "epochs": Epochs,
            "optimizer": Optimizer,
            "momentum": Momentum,
            "model": Model,
            "criterion": Criterion,
        }
    )
    # 1-加载数据集
    dataset_root = "E:/PMN_WS/torch_test/datasets/classification/FLSMDD"
    
    train_loader, val_loader, test_loader = load_data.get_data_loaders(
        dataset_root=dataset_root,
        split_ratio=Split_Ratio,
        batch_size=Batch_Size,
    )

    # 2-初始化模型、优化器、损失函数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 3-初始化训练器
    trainer = Trainer(
        model = Model,
        optimizer = Optimizer,
        criterion = Criterion,
        device = device,
        lr = Learning_Rate
    )
    # 4-训练模型
    # 从dataset_root中提取数据集名称
    dataset_name = os.path.basename(dataset_root)
    
    for epoch in range(Epochs):
        print(f'\nEpoch {epoch + 1}/{Epochs}')
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
            model_name=Model,
            dataset_name=dataset_name,
            epoch=epoch,
            train_acc=train_acc,
            val_acc=val_acc
        )

if __name__ == "__main__":
    main()