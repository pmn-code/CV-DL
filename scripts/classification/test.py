import torch
import torch.nn as nn
import swanlab
import os
from model import Tudui, Net
import load_data
import time
class TestModel:
    def __init__(self, model_name, model_params_path, val_dataset=None):
        """
        初始化测试模型类
        
        Args:
            model_name: 模型名称，支持'tudui'或'net'
            model_params_path: 模型参数文件路径
            val_dataset: 验证数据集，如果为None则需要通过load_data加载
        """
        self.model_name = model_name
        self.model_params_path = model_params_path
        self.val_dataset = val_dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"测试配置信息:")
        print(f"- 模型名称: {model_name}")
        print(f"- 模型参数路径: {model_params_path}")
        print(f"- 使用设备: {self.device}")
        print(f"- 验证数据集: {'已提供' if val_dataset is not None else '未提供'}")
    
    def _load_model(self):
        """
        加载模型架构并初始化
        """
        if self.model_name == 'Tudui':
            model = Tudui().to(self.device)
        elif self.model_name == 'Net':
            model = Net().to(self.device)
        else:
            raise ValueError(f"不支持的模型名称: {self.model_name}")
        
        return model
    
    def load_checkpoint(self):
        """
        加载模型参数
        """
        if not os.path.exists(self.model_params_path):
            raise FileNotFoundError(f"模型参数文件不存在: {self.model_params_path}")
        
        checkpoint = torch.load(self.model_params_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"成功加载模型参数!")
        if 'epoch' in checkpoint:
            print(f"- 训练轮次: {checkpoint['epoch']}")
        if 'train_accuracy' in checkpoint:
            print(f"- 训练准确率: {checkpoint['train_accuracy']:.2f}%")
        if 'test_accuracy' in checkpoint:
            print(f"- 测试准确率: {checkpoint['test_accuracy']:.2f}%")
        
        return checkpoint
    
    def setup_swanlab(self, experiment_name=None):
        """
        设置swanlab实验
        
        Args:
            experiment_name: 实验名称，如果为None则自动生成
        """
        
        # 初始化swanlab实验
        swanlab.init(
            project="model-testing",
            experiment_name=experiment_name,
            description=f"测试{self.model_name}模型效果",
            config={
                "model_name": self.model_name,
                "model_params_path": self.model_params_path,
                "device": str(self.device)
            }
        )
        
        print(f"SwanLab实验已初始化: {experiment_name}")
    
    def test(self, test_loader=None, log_results=True):
        """
        测试模型效果
        
        Args:
            test_loader: 验证数据加载器，如果为None则需要通过val_dataset创建
            log_results: 是否通过swanlab记录结果
            
        Returns:
            包含测试指标的字典
        """
        # 强制加载模型参数（确保使用正确的参数）
        checkpoint = self.load_checkpoint()
        
        print(f"\n调试信息：")
        print(f"- 模型设备: {next(self.model.parameters()).device}")
        print(f"- 检查点包含键: {list(checkpoint.keys())}")
        
        # 确保有验证数据加载器
        if test_loader is None:
            if self.val_dataset is not None:
                test_loader = torch.utils.data.DataLoader(
                    self.val_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=0
                )
            else:
                raise ValueError("验证数据集和验证数据加载器都未提供")
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 初始化统计变量
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        # 混淆矩阵初始化
        num_classes = len(test_loader.dataset.class_to_idx) if hasattr(test_loader.dataset, 'class_to_idx') else 10
        confusion_matrix = torch.zeros(num_classes, num_classes).to(self.device)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                # 统计损失
                test_loss += loss.item() * data.size(0)
                
                # 获取预测结果
                _, predicted = outputs.max(1)
                
                # 统计正确数量
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
                
                # 保存预测结果和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                
                # 更新混淆矩阵
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                # 打印进度
                if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(test_loader):
                    print(f"批次 [{batch_idx + 1}/{len(test_loader)}] - 当前准确率: {100. * correct / total:.2f}%")
        
        # 添加调试信息：分析预测和标签分布
        print(f"\n预测结果分析：")
        print(f"- 总样本数: {total}")
        print(f"- 正确预测数: {correct}")
        print(f"- 预测分布统计: 前10个预测值: {all_predictions[:10]}")
        print(f"- 标签分布统计: 前10个真实标签: {all_labels[:10]}")
        
        # # 计算最终指标
        # print(all_predictions)
        # print(all_labels)
        loss = test_loss / total
        accuracy = 100. * correct / total
        
        # 计算各类别的精确率、召回率和F1分数
        precision = []
        recall = []
        f1_score = []
        
        for i in range(num_classes):
            tp = confusion_matrix[i, i].item()
            fp = confusion_matrix[:, i].sum().item() - tp
            fn = confusion_matrix[i, :].sum().item() - tp
            
            if tp + fp > 0:
                p = tp / (tp + fp)
            else:
                p = 0.0
                
            if tp + fn > 0:
                r = tp / (tp + fn)
            else:
                r = 0.0
                
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
                
            precision.append(p)
            recall.append(r)
            f1_score.append(f1)
        
        # 计算宏平均
        avg_precision = sum(precision) / num_classes
        avg_recall = sum(recall) / num_classes
        avg_f1 = sum(f1_score) / num_classes
        
        # 打印测试结果
        print("\n测试结果:")
        print(f"- 平均损失: {loss:.4f}")
        print(f"- 准确率: {accuracy:.2f}%")
        print(f"- 平均精确率: {avg_precision:.4f}")
        print(f"- 平均召回率: {avg_recall:.4f}")
        print(f"- 平均F1分数: {avg_f1:.4f}")
        
        # 通过swanlab记录结果
        if log_results:
            swanlab.log({
                "test/loss": loss,
                "test/accuracy": accuracy,
                "test/precision": avg_precision,
                "test/recall": avg_recall,
                "test/f1_score": avg_f1
            })
            
            # 记录混淆矩阵
            swanlab.log({
                "test/confusion_matrix": swanlab.Image(
                    confusion_matrix.cpu().numpy(),
                    caption="混淆矩阵"
                )
            })
            
            # 记录各类别的指标
            for i in range(num_classes):
                class_name = f"class_{i}"
                swanlab.log({
                    f"test/{class_name}/precision": precision[i],
                    f"test/{class_name}/recall": recall[i],
                    f"test/{class_name}/f1_score": f1_score[i]
                })
        
        # 返回测试指标
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "class_metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "confusion_matrix": confusion_matrix.cpu().numpy()
        }
        
        return metrics

def batch_test_model_files(model_dir, model_name, dataset_root, split_ratio, batch_size=32):
    """
    批量测试指定目录下的所有模型参数文件
    
    Args:
        model_dir: 模型参数文件目录
        model_name: 模型名称
        dataset_root: 数据集根目录
        split_ratio: 数据集分割比例
        batch_size: 批量大小
    """
    # 获取目录中所有.pth文件
    model_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.pth'):
            model_files.append(os.path.join(model_dir, file))
    
    # 按文件名中的最后数字升序排序
    def get_file_number(file_path):
        # 提取文件名
        filename = os.path.basename(file_path)
        # 尝试从文件名中提取数字
        import re
        numbers = re.findall(r'\d+', filename)
        if numbers:
            # 返回最后一个数字作为排序键
            return int(numbers[-1])
        return 0
    
    # 使用自定义排序函数
    model_files.sort(key=get_file_number)
    
    print(f"找到 {len(model_files)} 个模型文件待测试：")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {os.path.basename(model_file)}")
    
    # 加载验证数据集（与train.py保持一致，用于测试比较）
    train_loader, val_loader, test_loader = load_data.get_data_loaders(
        dataset_root=dataset_root,
        split_ratio=split_ratio,
        batch_size=32
    )
    
    # 输出test_loader的信息（现在使用测试集进行测试）
    print(f"\n测试数据集信息：")
    print(f"- 批次大小: {test_loader.batch_size}")
    print(f"- 数据集大小: {len(test_loader.dataset)}")
    print(f"- 总批次数: {len(test_loader)}")
    
    # 尝试获取并显示类别信息
    if hasattr(test_loader.dataset, 'class_to_idx'):
        classes = list(test_loader.dataset.class_to_idx.keys())
        print(f"- 类别数量: {len(classes)}")
        print(f"- 类别列表: {classes}")
    else:
        print(f"- 类别信息不可用")
    
    # 尝试获取并显示第一个批次的数据形状
    try:
        first_batch = next(iter(test_loader))
        if len(first_batch) >= 2:
            data, target = first_batch
            print(f"- 数据形状: {data.shape}")
            print(f"- 标签形状: {target.shape}")
            print(f"- 标签类型: {target.dtype}")
    except Exception as e:
        print(f"- 获取批次信息时出错: {str(e)}")
    
    # 存储所有测试结果
    all_results = []
    
    # 逐个测试模型
    for i, model_file in enumerate(model_files):
        print(f"\n{'='*60}")
        print(f"开始测试模型 {i+1}/{len(model_files)}: {os.path.basename(model_file)}")
        print(f"{'='*60}")
        print(f"模型路径: {model_file}")
        print(f"使用数据集: {test_loader.dataset.__class__.__name__} ({len(test_loader.dataset)}样本)")
        
        try:
            # 初始化测试类
            tester = TestModel(
                model_name=model_name,
                model_params_path=model_file
            )
            
            # 设置swanlab，使用不同的实验名称避免冲突
            experiment_name = f"FLSMDD-test-{int(time.time())}"
            tester.setup_swanlab(experiment_name=experiment_name)
            
            # 测试模型（使用真正的测试集）
            metrics = tester.test(test_loader=test_loader)
            
            # 存储结果
            result = {
                'model_file': os.path.basename(model_file),
                'metrics': metrics
            }
            all_results.append(result)
            
            print(f"模型 {os.path.basename(model_file)} 测试完成！")
            
        except Exception as e:
            print(f"测试模型 {os.path.basename(model_file)} 时出错: {str(e)}")
            continue
    
    # 打印总结
    print(f"\n{'='*60}")
    print("所有模型测试完成！")
    print(f"成功测试: {len(all_results)}/{len(model_files)}")
    print(f"{'='*60}")
    
    # 找出最佳模型
    if all_results:
        best_accuracy = -1
        best_model = None
        
        print("\n各模型测试结果摘要：")
        print(f"{'模型文件':<30} {'准确率':<10} {'损失':<10} {'F1分数':<10}")
        print(f"{'-'*70}")
        
        for result in all_results:
            file_name = result['model_file']
            accuracy = result['metrics']['accuracy']
            loss = result['metrics']['loss']
            f1 = result['metrics']['f1_score']
            
            print(f"{file_name:<30} {accuracy:.2f}%      {loss:.4f}    {f1:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = file_name
        
        print(f"\n最佳模型: {best_model} (准确率: {best_accuracy:.2f}%)")
    
    return all_results

# 示例用法
def main():
    # 批量测试配置
    model_dir = "E:/PMN_WS/torch_test/scripts/classification/Tudui_FLSMDD"
    model_name = "Tudui"
    dataset_root = "E:/PMN_WS/torch_test/datasets/classification/FLSMDD"
    split_ratio = "0.80_0.10_0.10"
    
    # 执行批量测试
    batch_test_model_files(model_dir, model_name, dataset_root, split_ratio)

if __name__ == "__main__":
    main()