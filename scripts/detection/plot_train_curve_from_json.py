import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 日志文件路径
log_file = r'D:\Workplace\CV-DL\scripts\detection\20251216_144800.log.json'

# 检查日志文件是否存在
if not os.path.exists(log_file):
    print(f"日志文件 {log_file} 不存在，请检查路径是否正确")
    exit(1)

# 初始化数据容器
epochs = []
iters = []
lrs = []
total_losses = []
loss_rpn_cls = []
loss_rpn_bbox = []
loss_cls = []
loss_bbox = []
accs = []

# 读取并解析日志文件
print(f"正在读取日志文件 {log_file}...")
with open(log_file, 'r') as f:
    for line_num, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # 跳过第一行环境信息
            if line_num == 0 and 'env_info' in data:
                continue
            # 只处理训练模式的日志
            if data.get('mode') == 'train':
                epochs.append(data.get('epoch', 0))
                iters.append(data.get('iter', 0))
                lrs.append(data.get('lr', 0))
                total_losses.append(data.get('loss', 0))
                loss_rpn_cls.append(data.get('loss_rpn_cls', 0))
                loss_rpn_bbox.append(data.get('loss_rpn_bbox', 0))
                loss_cls.append(data.get('loss_cls', 0))
                loss_bbox.append(data.get('loss_bbox', 0))
                accs.append(data.get('acc', 0))
        except json.JSONDecodeError as e:
            print(f"第 {line_num + 1} 行解析错误: {e}")
            continue

print(f"共读取 {len(epochs)} 条训练日志记录")

# 如果没有读取到数据，退出
if not epochs:
    print("没有读取到训练数据，请检查日志文件格式")
    exit(1)

# 绘制训练曲线
print("正在绘制训练曲线...")

# 创建一个大图，包含多个子图
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# 1. 总损失曲线
axes[0, 0].plot(range(len(total_losses)), total_losses, label='Total Loss')
axes[0, 0].set_title('Total Loss with Iterations')
axes[0, 0].set_xlabel('Iterations')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. 学习率曲线
axes[0, 1].plot(range(len(lrs)), lrs, label='Learning Rate', color='orange')
axes[0, 1].set_title('Learning Rate with Iterations')
axes[0, 1].set_xlabel('Iterations')
axes[0, 1].set_ylabel('LR')
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3. RPN损失曲线
axes[1, 0].plot(range(len(loss_rpn_cls)), loss_rpn_cls, label='RPN Classification Loss', color='red')
axes[1, 0].plot(range(len(loss_rpn_bbox)), loss_rpn_bbox, label='RPN Regression Loss', color='blue')
axes[1, 0].set_title('RPN Losses with Iterations')
axes[1, 0].set_xlabel('Iterations')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4. RCNN损失曲线
axes[1, 1].plot(range(len(loss_cls)), loss_cls, label='RCNN Classification Loss', color='red')
axes[1, 1].plot(range(len(loss_bbox)), loss_bbox, label='RCNN Regression Loss', color='blue')
axes[1, 1].set_title('RCNN Losses with Iterations')
axes[1, 1].set_xlabel('Iterations')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

# 5. 准确率曲线
axes[2, 0].plot(range(len(accs)), accs, label='Accuracy', color='green')
axes[2, 0].set_title('Accuracy with Iterations')
axes[2, 0].set_xlabel('Iterations')
axes[2, 0].set_ylabel('Accuracy (%)')
axes[2, 0].legend()
axes[2, 0].grid(True)

# 6. 所有损失曲线对比
axes[2, 1].plot(range(len(total_losses)), total_losses, label='Total Loss', color='black')
axes[2, 1].plot(range(len(loss_rpn_cls)), loss_rpn_cls, label='RPN Cls Loss', color='red', linestyle='--')
axes[2, 1].plot(range(len(loss_rpn_bbox)), loss_rpn_bbox, label='RPN Bbox Loss', color='red', linestyle=':')
axes[2, 1].plot(range(len(loss_cls)), loss_cls, label='RCNN Cls Loss', color='blue', linestyle='--')
axes[2, 1].plot(range(len(loss_bbox)), loss_bbox, label='RCNN Bbox Loss', color='blue', linestyle=':')
axes[2, 1].set_title('All Losses with Iterations')
axes[2, 1].set_xlabel('Iterations')
axes[2, 1].set_ylabel('Loss')
axes[2, 1].legend()
axes[2, 1].grid(True)

# 调整布局
plt.tight_layout()

# 保存图片
output_file = 'train_curves.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"训练曲线已保存到 {output_file}")

# 显示图片
plt.show()
