import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 解析评估日志，提取AP和mAP
def parse_evaluation_log(log_file):
    current_epoch = None
    in_evaluation = False
    eval_results = defaultdict(list)
    map_values = []
    epochs_list = []
    
    # 评估日志的正则表达式
    training_epoch_pattern = re.compile(r'Epoch \[(\d+)\]\[(\d+)/(\d+)\]')  # 从训练行提取epoch
    table_start_pattern = re.compile(r'\+--------------------\+-------\+-------\+--------\+-------\+')  # 评估表起始标记
    class_ap_pattern = re.compile(r'\|\s*([\w\-]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|')  # 匹配类别AP行（支持连字符类名）
    map_pattern = re.compile(r'\|\s*mAP\s*\|\s*\|\s*\|\s*\|\s*([\d\.]+)\s*\|')  # 匹配mAP行
    
    with open(log_file, 'r') as f:
        for line in f:
            # 从训练迭代行提取当前epoch
            training_epoch_match = training_epoch_pattern.search(line)
            if training_epoch_match:
                current_epoch = int(training_epoch_match.group(1))
            
            # 检测评估表开始
            if table_start_pattern.search(line):
                in_evaluation = True
                continue
            
            # 在评估表内提取类别AP
            if in_evaluation:
                class_match = class_ap_pattern.search(line)
                if class_match:
                    class_name = class_match.group(1)
                    ap = float(class_match.group(5))
                    if current_epoch is not None:
                        eval_results[class_name].append((current_epoch, ap))
                
                # 提取mAP并结束评估表解析
                map_match = map_pattern.search(line)
                if map_match:
                    map_val = float(map_match.group(1))
                    if current_epoch is not None:
                        map_values.append((current_epoch, map_val))
                        epochs_list.append(current_epoch)
                    in_evaluation = False
    
    # 整理类别AP数据（按epoch排序）
    class_aps = defaultdict(list)
    for class_name, results in eval_results.items():
        sorted_results = sorted(results, key=lambda x: x[0])
        class_aps[class_name] = [ap for epoch, ap in sorted_results]
    
    # 整理mAP数据
    sorted_map = sorted(map_values, key=lambda x: x[0])
    map_epochs = [epoch for epoch, _ in sorted_map]
    map_values = [ap for _, ap in sorted_map]
    
    return {
        'class_aps': class_aps,
        'map_epochs': map_epochs,
        'map_values': map_values
    }

# 绘制AP/mAP曲线
def plot_ap_curves(eval_data):
    class_aps = eval_data['class_aps']
    map_epochs = eval_data['map_epochs']
    map_values = eval_data['map_values']
    
    # 绘制mAP曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(map_epochs, map_values, label='mAP', linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP with Epochs')
    plt.legend()
    plt.grid(True)
    
    # 绘制各类别AP曲线
    plt.subplot(1, 2, 2)
    for class_name, aps in class_aps.items():
        if len(aps) == len(map_epochs):  # 确保数据点数量一致
            plt.plot(map_epochs, aps, label=class_name)
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title('Per-class AP with Epochs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ap_mAP_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主函数
if __name__ == '__main__':
    log_file = 'd:\\Workplace\\CV-DL\\scripts\\detection\\20251216_144800.log'
    eval_data = parse_evaluation_log(log_file)
    
    # 打印解析结果以验证
    print(f"mAP epochs: {eval_data['map_epochs']}")
    print(f"mAP values: {eval_data['map_values']}")
    print(f"Classes found: {list(eval_data['class_aps'].keys())}")
    
    # 绘制曲线
    plot_ap_curves(eval_data)
    print("Curves saved as ap_mAP_curves.png")
