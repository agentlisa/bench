import json
import os
import matplotlib.pyplot as plt
import re
from pathlib import Path

def draw_accuracy_chart():
    """绘制模型准确率横向柱状图"""
    results_dir = Path("results")
    
    # 存储模型名称和准确率
    model_data = []
    
    # 读取所有summary.json文件
    for file in results_dir.glob("evaluation_result_*_summary.json"):
        # 从文件名提取厂商和模型名称
        # 格式: evaluation_result_{厂商}_{模型}_summary.json
        filename = file.stem  # 去掉.json后缀
        # 移除前缀 "evaluation_result_" 和后缀 "_summary"
        name_part = filename.replace("evaluation_result_", "").replace("_summary", "")
        
        # 分割厂商和模型
        parts = name_part.split("_", 1)
        if len(parts) == 2:
            vendor, model = parts
            display_name = f"{vendor}/{model}"
        else:
            display_name = name_part
        
        # 读取文件内容
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            correct_cases = data["final_accuracy"]["correct_cases"]
            total_cases = 500
            accuracy = (correct_cases / total_cases) * 100  # 转换为百分比
            
            model_data.append({
                'name': display_name,
                'accuracy': accuracy,
                'correct': correct_cases
            })
    
    # 按准确率升序排序（在图上显示为降序，因为横向柱状图从下往上）
    model_data.sort(key=lambda x: x['accuracy'], reverse=False)
    
    # 提取数据用于绘图
    model_names = [item['name'] for item in model_data]
    accuracies = [item['accuracy'] for item in model_data]
    correct_counts = [item['correct'] for item in model_data]
    
    # 为每个模型分配不同的颜色
    colors = plt.cm.tab10(range(len(model_data)))  # 使用tab10调色板
    
    # 创建横向柱状图
    fig, ax = plt.subplots(figsize=(12, len(model_data) * 0.6))
    
    # 绘制横向柱状图，每个柱子使用不同颜色
    bars = ax.barh(model_names, accuracies, color=colors, edgecolor='black', alpha=0.8, linewidth=1.2)
    
    # 在柱子上添加数值标签
    for i, (bar, acc, correct) in enumerate(zip(bars, accuracies, correct_counts)):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}% ({correct}/500)',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 设置标题和标签
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison on Smart Contract Security Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 设置x轴范围
    ax.set_xlim(0, 110)
    
    # 添加网格线
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 调整布局
    # plt.tight_layout()
    
    # 保存图片
    output_path = results_dir / "model_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    # 显示图片
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("模型准确率排名:")
    print("="*60)
    # 降序打印排名
    for i, item in enumerate(reversed(model_data), 1):
        print(f"{i}. {item['name']:<40} {item['accuracy']:.2f}% ({item['correct']}/500)")
    print("="*60)

if __name__ == "__main__":
    draw_accuracy_chart()
