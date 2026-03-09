"""
Tactix 引擎 - 足球核心目标检测训练脚本 (Ultralytics Windows 专版)
环境要求: pip install ultralytics
"""
import os
import yaml
from ultralytics import RTDETR

def fix_roboflow_paths(yaml_path):
    """自动修复 Roboflow 导出的错误相对路径，转换为 Windows 绝对路径"""
    print(f"🔧 正在检查并修复数据集路径配置: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # 获取 yaml 文件所在的绝对大文件夹路径
    base_dir = os.path.dirname(yaml_path)
    
    # 强制覆写为绝对路径，彻底消灭 ../ 带来的找不到文件的 Bug
    data['train'] = os.path.join(base_dir, 'train', 'images')
    data['val'] = os.path.join(base_dir, 'valid', 'images')
    data['test'] = os.path.join(base_dir, 'test', 'images')
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print("✅ 路径修复完成！")

def main():
    print("🚀 正在初始化 Tactix 训练引擎...")

    # 1. 精准定位到你指定的那个长串文件夹名字里的 data.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_folder = "football-players-detection.v20-rf-detr-m.yolo26"
    yaml_path = os.path.join(current_dir, dataset_folder, "data.yaml")

    # 2. 触发修复魔法
    fix_roboflow_paths(yaml_path)

    # 3. 加载官方预编译的 RT-DETR Large 权重
    model = RTDETR("rtdetr-l.pt")

    # 4. 启动训练流水线
    results = model.train(
        data=yaml_path,           # 传入刚刚修好的绝对路径
        epochs=50,
        imgsz=576,                # 锁定 Roboflow 的 Stretch 尺寸
        batch=16,                 # 满载榨干 4090 显存
        device=0,                 # 启动第一张显卡
        workers=8,                # 8 线程高速喂图
        project="Tactix_Models", 
        name="rtdetr_large_v20"  
    )
    
    print("🎉 训练大功告成！最佳模型权重已保存在 Tactix_Models/rtdetr_large_v20/weights/best.pt")

# Windows 多线程训练的绝对死规矩，必须保留
if __name__ == '__main__':
    main()