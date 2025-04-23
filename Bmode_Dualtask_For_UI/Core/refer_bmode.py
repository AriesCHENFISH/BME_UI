import sys
import os
sys.path.append(os.path.dirname(__file__))
# refer_bmode.py
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from datetime import datetime
from Predict import load_model_for_inference, predict_single_image

def refer_bmode(image_file):
    """
    传入图像文件，返回分类结果、mask图像路径
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建输出目录
    output_dir = os.path.join('static', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    model_path = r'D:\BME_ui\Bmode_Dualtask_For_UI\model_Weights\best_model_fold3.pth'
    model = load_model_for_inference(model_path, device)
    print("🧠模型加载成功！")

    # 保存上传图像到临时路径
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img_path = os.path.join(output_dir, f"input_{timestamp}.png")
    image_file.save(img_path)

    print("🎯bmode数据加载成功！")

    # 调用预测函数
    cls_pred, seg_pred = predict_single_image(model, img_path, device, output_dir)

    # 生成 mask 文件路径
    mask_path = os.path.join(output_dir, f"mask_{timestamp}.png")

    if seg_pred is not None:
        # 保存分割图像
        seg_pred_binary = (seg_pred > 0.8).astype(np.uint8) * 255
        seg_image = Image.fromarray(seg_pred_binary)
        seg_image.save(mask_path)

    print("✅bmode分割结果保存成功！")
        

    return {
        "classification": int(cls_pred),
        "mask_path": '/' + mask_path.replace('\\', '/')  # 用于前端 <img src="">
    }
