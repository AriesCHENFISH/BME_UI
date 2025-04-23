# Bmode_Dualtask_For_UI/refer_ceus.py
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNNRNNUNetModel
from datetime import datetime

def refer_ceus(folder_path):
    # 模型 & 设备
    model_path = 'D:/BME_ui/ceus_dual_task/model_weights/best_model_fold2.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNRNNUNetModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    

    # 创建输出目录
    output_dir = os.path.join('static', 'output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # 加载图像序列
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
    ])

    sequence = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = test_transform(img)
        sequence.append(img_tensor)

    if len(sequence) == 0:
        raise ValueError("No valid images in folder.")

    seq_len = 60
    if len(sequence) < seq_len:
        sequence += [sequence[-1]] * (seq_len - len(sequence))
    elif len(sequence) > seq_len:
        step = len(sequence) / seq_len
        indices = [int(i * step) for i in range(seq_len)]
        sequence = [sequence[i] for i in indices]

    sequence = torch.stack(sequence).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    print("🎯ceus数据加载完成！")

    # 推理
    with torch.no_grad():
        cls_output, seg_output = model(sequence)
        cls_pred = (torch.sigmoid(cls_output) > 0.5).item()
        classification_result = 'Positive' if cls_pred == 1 else 'Negative'

        seg_output = seg_output.squeeze(0).cpu().numpy()
        seg_binary = (seg_output > 0.3).astype(np.uint8) * 255

        mask_path = os.path.join(output_dir, f'ceus_mask_{timestamp}.png')
        Image.fromarray(seg_binary[0]).save(mask_path)
        print("✅ceus分割结果保存成功！")

    print("诊断结果：", '✅良性' if int(cls_pred) == 0 else '❌恶性')


    return {
        "classification": int(cls_pred),
        "mask_path": '/' + mask_path.replace('\\', '/')
    }
