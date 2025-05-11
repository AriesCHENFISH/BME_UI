import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from model import CNNRNNUNetModel  # 确保你的模型类在 model.py 中
import torchvision.transforms.functional as TF

# ------------------ 路径设置 ------------------
model_path = 'D:/BME_ui/ceus_dual_task/model_weights/best_model_fold2.pth'
image_folder = 'D:/BME_ui/for_test/2025041301/60frames'
save_seg_dir = 'segmentation_result'
os.makedirs(save_seg_dir, exist_ok=True)

# ------------------ 模型加载 ------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNRNNUNetModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------ 数据预处理 ------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ------------------ 加载图像序列 ------------------
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
])

sequence = []
for img_path in image_files:
    img = Image.open(img_path).convert('RGB')
    img_tensor = test_transform(img)
    sequence.append(img_tensor)

if len(sequence) == 0:
    raise ValueError("序列中没有可用的图像")

# Padding 或裁剪至固定长度（60帧）
seq_len = 60
if len(sequence) < seq_len:
    sequence += [sequence[-1]] * (seq_len - len(sequence))
elif len(sequence) > seq_len:
    step = len(sequence) / seq_len
    indices = [int(i * step) for i in range(seq_len)]
    sequence = [sequence[i] for i in indices]

sequence = torch.stack(sequence)  # (T, C, H, W)
sequence = sequence.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (B=1, C, T, H, W)

# ------------------ 推理 ------------------
with torch.no_grad():
    cls_output, seg_output = model(sequence)
    cls_pred = (torch.sigmoid(cls_output) > 0.5).item()
    classification_result = 'Positive' if cls_pred == 1 else 'Negative'
    print(f"\n✅ 分类结果: {classification_result}")

    seg_output = seg_output.squeeze(0).cpu().numpy()  # (1, H, W)

    # 保存分割图
    seg_pred_binary = (seg_output > 0.3).astype(np.uint8) * 255  # 二值化
    seg_image = Image.fromarray(seg_pred_binary[0])
    seg_image.save(os.path.join(save_seg_dir, 'seg_result.png'))
    print(f"✅ 分割图已保存: {os.path.join(save_seg_dir, 'seg_result.png')}")
