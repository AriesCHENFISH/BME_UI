import sys
import os
sys.path.append(os.path.dirname(__file__))
import os
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from Model import CNNUNetModel

def test_model(model, loader, device, output_dir=None):
    """
    测试模型并计算指标

    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        output_dir: 输出目录，若不为None则保存分割结果

    Returns:
        tuple: (acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        seg_dir = os.path.join(output_dir, 'segmentation')
        os.makedirs(seg_dir, exist_ok=True)

    model.eval()
    all_labels_cls, all_preds_cls = [], []
    all_masks, all_seg_preds = [], []

    with torch.no_grad():
        for idx, (inputs, labels_cls, masks) in enumerate(loader):
            inputs = inputs.to(device)
            labels_cls = labels_cls.float().view(-1, 1)
            outputs_cls, outputs_seg = model(inputs)
            predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            all_labels_cls.extend(labels_cls.cpu().numpy().flatten())
            all_preds_cls.extend(predicted_cls.cpu().numpy().flatten())

            for i in range(inputs.size(0)):
                seg_pred = outputs_seg[i].cpu().numpy()  # (1, H, W)
                all_masks.append(masks[i].cpu().numpy())
                all_seg_preds.append(seg_pred)

                if output_dir:
                    # 二值化处理
                    seg_pred_binary = (seg_pred > 0.5).astype(np.uint8) * 255
                    Image.fromarray(seg_pred_binary[0]).save(
                        os.path.join(seg_dir, f'seg_{idx * loader.batch_size + i}.png'))

    # 分类指标
    acc_cls = accuracy_score(all_labels_cls, all_preds_cls)
    recall_cls = recall_score(all_labels_cls, all_preds_cls)
    f1_cls = f1_score(all_labels_cls, all_preds_cls)
    cm_cls = confusion_matrix(all_labels_cls, all_preds_cls)

    # 分割指标
    pixel_acc, iou = 0, 0
    if len(all_masks) > 0:
        all_masks = np.stack(all_masks)
        all_seg_preds = np.stack(all_seg_preds)
        seg_preds_binary = (all_seg_preds > 0.5).astype(float)
        pixel_acc = np.mean(seg_preds_binary == all_masks)
        intersection = np.logical_and(seg_preds_binary, all_masks).sum()
        union = np.logical_or(seg_preds_binary, all_masks).sum()
        iou = intersection / union if union != 0 else 0

    return acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou


def predict_single_image(model, image_path, device, output_dir=None, transform=None):
    """
    预测单张图像

    Args:
        model: 模型
        image_path: 图像路径
        device: 设备
        output_dir: 输出目录，若不为None则保存分割结果
        transform: 图像变换

    Returns:
        tuple: (class_pred, seg_pred)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载并预处理图像
    try:
        img = Image.open(image_path).convert('RGB')
        if transform:
            img_tensor = transform(img).unsqueeze(0).to(device)
        else:
            # 默认变换
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = default_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"加载图像 {image_path} 失败: {e}")
        return None, None

    # 预测
    model.eval()
    with torch.no_grad():
        cls_output, seg_output = model(img_tensor)
        cls_pred = (torch.sigmoid(cls_output) > 0.5).float().item()
        seg_pred = seg_output.squeeze().cpu().numpy()

    # 保存分割结果
    if output_dir:
        # 二值化处理
        seg_pred_binary = (seg_pred > 0.5).astype(np.uint8) * 255
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        Image.fromarray(seg_pred_binary).save(os.path.join(output_dir, f'{base_name}_seg.png'))

    return cls_pred, seg_pred


def load_model_for_inference(model_path, device, model_class=None):
    """
    加载模型用于推理

    Args:
        model_path: 模型路径
        device: 设备
        model_class: 模型类，若为None则使用默认的CNNUNetModel

    Returns:
        model: 加载的模型
    """
    if model_class is None:
        
        model = CNNUNetModel().to(device)
    else:
        model = model_class().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_batch(model, image_paths, device, output_dir=None, transform=None):
    """
    批量预测多张图像

    Args:
        model: 模型
        image_paths: 图像路径列表
        device: 设备
        output_dir: 输出目录，若不为None则保存分割结果
        transform: 图像变换

    Returns:
        dict: {image_path: (class_pred, seg_pred)}
    """
    results = {}
    for img_path in image_paths:
        cls_pred, seg_pred = predict_single_image(model, img_path, device, output_dir, transform)
        results[img_path] = (cls_pred, seg_pred)
    return results