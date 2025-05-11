import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_seed(seed):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_train_test(samples, test_ratio=0.2, random_seed=42):
    """
    将样本分割为训练集和测试集

    Args:
        samples: 样本列表
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        tuple: (train_samples, test_samples)
    """
    random.seed(random_seed)
    random.shuffle(samples)
    test_size = int(len(samples) * test_ratio)
    return samples[test_size:], samples[:test_size]


def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_path):
    """
    绘制训练和验证指标的曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        output_path: 输出路径
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm, classes, output_path):
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        classes: 类别名称
        output_path: 输出路径
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_segmentation(image, mask, pred_mask, output_path):
    """
    可视化分割结果

    Args:
        image: 原始图像 (C, H, W)
        mask: 真实掩码 (H, W)
        pred_mask: 预测掩码 (H, W)
        output_path: 输出路径
    """
    # 转换为numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy()

    # 转换通道顺序并反归一化
    if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics, output_path):
    """
    保存评估指标到文本文件

    Args:
        metrics: 指标字典
        output_path: 输出路径
    """
    with open(output_path, 'w') as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")