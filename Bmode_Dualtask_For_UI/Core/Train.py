import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch.optim as optim


def train(model, loader, criterion_cls, criterion_seg, optimizer, device):
    model.train()
    running_loss_cls, running_loss_seg, correct_cls, total_cls = 0.0, 0.0, 0, 0
    for inputs, labels_cls, masks in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels_cls = labels_cls.to(device).float().view(-1, 1)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs_cls, outputs_seg = model(inputs)

        # 分类损失
        loss_cls = criterion_cls(outputs_cls, labels_cls)

        # 分割损失
        loss_seg = criterion_seg(outputs_seg, masks)

        loss = loss_cls + loss_seg
        loss.backward()
        optimizer.step()

        running_loss_cls += loss_cls.item()
        running_loss_seg += loss_seg.item()

        predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
        correct_cls += (predicted_cls == labels_cls).sum().item()
        total_cls += labels_cls.size(0)

    train_loss_cls = running_loss_cls / len(loader)
    train_loss_seg = running_loss_seg / len(loader)
    train_acc_cls = correct_cls / total_cls
    return train_loss_cls, train_loss_seg, train_acc_cls


def validate(model, loader, criterion_cls, criterion_seg, device):
    model.eval()
    running_loss_cls, running_loss_seg, correct_cls, total_cls = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels_cls, masks in loader:
            inputs = inputs.to(device)
            labels_cls = labels_cls.to(device).float().view(-1, 1)
            masks = masks.to(device)

            outputs_cls, outputs_seg = model(inputs)

            # 分类损失
            loss_cls = criterion_cls(outputs_cls, labels_cls)
            running_loss_cls += loss_cls.item()

            # 分割损失
            loss_seg = criterion_seg(outputs_seg, masks)
            running_loss_seg += loss_seg.item()

            predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            correct_cls += (predicted_cls == labels_cls).sum().item()
            total_cls += labels_cls.size(0)

    val_loss_cls = running_loss_cls / len(loader)
    val_loss_seg = running_loss_seg / len(loader)
    val_acc_cls = correct_cls / total_cls
    return val_loss_cls, val_loss_seg, val_acc_cls


def train_fold(model, train_loader, val_loader, criterion_cls, criterion_seg,
               optimizer, device, fold, output_dir, num_epochs=80, patience=15):
    best_val_loss, epochs_no_improve = float('inf'), 0
    train_losses_cls, val_losses_cls = [], []
    train_losses_seg, val_losses_seg = [], []
    train_accs_cls, val_accs_cls = [], []

    for epoch in range(num_epochs):
        train_loss_cls, train_loss_seg, train_acc_cls = train(
            model, train_loader, criterion_cls, criterion_seg, optimizer, device
        )
        val_loss_cls, val_loss_seg, val_acc_cls = validate(
            model, val_loader, criterion_cls, criterion_seg, device
        )

        train_losses_cls.append(train_loss_cls)
        val_losses_cls.append(val_loss_cls)
        train_losses_seg.append(train_loss_seg)
        val_losses_seg.append(val_loss_seg)
        train_accs_cls.append(train_acc_cls)
        val_accs_cls.append(val_acc_cls)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss Cls: {train_loss_cls:.4f}, Train Loss Seg: {train_loss_seg:.4f}, Train Acc Cls: {train_acc_cls:.4f}, "
              f"Val Loss Cls: {val_loss_cls:.4f}, Val Loss Seg: {val_loss_seg:.4f}, Val Acc Cls: {val_acc_cls:.4f}")

        total_val_loss = val_loss_cls + val_loss_seg
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            epochs_no_improve = 0
            model_path = os.path.join(output_dir, f'best_model_fold{fold + 1}.pth')
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses_cls, label='Train Loss Cls')
    plt.plot(val_losses_cls, label='Val Loss Cls')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Fold {fold + 1} Classification Loss')

    plt.subplot(2, 2, 2)
    plt.plot(train_losses_seg, label='Train Loss Seg')
    plt.plot(val_losses_seg, label='Val Loss Seg')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Fold {fold + 1} Segmentation Loss')

    plt.subplot(2, 2, 3)
    plt.plot(train_accs_cls, label='Train Acc Cls')
    plt.plot(val_accs_cls, label='Val Acc Cls')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Fold {fold + 1} Classification Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold_{fold + 1}_curves_BMODE.png'))
    plt.close()

    return model_path


def train_model_kfold(model_factory, train_samples, test_loader, criterion_cls, criterion_seg,
                      device, output_dir, n_splits=5, batch_size=16, lr=0.0005):
    """
    使用K折交叉验证训练模型

    Args:
        model_factory: 创建模型的函数
        train_samples: 训练样本
        test_loader: 测试数据加载器
        criterion_cls: 分类损失函数
        criterion_seg: 分割损失函数
        device: 设备(cuda/cpu)
        output_dir: 输出目录
        n_splits: 折数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        fold_results: 每折的结果
    """
    os.makedirs(output_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    best_models = []

    from data import get_dataloaders, PathologyImageDataset
    from utils import test_model

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        print(f"\nFold {fold + 1}/{n_splits}")
        train_fold_samples = [train_samples[i] for i in train_idx]
        val_fold_samples = [train_samples[i] for i in val_idx]

        # 创建数据加载器
        train_loader, val_loader, _ = get_dataloaders(
            train_fold_samples, val_fold_samples, [], batch_size
        )

        # 创建模型
        model = model_factory(device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=1e-4
        )

        # 训练
        best_model_path = train_fold(
            model, train_loader, val_loader, criterion_cls, criterion_seg,
            optimizer, device, fold, output_dir
        )
        best_models.append(best_model_path)

        # 测试
        model.load_state_dict(torch.load(best_model_path))
        fold_result = test_model(model, test_loader, device, os.path.join(output_dir, f'fold_{fold + 1}_results'))
        fold_results.append(fold_result)

        # 打印结果
        acc_cls, recall_cls, f1_cls, _, pixel_acc, iou = fold_result
        print(f"Fold {fold + 1} Test Results:")
        print(f"Classification - Accuracy: {acc_cls:.4f}, Recall: {recall_cls:.4f}, F1 Score: {f1_cls:.4f}")
        print(f"Segmentation - Pixel Accuracy: {pixel_acc:.4f}, IoU: {iou:.4f}")

    # 计算平均结果
    avg_acc_cls = np.mean([r[0] for r in fold_results])
    avg_recall_cls = np.mean([r[1] for r in fold_results])
    avg_f1_cls = np.mean([r[2] for r in fold_results])
    avg_pixel_acc = np.mean([r[4] for r in fold_results])
    avg_iou = np.mean([r[5] for r in fold_results])

    print(f"\n{n_splits}折交叉验证平均测试结果:")
    print(f"分类 - 准确率: {avg_acc_cls:.4f}, 召回率: {avg_recall_cls:.4f}, F1分数: {avg_f1_cls:.4f}")
    print(f"分割 - 像素准确率: {avg_pixel_acc:.4f}, IoU: {avg_iou:.4f}")

    # 保存最佳模型路径
    with open(os.path.join(output_dir, 'best_models.txt'), 'w') as f:
        for path in best_models:
            f.write(f"{path}\n")

    return fold_results, best_models
