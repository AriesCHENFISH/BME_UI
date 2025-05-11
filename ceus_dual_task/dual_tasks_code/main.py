import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from data import PathologySequenceDataset, train_transform, test_transform, load_samples, split_train_test
from model import CNNRNNUNetModel
from train import train, validate
from test import test

def main():
    root_dir = '/root/autodl-fs/sq_unet_resnet34/data/dpran'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_samples = load_samples(root_dir)
    train_samples, test_samples = split_train_test(all_samples)
    print(f"训练集数量: {len(train_samples)}, 测试集数量: {len(test_samples)}")

    test_dataset = PathologySequenceDataset(test_samples, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        print(f"\nFold {fold + 1}/5")
        train_fold_samples = [train_samples[i] for i in train_idx]
        val_fold_samples = [train_samples[i] for i in val_idx]

        train_dataset = PathologySequenceDataset(train_fold_samples, transform=train_transform, is_train=True)
        val_dataset = PathologySequenceDataset(val_fold_samples, transform=test_transform, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        model = CNNRNNUNetModel().to(device)
        criterion_cls = nn.BCEWithLogitsLoss()
        criterion_seg = nn.BCELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)

        num_epochs, patience = 200, 20
        best_val_loss, epochs_no_improve = float('inf'), 0
        train_losses_cls, val_losses_cls = [], []
        train_losses_seg, val_losses_seg = [], []
        train_accs_cls, val_accs_cls = [], []

        for epoch in range(num_epochs):
            train_loss_cls, train_loss_seg, train_acc_cls = train(model, train_loader, criterion_cls, criterion_seg, optimizer, device)
            val_loss_cls, val_loss_seg, val_acc_cls = validate(model, val_loader, criterion_cls, criterion_seg, device)

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
                torch.save(model.state_dict(), f'best_model_fold{fold + 1}.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping")
                    break

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
        plt.savefig(f'fold_{fold + 1}_curves.png')
        plt.close()

        model.load_state_dict(torch.load(f'best_model_fold{fold + 1}.pth'))
        acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou = test(model, test_loader, device)
        print(f"Fold {fold + 1} Test Results:")
        print(f"Classification - Accuracy: {acc_cls:.4f}, Recall: {recall_cls:.4f}, F1 Score: {f1_cls:.4f}")
        print(f"Confusion Matrix:\n{cm_cls}")
        print(f"Segmentation - Pixel Accuracy: {pixel_acc:.4f}, IoU: {iou:.4f}")
        fold_results.append((acc_cls, recall_cls, f1_cls, pixel_acc, iou))

    avg_acc_cls = np.mean([r[0] for r in fold_results])
    avg_recall_cls = np.mean([r[1] for r in fold_results])
    avg_f1_cls = np.mean([r[2] for r in fold_results])
    avg_pixel_acc = np.mean([r[3] for r in fold_results])
    avg_iou = np.mean([r[4] for r in fold_results])
    print(f"\nAverage Test Results Across 5 Folds:")
    print(f"Classification - Accuracy: {avg_acc_cls:.4f}, Recall: {avg_recall_cls:.4f}, F1 Score: {avg_f1_cls:.4f}")
    print(f"Segmentation - Pixel Accuracy: {avg_pixel_acc:.4f}, IoU: {avg_iou:.4f}")

if __name__ == "__main__":
    main()