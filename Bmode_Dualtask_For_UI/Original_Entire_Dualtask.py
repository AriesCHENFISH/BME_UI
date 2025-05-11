import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# 修改后的数据集类 - 处理单张bmode.png和mask_roi
class PathologyImageDataset(Dataset):
    def __init__(self, samples, transform=None, is_train=True):
        self.samples = self._validate_samples(samples)
        self.transform = transform
        self.is_train = is_train
        self.has_segmentation = [self._has_segmentation(patient_dir) for patient_dir, _ in self.samples]
        print(f"成功加载 {len(self.samples)} 个有效样本，其中 {sum(self.has_segmentation)} 个有分割标签")

    def _has_segmentation(self, patient_dir):
        mask_dir = os.path.join(patient_dir, 'mask_roi')
        return os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0

    def _validate_samples(self, raw_samples):
        valid_samples = []
        for patient_dir, label in raw_samples:
            bmode_path = os.path.join(patient_dir, 'bmode.png')
            mask_dir = os.path.join(patient_dir, 'mask_roi')

            if not os.path.exists(bmode_path):
                print(f"警告: 文件 {bmode_path} 不存在，跳过")
                continue

            if not os.path.exists(mask_dir) or len(os.listdir(mask_dir)) == 0:
                print(f"警告: 目录 {mask_dir} 不存在或无有效分割标签，跳过")
                continue

            valid_samples.append((patient_dir, label))
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_dir, label = self.samples[idx]
        bmode_path = os.path.join(patient_dir, 'bmode.png')

        # 随机增强 (仅在训练时)
        if self.is_train:
            do_flip = random.random() < 0.5
            angle = random.uniform(-10, 10)
        else:
            do_flip = False
            angle = 0

        # 加载bmode图像
        try:
            img = Image.open(bmode_path).convert('RGB')
            if do_flip:
                img = TF.hflip(img)
            img = TF.rotate(img, angle)
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"加载 {bmode_path} 失败: {e}")
            img = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))

        # 加载分割掩码
        mask_dir = os.path.join(patient_dir, 'mask_roi')
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        if mask_files:
            mask_path = os.path.join(mask_dir, mask_files[0])  # 使用第一个掩码文件
            try:
                mask = Image.open(mask_path).convert('L')  # 灰度图
                if do_flip:
                    mask = TF.hflip(mask)
                mask = TF.rotate(mask, angle)
                mask = TF.resize(mask, (224, 224))
                mask = TF.to_tensor(mask)
                mask = (mask > 0.5).float()  # 二值化
            except Exception as e:
                print(f"加载 {mask_path} 失败: {e}")
                mask = torch.zeros(1, 224, 224)
        else:
            mask = torch.zeros(1, 224, 224)

        return img, label, mask


# 修改后的模型 - 移除RNN组件，只使用CNN
class CNNUNetModel(nn.Module):
    def __init__(self):
        super(CNNUNetModel, self).__init__()
        # 共享ResNet-50编码器（冻结）
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        # 编码器层
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # -> (B, 64, 112, 112)
        self.maxpool = resnet.maxpool  # (B, 64, 56, 56)
        self.encoder_layer1 = resnet.layer1  # -> (B, 256, 56, 56)
        self.encoder_layer2 = resnet.layer2  # -> (B, 512, 28, 28)
        self.encoder_layer3 = resnet.layer3  # -> (B, 1024, 14, 14)
        self.encoder_layer4 = resnet.layer4  # -> (B, 2048, 7, 7)

        # 分割解码器
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # -> (B, 1024, 14, 14)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # -> (B, 512, 28, 28)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # -> (B, 256, 56, 56)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # -> (B, 128, 112, 112)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # -> (B, 32, 224, 224)
        self.conv_up5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 分类分支 - 直接从全局特征预测
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, 2048, 1, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        # 输入x: (B, C, H, W)

        # 编码器
        enc0 = self.encoder_conv1(x)  # -> (B, 64, 112, 112)
        x = self.maxpool(enc0)  # -> (B, 64, 56, 56)
        enc1 = self.encoder_layer1(x)  # -> (B, 256, 56, 56)
        enc2 = self.encoder_layer2(enc1)  # -> (B, 512, 28, 28)
        enc3 = self.encoder_layer3(enc2)  # -> (B, 1024, 14, 14)
        enc4 = self.encoder_layer4(enc3)  # -> (B, 2048, 7, 7)

        # 分割分支
        dec1 = self.upconv1(enc4)  # -> (B, 1024, 14, 14)
        dec1 = torch.cat((dec1, enc3), dim=1)  # -> (B, 2048, 14, 14)
        dec1 = self.conv_up1(dec1)  # -> (B, 1024, 14, 14)

        dec2 = self.upconv2(dec1)  # -> (B, 512, 28, 28)
        dec2 = torch.cat((dec2, enc2), dim=1)  # -> (B, 1024, 28, 28)
        dec2 = self.conv_up2(dec2)  # -> (B, 256, 28, 28)

        dec3 = self.upconv3(dec2)  # -> (B, 256, 56, 56)
        dec3 = torch.cat((dec3, enc1), dim=1)  # -> (B, 512, 56, 56)
        dec3 = self.conv_up3(dec3)  # -> (B, 128, 56, 56)

        dec4 = self.upconv4(dec3)  # -> (B, 128, 112, 112)
        dec4 = torch.cat((dec4, enc0), dim=1)  # -> (B, 192, 112, 112)
        dec4 = self.conv_up4(dec4)  # -> (B, 64, 112, 112)

        seg_output = self.upconv5(dec4)  # -> (B, 32, 224, 224)
        seg_output = self.conv_up5(seg_output)  # -> (B, 1, 224, 224)

        # 分类分支
        features_cls = self.global_pool(enc4)  # -> (B, 2048, 1, 1)
        features_cls = features_cls.view(features_cls.size(0), -1)  # -> (B, 2048)
        features_cls = self.dropout(features_cls)
        cls_output = self.fc(features_cls)  # -> (B, 1)

        return cls_output, seg_output


# 数据转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载样本 - 修改为适应新的数据结构
def load_samples(root_dir):
    samples = []
    class_to_idx = {'benign': 0, 'Her+': 1, 'sanyin': 1}
    for cls_name in class_to_idx:
        class_dir = os.path.join(root_dir, cls_name)
        if not os.path.exists(class_dir):
            print(f"警告: {class_dir} 类别目录不存在")
            continue
        for patient_id in os.listdir(class_dir):
            patient_dir = os.path.join(class_dir, patient_id)
            if not os.path.isdir(patient_dir):
                continue
            bmode_path = os.path.join(patient_dir, 'bmode.png')
            if not os.path.exists(bmode_path):
                print(f"警告: {bmode_path} 不存在，跳过")
                continue
            samples.append((patient_dir, class_to_idx[cls_name]))
    return samples


def split_train_test(samples, test_ratio=0.2):
    random.shuffle(samples)
    test_size = int(len(samples) * test_ratio)
    return samples[test_size:], samples[:test_size]


# 训练函数 - 修改以适应新的模型和数据集
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


# 验证函数 - 修改以适应新的模型和数据集
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


# 测试函数 - 修改以适应新的模型和数据集
def test(model, loader, device):
    model.eval()
    all_labels_cls, all_preds_cls = [], []
    all_masks, all_seg_preds = [], []
    os.makedirs('segmentation', exist_ok=True)
    idx = 0
    with torch.no_grad():
        for inputs, labels_cls, masks in loader:
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
                # 二值化处理
                seg_pred_binary = (seg_pred > 0.5).astype(np.uint8) * 255
                Image.fromarray(seg_pred_binary[0]).save(os.path.join('segmentation', f'seg_{idx}.png'))
                idx += 1

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


# 主函数 - 修改以适应新的模型和数据集
def main():
    root_dir = '/autodl-fs/data/DeepLearning_SRTP/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    all_samples = load_samples(root_dir)
    train_samples, test_samples = split_train_test(all_samples)
    print(f"训练集数量: {len(train_samples)}, 测试集数量: {len(test_samples)}")

    test_dataset = PathologyImageDataset(test_samples, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        print(f"\nFold {fold + 1}/5")
        train_fold_samples = [train_samples[i] for i in train_idx]
        val_fold_samples = [train_samples[i] for i in val_idx]

        train_dataset = PathologyImageDataset(train_fold_samples, transform=train_transform, is_train=True)
        val_dataset = PathologyImageDataset(val_fold_samples, transform=test_transform, is_train=False)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 增大批次大小
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        model = CNNUNetModel().to(device)
        criterion_cls = nn.BCEWithLogitsLoss()
        criterion_seg = nn.BCELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005, weight_decay=1e-4)

        num_epochs, patience = 80, 15
        best_val_loss, epochs_no_improve = float('inf'), 0
        train_losses_cls, val_losses_cls = [], []
        train_losses_seg, val_losses_seg = [], []
        train_accs_cls, val_accs_cls = [], []

        for epoch in range(num_epochs):
            train_loss_cls, train_loss_seg, train_acc_cls = train(model, train_loader, criterion_cls, criterion_seg,
                                                                  optimizer, device)
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
        plt.savefig(f'fold_{fold + 1}_curves_BMODE.png')
        plt.close()

        model.load_state_dict(torch.load(f'best_model_fold{fold + 1}.pth'))
        acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou = test(model, test_loader, device)
        print(f"Fold {fold + 1} Test Results:")
        print(f"Classification - Accuracy: {acc_cls:.4f}, Recall: {recall_cls:.4f}, F1 Score: {f1_cls:.4f}")
        print(f"Confusion Matrix:\n{cm_cls}")
        print(f"Segmentation - Pixel Accuracy: {pixel_acc:.4f}, IoU: {iou:.4f}")
        fold_results.append((acc_cls, recall_cls, f1_cls, pixel_acc, iou))

    # 平均各折结果
    avg_acc_cls = np.mean([r[0] for r in fold_results])
    avg_recall_cls = np.mean([r[1] for r in fold_results])
    avg_f1_cls = np.mean([r[2] for r in fold_results])
    avg_pixel_acc = np.mean([r[3] for r in fold_results])
    avg_iou = np.mean([r[4] for r in fold_results])
    print(f"\n五折交叉验证平均测试结果:")
    print(f"分类 - 准确率: {avg_acc_cls:.4f}, 召回率: {avg_recall_cls:.4f}, F1分数: {avg_f1_cls:.4f}")
    print(f"Segmentation - Pixel Accuracy: {avg_pixel_acc:.4f}, IoU: {avg_iou:.4f}")


if __name__ == "__main__":
    main()