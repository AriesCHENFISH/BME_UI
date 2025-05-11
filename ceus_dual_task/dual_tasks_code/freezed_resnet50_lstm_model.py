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

# Custom Dataset for Classification and Segmentation
class PathologySequenceDataset(Dataset):
    def __init__(self, samples, transform=None, seq_len=60, is_train=True):
        self.samples = self._validate_samples(samples)
        self.transform = transform
        self.seq_len = seq_len
        self.is_train = is_train
        self.has_segmentation = [self._has_segmentation(patient_dir) for patient_dir, _ in self.samples]
        print(f"成功加载 {len(self.samples)} 个有效样本，其中 {sum(self.has_segmentation)} 个有分割标签")

    def _has_segmentation(self, patient_dir):
        mask_dir = os.path.join(patient_dir, 'mask_roi')
        return os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0

    def _validate_samples(self, raw_samples):
        valid_samples = []
        for patient_dir, label in raw_samples:
            frames_dir = os.path.join(patient_dir, '60frames')
            mask_dir = os.path.join(patient_dir, 'mask_roi')
            if not os.path.exists(frames_dir):
                print(f"警告: 目录 {frames_dir} 不存在，跳过")
                continue
            if not os.path.exists(mask_dir) or len(os.listdir(mask_dir)) == 0:
                print(f"警告: 目录 {mask_dir} 不存在或无有效分割标签，跳过")
                continue
            image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
            if not image_files:
                print(f"警告: 目录 {frames_dir} 无有效图像，跳过")
                continue
            valid_samples.append((patient_dir, label))
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient_dir, label = self.samples[idx]
        frames_dir = os.path.join(patient_dir, '60frames')
        image_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        image_files.sort()

        if len(image_files) < self.seq_len:
            last_file = image_files[-1]
            image_files += [last_file] * (self.seq_len - len(image_files))
        elif len(image_files) > self.seq_len:
            step = len(image_files) / self.seq_len
            image_files = [image_files[min(len(image_files) - 1, int(i * step))] for i in range(self.seq_len)]

        if self.is_train:
            do_flip = random.random() < 0.5
            angle = random.uniform(-10, 10)
        else:
            do_flip = False
            angle = 0

        sequence = []
        for img_file in image_files[:self.seq_len]:
            img_path = os.path.join(frames_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                if do_flip:
                    img = TF.hflip(img)
                img = TF.rotate(img, angle)
                if self.transform:
                    img = self.transform(img)
            except Exception as e:
                print(f"加载 {img_path} 失败: {e}")
                img = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            sequence.append(img)

        sequence = torch.stack(sequence).permute(1, 0, 2, 3)  # (C,T,H,W)

        # Load segmentation mask (assuming it corresponds to the first frame)
        mask_dir = os.path.join(patient_dir, 'mask_roi')
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        if mask_files:
            mask_path = os.path.join(mask_dir, mask_files[0])  # Use the first mask file
            try:
                mask = Image.open(mask_path).convert('L')  # Grayscale
                if do_flip:
                    mask = TF.hflip(mask)
                mask = TF.rotate(mask, angle)
                mask = TF.resize(mask, (224, 224))
                mask = TF.to_tensor(mask)
                mask = (mask > 0.5).float()  # Binarize
            except Exception as e:
                print(f"加载 {mask_path} 失败: {e}")
                mask = torch.zeros(1, 224, 224)
        else:
            mask = torch.zeros(1, 224, 224)

        return sequence, label, mask

# Dual-Task Model with Shared ResNet-50 Encoder and U-Net like Decoder
class CNNRNNUNetModel(nn.Module):
    def __init__(self):
        super(CNNRNNUNetModel, self).__init__()
        # Shared ResNet-50 encoder (frozen)
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        # Encoder layers
        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # -> (B*T, 64, 112, 112)
        self.maxpool = resnet.maxpool  # 添加maxpool，输出 (B*T, 64, 56, 56)
        self.encoder_layer1 = resnet.layer1  # -> (B*T, 256, 56, 56)
        self.encoder_layer2 = resnet.layer2  # -> (B*T, 512, 28, 28)
        self.encoder_layer3 = resnet.layer3  # -> (B*T, 1024, 14, 14)
        self.encoder_layer4 = resnet.layer4  # -> (B*T, 2048, 7, 7)

        # Segmentation decoder
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

        # Classification branch
        self.rnn = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 2, 1)  # Bidirectional LSTM output

    def forward(self, x):
        B, C, T, H, W = x.size()  # 假设输入是 (B, 3, 60, 224, 224)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)  # (B*T, 3, 224, 224)

        # Encoder
        enc0 = self.encoder_conv1(x)        # -> (B*T, 64, 112, 112)
        x = self.maxpool(enc0)                # -> (B*T, 64, 56, 56)
        enc1 = self.encoder_layer1(x)         # -> (B*T, 256, 56, 56)
        enc2 = self.encoder_layer2(enc1)      # -> (B*T, 512, 28, 28)
        enc3 = self.encoder_layer3(enc2)      # -> (B*T, 1024, 14, 14)
        enc4 = self.encoder_layer4(enc3)      # -> (B*T, 2048, 7, 7)

        # 分割分支：聚合时间维度，保留空间维度
        features_seg = enc4.view(B, T, 2048, 7, 7).mean(dim=1)  # (B, 2048, 7, 7)

        # Decoder with skip connections
        dec1 = self.upconv1(features_seg)  # -> (B, 1024, 14, 14)
        skip3 = enc3.view(B, T, 1024, 14, 14).mean(dim=1)       # (B, 1024, 14, 14)
        dec1 = torch.cat((dec1, skip3), dim=1)  # -> (B, 2048, 14, 14)
        dec1 = self.conv_up1(dec1)             # -> (B, 1024, 14, 14)

        dec2 = self.upconv2(dec1)              # -> (B, 512, 28, 28)
        skip2 = enc2.view(B, T, 512, 28, 28).mean(dim=1)        # (B, 512, 28, 28)
        dec2 = torch.cat((dec2, skip2), dim=1)  # -> (B, 1024, 28, 28)
        dec2 = self.conv_up2(dec2)             # -> (B, 256, 28, 28)

        dec3 = self.upconv3(dec2)              # -> (B, 256, 56, 56)
        skip1 = enc1.view(B, T, 256, 56, 56).mean(dim=1)        # (B, 256, 56, 56)
        dec3 = torch.cat((dec3, skip1), dim=1)  # -> (B, 512, 56, 56)
        dec3 = self.conv_up3(dec3)             # -> (B, 128, 56, 56)

        dec4 = self.upconv4(dec3)              # -> (B, 128, 112, 112)
        skip0 = enc0.view(B, T, 64, 112, 112).mean(dim=1)       # (B, 64, 112, 112)
        dec4 = torch.cat((dec4, skip0), dim=1)  # -> (B, 128+64, 112, 112) = (B, 192, 112, 112)
        dec4 = self.conv_up4(dec4)             # -> (B, 64, 112, 112)

        seg_output = self.upconv5(dec4)        # -> (B, 32, 224, 224)
        seg_output = self.conv_up5(seg_output) # -> (B, 1, 224, 224)

        # 分类分支
        features_cls = enc4.view(B, T, 2048, 7, 7).mean(dim=3).mean(dim=3)  # (B, T, 2048)
        _, (h_n, _) = self.rnn(features_cls)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)  # 拼接前向和后向状态
        h_n = self.dropout(h_n)
        cls_output = self.fc(h_n)  # (B, 1)

        return cls_output, seg_output


# Data Transforms
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

def compute_mean_std(image_list):
    means, stds = [], []
    for img_path in tqdm(image_list, desc="计算统计量"):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_np = np.array(img) / 255.0
                means.append(img_np.mean(axis=(0, 1)))
                stds.append(img_np.std(axis=(0, 1)))
        except Exception as e:
            print(f"跳过损坏文件: {img_path} ({str(e)})")
            continue
    if len(means) == 0:
        return np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    return np.mean(means, axis=0), np.mean(stds, axis=0)

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
            frames_dir = os.path.join(patient_dir, '60frames')
            if not os.path.exists(frames_dir):
                print(f"警告: {frames_dir} 不存在，跳过")
                continue
            has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')) for f in os.listdir(frames_dir))
            if not has_images:
                print(f"警告: {frames_dir} 无图像文件，跳过")
                continue
            samples.append((patient_dir, class_to_idx[cls_name]))
    return samples

def split_train_test(samples, test_ratio=0.2):
    random.shuffle(samples)
    test_size = int(len(samples) * test_ratio)
    return samples[test_size:], samples[:test_size]

# Training Function
def train(model, loader, criterion_cls, criterion_seg, optimizer, device):
    model.train()
    running_loss_cls, running_loss_seg, correct_cls, total_cls = 0.0, 0.0, 0, 0
    for inputs, labels_cls, masks in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels_cls = labels_cls.to(device).float().view(-1, 1)
        masks = masks.to(device) if isinstance(masks, torch.Tensor) else [m.to(device) for m in masks]

        optimizer.zero_grad()
        outputs_cls, outputs_seg = model(inputs)

        # Classification loss
        loss_cls = criterion_cls(outputs_cls, labels_cls)

        # Segmentation loss
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

# Validation Function
def validate(model, loader, criterion_cls, criterion_seg, device):
    model.eval()
    running_loss_cls, running_loss_seg, correct_cls, total_cls = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels_cls, masks in loader:
            inputs = inputs.to(device)
            labels_cls = labels_cls.to(device).float().view(-1, 1)
            masks = masks.to(device) if isinstance(masks, torch.Tensor) else [m.to(device) for m in masks]

            outputs_cls, outputs_seg = model(inputs)  # outputs_seg: (B, 1, H, W)

            # Classification loss
            loss_cls = criterion_cls(outputs_cls, labels_cls)
            running_loss_cls += loss_cls.item()

            # Segmentation loss
            loss_seg = criterion_seg(outputs_seg, masks)
            running_loss_seg += loss_seg.item()

            predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            correct_cls += (predicted_cls == labels_cls).sum().item()
            total_cls += labels_cls.size(0)

    val_loss_cls = running_loss_cls / len(loader)
    val_loss_seg = running_loss_seg / len(loader)
    val_acc_cls = correct_cls / total_cls
    return val_loss_cls, val_loss_seg, val_acc_cls

# Testing Function
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
            outputs_cls, outputs_seg = model(inputs)  # outputs_seg: (B, 1, H, W)
            predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            all_labels_cls.extend(labels_cls.cpu().numpy().flatten())
            all_preds_cls.extend(predicted_cls.cpu().numpy().flatten())

            for i in range(inputs.size(0)):
                if masks[i] is not None:
                    seg_pred = outputs_seg[i].cpu().numpy()  # (1, H, W)
                    all_masks.append(masks[i].cpu().numpy())
                    all_seg_preds.append(seg_pred)
                    # 二值化处理
                    seg_pred_binary = (seg_pred > 0.5).astype(np.uint8) * 255
                    Image.fromarray(seg_pred_binary[0]).save(os.path.join('segmentation', f'seg_{idx}.png'))
                idx += 1

    # Classification metrics
    acc_cls = accuracy_score(all_labels_cls, all_preds_cls)
    recall_cls = recall_score(all_labels_cls, all_preds_cls)
    f1_cls = f1_score(all_labels_cls, all_preds_cls)
    cm_cls = confusion_matrix(all_labels_cls, all_preds_cls)

    # Segmentation metrics
    pixel_acc, iou = 0, 0
    if len(all_masks) > 0:
        all_masks = np.concatenate(all_masks, axis=0)
        all_seg_preds = np.concatenate(all_seg_preds, axis=0)
        seg_preds_binary = (all_seg_preds > 0.5).astype(float)
        pixel_acc = np.mean(seg_preds_binary == all_masks)
        intersection = np.logical_and(seg_preds_binary, all_masks).sum()
        union = np.logical_or(seg_preds_binary, all_masks).sum()
        iou = intersection / union if union != 0 else 0

    return acc_cls, recall_cls, f1_cls, cm_cls, pixel_acc, iou

# Main Function
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

        # Plotting
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

    # Average results across folds
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