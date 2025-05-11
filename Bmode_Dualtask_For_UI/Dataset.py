import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


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


def get_transforms():
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

    return train_transform, test_transform


def get_dataloaders(train_samples, val_samples, test_samples, batch_size=16):
    train_transform, test_transform = get_transforms()

    train_dataset = PathologyImageDataset(train_samples, transform=train_transform, is_train=True)
    val_dataset = PathologyImageDataset(val_samples, transform=test_transform, is_train=False)
    test_dataset = PathologyImageDataset(test_samples, transform=test_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader