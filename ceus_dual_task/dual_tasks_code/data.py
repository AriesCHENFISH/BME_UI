import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

# 自定义数据集类，用于分类和分割任务
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

        mask_dir = os.path.join(patient_dir, 'mask_roi')
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        if mask_files:
            mask_path = os.path.join(mask_dir, mask_files[0])
            try:
                mask = Image.open(mask_path).convert('L')
                if do_flip:
                    mask = TF.hflip(mask)
                mask = TF.rotate(mask, angle)
                mask = TF.resize(mask, (224, 224))
                mask = TF.to_tensor(mask)
                mask = (mask > 0.5).float()
            except Exception as e:
                print(f"加载 {mask_path} 失败: {e}")
                mask = torch.zeros(1, 224, 224)
        else:
            mask = torch.zeros(1, 224, 224)

        return sequence, label, mask

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