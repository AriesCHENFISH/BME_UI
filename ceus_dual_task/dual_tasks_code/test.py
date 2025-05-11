import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import os
from PIL import Image

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
                if masks[i].sum() > 0:  # 仅处理有分割标签的样本
                    seg_pred = outputs_seg[i].cpu().numpy()
                    all_masks.append(masks[i].cpu().numpy())
                    all_seg_preds.append(seg_pred)
                    seg_pred_binary = (seg_pred > 0.5).astype(np.uint8) * 255
                    Image.fromarray(seg_pred_binary[0]).save(os.path.join('segmentation', f'seg_{idx}.png'))
                idx += 1

    acc_cls = accuracy_score(all_labels_cls, all_preds_cls)
    recall_cls = recall_score(all_labels_cls, all_preds_cls)
    f1_cls = f1_score(all_labels_cls, all_preds_cls)
    cm_cls = confusion_matrix(all_labels_cls, all_preds_cls)

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