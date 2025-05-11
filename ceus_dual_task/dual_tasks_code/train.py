import torch
from tqdm import tqdm

def train(model, loader, criterion_cls, criterion_seg, optimizer, device):
    model.train()
    running_loss_cls, running_loss_seg, correct_cls, total_cls = 0.0, 0.0, 0, 0
    for inputs, labels_cls, masks in tqdm(loader, desc="Training"):
        inputs = inputs.to(device)
        labels_cls = labels_cls.to(device).float().view(-1, 1)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs_cls, outputs_seg = model(inputs)

        loss_cls = criterion_cls(outputs_cls, labels_cls)
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

            loss_cls = criterion_cls(outputs_cls, labels_cls)
            running_loss_cls += loss_cls.item()

            loss_seg = criterion_seg(outputs_seg, masks)
            running_loss_seg += loss_seg.item()

            predicted_cls = (torch.sigmoid(outputs_cls) > 0.5).float()
            correct_cls += (predicted_cls == labels_cls).sum().item()
            total_cls += labels_cls.size(0)

    val_loss_cls = running_loss_cls / len(loader)
    val_loss_seg = running_loss_seg / len(loader)
    val_acc_cls = correct_cls / total_cls
    return val_loss_cls, val_loss_seg, val_acc_cls