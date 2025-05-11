import torch
import torch.nn as nn
from torchvision import models

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


def get_model(device):
    model = CNNUNetModel().to(device)
    return model


def get_loss_functions():
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_seg = nn.BCELoss()
    return criterion_cls, criterion_seg