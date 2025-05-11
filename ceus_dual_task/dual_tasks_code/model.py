import torch
import torch.nn as nn
from torchvision import models

# 双任务模型，包含共享的 ResNet-50 编码器和 U-Net 风格的解码器
class CNNRNNUNetModel(nn.Module):
    def __init__(self):
        super(CNNRNNUNetModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        self.encoder_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.rnn = nn.LSTM(input_size=2048, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 2, 1)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        enc0 = self.encoder_conv1(x)
        x = self.maxpool(enc0)
        enc1 = self.encoder_layer1(x)
        enc2 = self.encoder_layer2(enc1)
        enc3 = self.encoder_layer3(enc2)
        enc4 = self.encoder_layer4(enc3)

        features_seg = enc4.view(B, T, 2048, 7, 7).mean(dim=1)
        dec1 = self.upconv1(features_seg)
        skip3 = enc3.view(B, T, 1024, 14, 14).mean(dim=1)
        dec1 = torch.cat((dec1, skip3), dim=1)
        dec1 = self.conv_up1(dec1)

        dec2 = self.upconv2(dec1)
        skip2 = enc2.view(B, T, 512, 28, 28).mean(dim=1)
        dec2 = torch.cat((dec2, skip2), dim=1)
        dec2 = self.conv_up2(dec2)

        dec3 = self.upconv3(dec2)
        skip1 = enc1.view(B, T, 256, 56, 56).mean(dim=1)
        dec3 = torch.cat((dec3, skip1), dim=1)
        dec3 = self.conv_up3(dec3)

        dec4 = self.upconv4(dec3)
        skip0 = enc0.view(B, T, 64, 112, 112).mean(dim=1)
        dec4 = torch.cat((dec4, skip0), dim=1)
        dec4 = self.conv_up4(dec4)

        seg_output = self.upconv5(dec4)
        seg_output = self.conv_up5(seg_output)

        features_cls = enc4.view(B, T, 2048, 7, 7).mean(dim=3).mean(dim=3)
        _, (h_n, _) = self.rnn(features_cls)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        h_n = self.dropout(h_n)
        cls_output = self.fc(h_n)

        return cls_output, seg_output