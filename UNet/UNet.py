
import torch.nn as nn
import torch

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_rate=0.1, l2_lambda=0.001):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                      padding_mode='zeros'),  # Remove weight_decay from here
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False,
                      padding_mode='zeros'),  # Remove weight_decay from here
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        return x
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, l2_lambda=0.001):
        super(UNet, self).__init__()
        # Encoder
        self.encoder1 = ConvBlock(in_channels, 64, dropout_rate, l2_lambda)
        self.encoder2 = ConvBlock(64, 128, dropout_rate, l2_lambda)
        self.encoder3 = ConvBlock(128, 256, dropout_rate, l2_lambda)
        self.encoder4 = ConvBlock(256, 512, dropout_rate, l2_lambda)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, dropout_rate, l2_lambda)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512, dropout_rate, l2_lambda)  # Concatenate with encoder4

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256, dropout_rate, l2_lambda)  # Concatenate with encoder3

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128, dropout_rate, l2_lambda)  # Concatenate with encoder2

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64, dropout_rate, l2_lambda)  # Concatenate with encoder1

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool2d(2)(e1))
        e3 = self.encoder3(nn.MaxPool2d(2)(e2))
        e4 = self.encoder4(nn.MaxPool2d(2)(e3))

        # Bottleneck
        b = self.bottleneck(nn.MaxPool2d(2)(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Concatenate
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)  # Concatenate
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)  # Concatenate
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)  # Concatenate
        d1 = self.decoder1(d1)

        out = self.out_conv(d1)
        return out