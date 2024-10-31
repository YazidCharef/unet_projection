import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_frames):
        super(TemporalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Encoder (downsampling)
        self.enc1 = self.conv_block(n_channels * n_frames, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Decoder (upsampling)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = nn.Conv2d(64, n_classes * n_frames, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, n_channels, n_frames, height, width = x.shape
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, n_channels * n_frames, height, width)

        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        d4 = self.dec4(torch.cat([self.pad_to_match(F.interpolate(e4, scale_factor=2), e3), e3], dim=1))
        d3 = self.dec3(torch.cat([self.pad_to_match(F.interpolate(d4, scale_factor=2), e2), e2], dim=1))
        d2 = self.dec2(torch.cat([self.pad_to_match(F.interpolate(d3, scale_factor=2), e1), e1], dim=1))
        d1 = self.dec1(d2)

        d1 = self.crop_to_match(d1, x)

        _, c, h, w = d1.shape
        out = d1.view(batch_size, self.n_frames, self.n_classes, h, w)

        return out

    def pad_to_match(self, x, target):
        if x.shape[2] > target.shape[2]:
            diff = x.shape[2] - target.shape[2]
            x = x[:, :, diff//2:-(diff-diff//2), diff//2:-(diff-diff//2)]
        elif x.shape[2] < target.shape[2]:
            diff = target.shape[2] - x.shape[2]
            x = F.pad(x, [diff//2, diff-diff//2, diff//2, diff-diff//2])
        return x

    def crop_to_match(self, x, target):
        if x.shape[2] > target.shape[2] or x.shape[3] > target.shape[3]:
            x = F.center_crop(x, (target.shape[2], target.shape[3]))
        return x