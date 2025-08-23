import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif type(layer) == nn.BatchNorm1d:
        layer.weight.data.normal_(1.0, 0.02)
    elif type(layer) == nn.Linear:
        layer.weight.data.normal_(0.0, 1e-4)


class Disentangler(nn.Module):
    def __init__(self):
        super(Disentangler, self).__init__()
        self.output_dim = 64
        self.reduc_factor = 1
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(self.output_dim, int(self.output_dim / self.reduc_factor))
        self.bn1_fc = nn.BatchNorm1d(int(self.output_dim / self.reduc_factor))
        self.apply(init_weights)

    def forward(self, x):
        x = self.avgpool(x)  # [B, 64, 1, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        x = F.relu(self.bn1_fc(self.fc1(x)))  # [B, 64/reduc_factor]
        return x


class PhysNet(nn.Module):
    def __init__(self, config, in_ch=3):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU(),
        )
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        # encoder
        self.encoder1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # rPPG
        self.rppg_loop = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.rppg_disentangler = Disentangler()
        self.rppg_decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.rppg_decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.rppg_end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)),
        )

        # Head motion
        self.head_loop = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.head_disentangler = Disentangler()
        self.head_decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.head_decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.head_end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)),
        )

        # Illumination
        self.illum_loop = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.illum_disentangler = Disentangler()
        self.illum_decoder1 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.illum_decoder2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.illum_end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)),
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.size()

        means = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        stds = torch.std(x, dim=(2, 3, 4), keepdim=True)
        x = (x - means) / stds  # [B, C, T, 128, 128]

        parity = []
        x = self.start(x)  # [B, C, T, 128, 128]
        x = self.loop1(x)  # [B, 64, T, 64, 64]
        parity.append(x.size(2) % 2)
        x = self.encoder1(x)  # [B, 64, T/2, 32, 32]
        parity.append(x.size(2) % 2)
        x = self.encoder2(x)  # [B, 64, T/4, 16, 16]
        x = self.loop4(x)  # [B, 64, T/4, 8, 8]

        # rPPG
        x_rppg = self.rppg_loop(x)  # [B, 64, T/4, 8, 8]
        x_fc_rppg = self.rppg_disentangler(x_rppg)  # Disentangling, [B, 64/reduc_factor]
        x_rppg = F.interpolate(x_rppg, scale_factor=(2, 1, 1))  # [B, 64, T/2, 8, 8]
        x_rppg = self.rppg_decoder1(x_rppg)  # [B, 64, T/2, 8, 8]
        x_rppg = F.pad(x_rppg, (0, 0, 0, 0, 0, parity[-1]), mode="replicate")
        x_rppg = F.interpolate(x_rppg, scale_factor=(2, 1, 1))  # [B, 64, T, 8, 8]
        x_rppg = self.rppg_decoder2(x_rppg)  # [B, 64, T, 8, 8]
        x_rppg = F.pad(x_rppg, (0, 0, 0, 0, 0, parity[-2]), mode="replicate")
        x_rppg = self.rppg_end(x_rppg)  # [B, 1, T, 1, 1]
        X_rppg = x_rppg.squeeze()  # [B, T]

        # Head motion
        x_head = self.head_loop(x)  # [B, 64, T/4, 8, 8]
        x_fc_head = self.head_disentangler(x_head)  # Disentangling, [B, 64/reduc_factor]
        x_head = F.interpolate(x_head, scale_factor=(2, 1, 1))  # [B, 64, T/2, 8, 8]
        x_head = self.head_decoder1(x_head)  # [B, 64, T/2, 8, 8]
        x_head = F.pad(x_head, (0, 0, 0, 0, 0, parity[-1]), mode="replicate")
        x_head = F.interpolate(x_head, scale_factor=(2, 1, 1))  # [B, 64, T, 8, 8]
        x_head = self.head_decoder2(x_head)  # [B, 64, T, 8, 8]
        x_head = F.pad(x_head, (0, 0, 0, 0, 0, parity[-2]), mode="replicate")
        x_head = self.head_end(x_head)  # [B, 1, T, 1, 1]
        X_head = x_head.squeeze()  # [B, T]

        # Illumination
        x_illum = self.illum_loop(x)  # [B, 64, T/4, 8, 8]
        x_fc_illum = self.illum_disentangler(x_illum)  # Disentangling, [B, 64/reduc_factor]
        x_illum = F.interpolate(x_illum, scale_factor=(2, 1, 1))  # [B, 64, T/2, 8, 8]
        x_illum = self.illum_decoder1(x_illum)  # [B, 64, T/2, 8, 8]
        x_illum = F.pad(x_illum, (0, 0, 0, 0, 0, parity[-1]), mode="replicate")
        x_illum = F.interpolate(x_illum, scale_factor=(2, 1, 1))  # [B, 64, T, 8, 8]
        x_illum = self.illum_decoder2(x_illum)  # [B, 64, T, 8, 8]
        x_illum = F.pad(x_illum, (0, 0, 0, 0, 0, parity[-2]), mode="replicate")
        x_illum = self.illum_end(x_illum)  # [B, 1, T, 1, 1]
        X_illum = x_illum.squeeze()  # [B, T]

        return X_rppg, X_head, X_illum, x_fc_rppg, x_fc_head, x_fc_illum  # [B, T]*3, [B, 64]*3
