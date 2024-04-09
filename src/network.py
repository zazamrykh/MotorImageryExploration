import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from src import params
from src.functions import MorletTransform, change_channels_order
from src.params import torch_dtype, device, out_dtype
from wavelets.dwt1d import cwt1d
import torch.nn.functional as F


class MWTConvNet(nn.Module):
    def __init__(self, name, scales, int_psi_scales, dropout=0.0):
        super().__init__()

        self.name = name
        self.scales = scales
        self.int_psi_scales = int_psi_scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=23, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 80x200 -> 40x100

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 5), padding=(1,2), stride=(1, 2)), # 40x100 -> 40x50
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 5), padding=1, stride=(1, 2)), # 40x50 -> 40x25
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 40x25 -> 20x12

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 20x12 -> 10x6

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2)), # 10x6 -> 5x3
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 5 * 3, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # [BatchSize, 22, 200] as input
        time_domain_signal = change_channels_order(x)
        time_domain_signal = time_domain_signal.unsqueeze(1)
        time_domain_signal = F.interpolate(time_domain_signal, size=(80, 200), mode='bilinear', align_corners=False)
        wt_result = cwt1d(x, self.scales, self.int_psi_scales, out_dtype='real', device=device)
        x = torch.cat([time_domain_signal, wt_result], dim=1)  # [BatchSize, 23, 80, 200] as input to neural network

        x = self.pool1(self.conv1(x))
        x = self.conv3(self.conv2(x))
        x = self.pool2(self.conv4(x))
        x = self.pool3(self.conv5(x))
        x = self.conv6(x)

        x = x.view(-1, 256 * 5 * 3)
        x = self.fully_connected(self.dropout(x))
        return x


class MWTConvNetL(nn.Module):
    def __init__(self, name, scales, int_psi_scales, dropout=0.0):
        super().__init__()

        self.name = name
        self.scales = scales
        self.int_psi_scales = int_psi_scales
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=23, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 80x200 -> 40x100

        self.block1 = Block(32, 64)
        # Residual
        self.stride_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=(1, 2)), # 40x100 -> 40x50
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block2 = Block(64, 128)
        self.block3 = Block(64, 128)

        self.stride_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2)), # 40x50 -> 20x25
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block4 = Block(128, 256)
        self.block5 = Block(128, 256)

        self.stride_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2)), # 20x25 -> 10x12
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block6 = Block(128, 256)
        self.block7 = Block(128, 256)

        self.stride_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, stride=(2, 2)), # 10x12 -> 5x6
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

        self.fully_connected = nn.Sequential(
            nn.Linear(256 * 5 * 7, 6),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):  # [BatchSize, 22, 200] as input
        time_domain_signal = change_channels_order(x)
        time_domain_signal = time_domain_signal.unsqueeze(1)
        time_domain_signal = F.interpolate(time_domain_signal, size=(80, 200), mode='bilinear', align_corners=False)
        wt_result = cwt1d(x, self.scales, self.int_psi_scales, out_dtype='real', device=device)
        x = torch.cat([time_domain_signal, wt_result], dim=1)  # [BatchSize, 23, 80, 200] as input to neural network

        x = self.pool1(self.conv2(self.conv1(x)))
        x = self.stride_conv1(self.block1(x) + x)

        x = self.block2(x) + x
        x = self.stride_conv2(self.block3(x) + x)

        x = self.block4(x) + x
        x = self.stride_conv3(self.block5(x) + x)

        x = self.block6(x) + x
        x = self.stride_conv4(self.block7(x) + x)

        x = x.view(-1, 256 * 5 * 7)
        x = self.fully_connected(self.dropout(x))
        return x


class Block(nn.Module):
    def __init__(self, small_size, big_size):
        super(Block, self).__init__()
        self.input_size = small_size
        self.output_size = big_size
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels=big_size, out_channels=small_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(small_size),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=small_size, out_channels=big_size, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(big_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(self.down_conv(x))



class WaveletTransform(nn.Module):
    def __init__(self, scales, int_psi_scales, out_dtype=out_dtype):
        self.scales = scales
        self.int_psi_scales = int_psi_scales
        self.out_dtype = out_dtype
        super(WaveletTransform, self).__init__()

    def forward(self, x):
        return cwt1d(x, self.scales, self.int_psi_scales, out_dtype=self.out_dtype, device=device)