import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import SPPF, C2f
from ultralytics.nn.modules.head import Detect



class YOLOv8m(nn.Module):
    def __init__(self, nc=80):
        super(YOLOv8m, self).__init__()
        width, depth = 0.75, 0.67
        self.backbone = nn.Sequential(
            Conv(3, int(64 * width), 3, 2),  # P1/2
            Conv(int(64 * width), int(128 * width), 3, 2),  # P2/4
            C2f(int(128 * width), int(128 * width), int(3 * depth)),
            Conv(int(128 * width), int(256 * width), 3, 2),  # P3/8
            C2f(int(256 * width), int(256 * width), int(6 * depth)),
            Conv(int(256 * width), int(512 * width), 3, 2),  # P4/16
            C2f(int(512 * width), int(512 * width), int(6 * depth)),
            Conv(int(512 * width), int(1024 * width), 3, 2),  # P5/32
            C2f(int(1024 * width), int(1024 * width), int(3 * depth)),
            SPPF(int(1024 * width), int(1024 * width), 5)
        )
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(),
            C2f(int(1024 * width) + int(512 * width), int(512 * width), int(3 * depth)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            Concat(),
            C2f(int(512 * width) + int(256 * width), int(256 * width), int(3 * depth)),
            Conv(int(256 * width), int(256 * width), 3, 2),
            Concat(),
            C2f(int(512 * width), int(512 * width), int(3 * depth)),
            Conv(int(512 * width), int(512 * width), 3, 2),
            Concat(),
            C2f(int(1024 * width), int(1024 * width), int(3 * depth)),
            Detect(nc, ch=(256, 512, 1024))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class Concat(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)


model = YOLOv8m(nc=80)
print(model)