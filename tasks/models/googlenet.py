'''
GoogLeNet with PyTorch
Reference: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple

class GoogLeNet(nn.Module):
    def __init__(
        self,*,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.4,
        **args: Any
    ) -> None:
        super().__init__()

        conv_block = BasicConv2d
        inception_block = Inception

        self.conv1 = conv_block(input_channels, 192, kernel_size=3, padding=1)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 32 x 32
        x = self.conv1(x) # Pre Convolution

        # N x 192 x 32 x 32
        x = self.inception3a(x)
        # N x 256 x 32 x 32
        x = self.inception3b(x)
        # N x 480 x 32 x 32
        x = self.maxpool1(x)
        # N x 480 x 16 x 16
        x = self.inception4a(x)
        # N x 512 x 16 x 16

        x = self.inception4b(x)
        # N x 512 x 16 x 16
        x = self.inception4c(x)
        # N x 512 x 16 x 16
        x = self.inception4d(x)
        # N x 528 x 16 x 16

        x = self.inception4e(x)
        # N x 832 x 16 x 16
        x = self.maxpool2(x)
        # N x 832 x 8 x 8
        x = self.inception5a(x)
        # N x 832 x 8 x 8
        x = self.inception5b(x)
        # N x 1024 x 8 x 8

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 10 (num_classes)
        return x

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return F.relu(x, inplace=True)


def test():
    net = GoogLeNet()
    x = torch.randn(10,3,32,32)
    y = net(x)
    print(y.size())
    net = GoogLeNet(input_channels=1, num_classes=10)
    x = torch.randn(10,1,32,32)
    y = net(x)
    print(y.size())
    x = torch.randn(10,1,28,28)
    y = net(x)
    print(y.size())

if __name__ == '__main__':
    test()
