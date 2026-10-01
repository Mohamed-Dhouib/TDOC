"""Frequency-side modules copied from the official FFDN implementation."""

from __future__ import annotations

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from efficientnet_pytorch.utils import (  # type: ignore[import]
        drop_connect,
        get_same_padding_conv2d,
    )
except ImportError as exc:  # pragma: no cover - provides a clear error for missing dependency
    raise ImportError(
        'efficientnet-pytorch is required for the FFDN frequency head. Install it via "pip install efficientnet-pytorch".'
    ) from exc


BlockArgs = collections.namedtuple(
    'BlockArgs',
    ['num_repeat', 'kernel_size', 'stride', 'expand_ratio', 'input_filters', 'output_filters', 'se_ratio', 'id_skip'],
)


GlobalParams = collections.namedtuple(
    'GlobalParams',
    [
        'width_coefficient',
        'depth_coefficient',
        'image_size',
        'dropout_rate',
        'num_classes',
        'batch_norm_momentum',
        'batch_norm_epsilon',
        'drop_connect_rate',
        'depth_divisor',
        'min_depth',
        'include_top',
    ],
)


global_params = GlobalParams(
    width_coefficient=1.8,
    depth_coefficient=2.6,
    image_size=528,
    dropout_rate=0.0,
    num_classes=1000,
    batch_norm_momentum=0.99,
    batch_norm_epsilon=0.001,
    drop_connect_rate=0.0,
    depth_divisor=8,
    min_depth=None,
    include_top=True,
)


def get_width_and_height_from_size(value):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError('image_size must be int or tuple')


def calculate_output_image_size(image_size, stride):
    if image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return image_height, image_width


class MBConvBlock(nn.Module):
    """Mobile inverted residual bottleneck block."""

    def __init__(self, block_args: BlockArgs, global_params: GlobalParams, image_size=25):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio

        if self._block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = int(self._block_args.kernel_size)
        s = int(self._block_args.stride) if isinstance(self._block_args.stride, (list, tuple)) else self._block_args.stride
        pad = k // 2
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=self._block_args.kernel_size,
            stride=self._block_args.stride,
            padding=pad,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, self._block_args.stride)

        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = nn.SiLU(inplace=True)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient: bool = True) -> None:
        # if memory_efficient:
        #     self._swish = MemoryEfficientSwish()
        # else:
        #     self._swish = Swish()
        self._swish = nn.SiLU(inplace=True)


class AddCoords(nn.Module):
    def __init__(self, with_r: bool = True) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_c, yy_c = torch.meshgrid(
            torch.arange(x_dim, dtype=input_tensor.dtype, device=input_tensor.device),
            torch.arange(y_dim, dtype=input_tensor.dtype, device=input_tensor.device),
            indexing='ij',
        )
        xx_c = xx_c.expand(batch_size, 1, x_dim, y_dim) / (x_dim - 1) * 2 - 1
        yy_c = yy_c.expand(batch_size, 1, x_dim, y_dim) / (y_dim - 1) * 2 - 1
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_c - 0.5, 2) + torch.pow(yy_c - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class FPH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.obembed = nn.Embedding(21, 21).from_pretrained(torch.eye(21))
        self.qtembed = nn.Embedding(64, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.addcoords = AddCoords()
        repeats = (1, 1, 1)
        in_channels = (256, 256, 256)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=256, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            MBConvBlock(
                BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=1, expand_ratio=6,
                          input_filters=in_channels[0], output_filters=in_channels[1], se_ratio=0.25,
                          id_skip=True),
                global_params,
            ),
            MBConvBlock(
                BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=1, expand_ratio=6,
                          input_filters=in_channels[1], output_filters=in_channels[1], se_ratio=0.25,
                          id_skip=True),
                global_params,
            ),
            MBConvBlock(
                BlockArgs(num_repeat=repeats[0], kernel_size=3, stride=1, expand_ratio=6,
                          input_filters=in_channels[1], output_filters=in_channels[1], se_ratio=0.25,
                          id_skip=True),
                global_params,
            ),
        )

    def forward(self, x, qtable):
        x = self.conv2(self.conv1(self.obembed(x).permute(0, 3, 1, 2).contiguous()))
        batch_size, channels, height, width = x.shape
        qtable = self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1).long()).transpose(1, 6).squeeze(6).contiguous()
        fused = (
            x.reshape(batch_size, channels, height // 8, 8, width // 8, 8)
            .permute(0, 1, 3, 5, 2, 4)
            * qtable
        )
        fused = fused.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, channels, height, width)
        return self.conv0(self.addcoords(torch.cat((fused, x), dim=1)))


class SCSEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


__all__ = ['FPH', 'SCSEModule']
