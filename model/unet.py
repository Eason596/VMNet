from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            norm_type(out_channels),
            act_type()
        )

    def forward(self, x):
        return self.layer(x)


class BasicConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU):
        super().__init__()
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    norm_type=norm_type,
                    act_type=act_type))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class FCNHead(nn.Module):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 dropout_ratio=0.1,
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU,
                 in_index=-1,
                 num_convs=2,
                 kernel_size=3,
                 dilation=1):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.kernel_size = kernel_size
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.norm_type = norm_type
        self.act_type = act_type
        self.in_index = in_index

        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm_type=norm_type,
                act_type=act_type))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm_type=norm_type,
                    act_type=act_type))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward function."""
        x = x[self.in_index]
        x = self.convs(x)
        x = self.dropout(x)
        x = self.conv_seg(x)
        return x


class InterpConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0):
        super().__init__()

        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_type=norm_type,
            act_type=act_type)
        upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        return self.interp_upsample(x)


class UpConvBlock(nn.Module):
    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU,
                 upsample=False):
        super().__init__()

        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            norm_type=norm_type,
            act_type=act_type)
        if upsample:
            self.upsample = InterpConv(in_channels=in_channels,
                                       out_channels=skip_channels,
                                       norm_type=norm_type,
                                       act_type=act_type)
        else:
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_type=norm_type,
                act_type=act_type)

    def forward(self, skip, x):
        x = self.upsample(x)
        out = torch.cat([skip, x], dim=1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 norm_type=nn.BatchNorm2d,
                 act_type=nn.ReLU,
                 norm_eval=False, ):
        super().__init__()
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, ' \
            f'while the strides is {strides}, the length of ' \
            f'strides is {len(strides)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, ' \
            f'while the enc_num_convs is {enc_num_convs}, the length of ' \
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages - 1), \
            'The length of dec_num_convs should be equal to (num_stages-1), ' \
            f'while the dec_num_convs is {dec_num_convs}, the length of ' \
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(downsamples) == (num_stages - 1), \
            'The length of downsamples should be equal to (num_stages-1), ' \
            f'while the downsamples is {downsamples}, the length of ' \
            f'downsamples is {len(downsamples)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, ' \
            f'while the enc_dilations is {enc_dilations}, the length of ' \
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages - 1), \
            'The length of dec_dilations should be equal to (num_stages-1), ' \
            f'while the dec_dilations is {dec_dilations}, the length of ' \
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is ' \
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if strides[i] == 1 and downsamples[i - 1]:
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2 ** i,
                        skip_channels=base_channels * 2 ** (i - 1),
                        out_channels=base_channels * 2 ** (i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        norm_type=norm_type,
                        act_type=act_type,
                        upsample=True if upsample else False))

            enc_conv_block.append(
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2 ** i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    norm_type=norm_type,
                    act_type=act_type))
            self.encoder.append(nn.Sequential(*enc_conv_block))
            in_channels = base_channels * 2 ** i

    def forward(self, x):
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return dec_outs


if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    head = FCNHead(in_channels=64,
                   in_index=4,
                   channels=64,
                   num_convs=1,
                   dropout_ratio=0.1,
                   num_classes=1)
    print(head(out).shape)