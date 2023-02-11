import torch
from torch import nn
import torch.nn.functional as F


class UNetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1),
                 init_stride=(1, 1, 1)):
        super(UNetConv3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                   nn.InstanceNorm3d(out_size) if is_batchnorm else nn.Identity(),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                   nn.InstanceNorm3d(out_size) if is_batchnorm else nn.Identity(),
                                   nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class InterpConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(InterpConv, self).__init__()
        self.conv = UNetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)

        if outputs1.shape[4] != outputs2.shape[4]:
            outputs1 = F.interpolate(outputs1, size=inputs1.shape[2:], mode="trilinear", align_corners=False)
            outputs2 = F.interpolate(outputs2, size=inputs1.shape[2:], mode="trilinear", align_corners=False)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class UNet3D(nn.Module):
    def __init__(self, cfg, is_batchnorm=True):
        super(UNet3D, self).__init__()
        filters = [64, 128, 256, 512]
        self.conv1 = UNetConv3(cfg['in_channels'], filters[0], is_batchnorm, kernel_size=3, padding_size=1,
                               init_stride=1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = UNetConv3(filters[0], filters[1], is_batchnorm, kernel_size=3, padding_size=1, init_stride=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = UNetConv3(filters[1], filters[2], is_batchnorm, kernel_size=3, padding_size=1, init_stride=1)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.center = UNetConv3(filters[2], filters[3], is_batchnorm, kernel_size=3, padding_size=1, init_stride=1)

        self.up3 = InterpConv(filters[3], filters[2], is_batchnorm)
        self.up2 = InterpConv(filters[2], filters[1], is_batchnorm)
        self.up1 = InterpConv(filters[1], filters[0], is_batchnorm)

        self.out_couv = nn.Conv3d(filters[0], cfg['num_classes'], 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.maxpool1(conv1)
        conv2 = self.conv2(x)
        x = self.maxpool2(conv2)
        conv3 = self.conv3(x)
        x = self.maxpool3(conv3)
        center = self.center(x)
        x = self.up3(conv3, center)
        x = self.up2(conv2, x)
        x = self.up1(conv1, x)
        x = self.out_couv(x)
        return x


if __name__ == '__main__':
    cfg = {
        'in_channels': 1,
        'num_classes': 1
    }
    device = torch.device('cuda:1')
    model = UNet3D(cfg).to(device)
    x = torch.randn((2, 1, 128, 128, 128)).to(device)
    print(model(x).shape)
    # torchsummary.summary(model, (3, 512, 512, 8), 1, 'cuda')
