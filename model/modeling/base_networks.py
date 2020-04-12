import torch
import math
import torch.nn.functional as F

import time

class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.fc = torch.nn.utils.spectral_norm(self.fc)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SELayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock_Pre(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.activation is not None:
            x = self.act(x)

        if (self.norm is not None) and (self.norm != 'spectral'):
            return self.bn(self.conv(x))
        else:
            return self.conv(x)

    

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.deconv = torch.nn.utils.spectral_norm(self.deconv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class RConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(RConvBlock, self).__init__()
        
        self.up = torch.nn.Upsample(scale_factor=stride, mode='bilinear')
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size-1, 1, kernel_size-(2*padding)-1, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.deconv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()


    def forward(self, x):
        # x = self.up(x)
        if (self.norm is not None) and (self.norm != 'spectral'):
            out = self.bn(self.conv(self.up(x)))
        else:
            out = self.conv(self.up(x))

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class InceptionDownBlock(torch.nn.Module):
    def __init__(self, in_channels, activation=None, norm=None):
        super(InceptionDownBlock, self).__init__()
        self.branch8x8_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch8x8_2 = ConvBlock(in_channels // 4, in_channels // 4, 8, 4, 2, activation=None)

        self.branch4x4_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch4x4_2 = ConvBlock(in_channels // 4, in_channels // 4, 4, 2, 1, activation=None)
        self.branch4x4_3 = ConvBlock(in_channels // 4, in_channels // 4, 4, 2, 1, activation=None)

        self.branch2x2_1 = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)
        self.branch2x2_2 = ConvBlock(in_channels // 4, in_channels // 4, 2, 2, 0, activation=None)
        self.branch2x2_3 = ConvBlock(in_channels // 4, in_channels // 4, 2, 2, 0, activation=None)

        self.branch_pool = ConvBlock(in_channels, in_channels // 4, 1, 1, 0, activation=None)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_size)
        elif self.norm == 'spectral':
            self.conv = torch.nn.utils.spectral_norm(self.conv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        branch8x8 = self.branch8x8_1(x)
        branch8x8 = self.branch8x8_2(branch8x8)

        branch4x4 = self.branch4x4_1(x)
        branch4x4 = self.branch4x4_2(branch4x4)
        branch4x4 = self.branch4x4_3(branch4x4)

        branch2x2 = self.branch2x2_1(x)
        branch2x2 = self.branch2x2_2(branch2x2)
        branch2x2 = self.branch2x2_3(branch2x2)

        branch_pool = self.branch_pool(x)
        branch_pool = F.avg_pool2d(branch_pool, kernel_size=8, stride=4, padding=2)

        outputs = [branch8x8, branch4x4, branch2x2, branch_pool]
        outputs = torch.cat(outputs, 1)

        if (self.norm is not None) and (self.norm != 'spectral'):
            outputs = self.bn(outputs)

        if self.activation is not None:
            outputs = self.act(outputs)

        return outputs

# class PSBlock(torch.nn.Module):
#     def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
#         super(PSBlock, self).__init__()
#         self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor**2, kernel_size, stride, padding, bias=bias)
#         self.ps = torch.nn.PixelShuffle(scale_factor)

#         self.norm = norm
#         if self.norm == 'batch':
#             self.bn = torch.nn.BatchNorm2d(output_size)
#         elif norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)
#         elif norm == 'group':
#             self,bn = torch.nn.GroupNorm(32, output_size)

#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()

#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.ps(self.conv(x)))
#         else:
#             out = self.ps(self.conv(x))

#         if self.activation is not None:
#             out = self.act(out)
#         return out


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(UpBlock, self).__init__()
            
        print(up_mode)
        if up_mode == 'deconv':
            self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv1 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'pixelshuffle':
            self.up_conv1 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        else:
            raise ValueError()

        if inception == True:
            self.down_conv = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.down_conv(h0)
        h1 = self.up_conv3(l0 - x)
        if self.se != None:
            return self.se(h1 + h0)
        else:
            return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv1 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv1 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.up_conv3 = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        else:
            raise ValueError()

        if inception == True:
            self.down_conv = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.down_conv(h0)
        h1 = self.up_conv3(l0 - x)
        if self.se != None:
            return self.se(h1 + h0)
        else:
            return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, ps_block=False, inception=False):
        super(DownBlock, self).__init__()

        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

        if inception == True:
            self.down_conv1 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
            self.down_conv3 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)


    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.up_conv(l0)
        l1 = self.down_conv3(h0 - x)
        if self.se != None:
            return self.se(l1 + l0)
        else:
            return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        
        if se_block == True:
            self.se = SELayer(num_filter, 8)
        else:
            self.se = None

        if inception == True:
            self.down_conv1 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
            self.down_conv3 = InceptionDownBlock(num_filter, activation=activation, norm=norm)
        else:
            self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
            self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

        if up_mode == 'pixelshuffle':
            self.up_conv = PSBlock(num_filter, num_filter, 4, 3, 1, 1, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'deconv':
            self.up_conv = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        elif up_mode == 'rconv':
            self.up_conv = RConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)


    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.up_conv(l0)
        l1 = self.down_conv3(h0 - x)
        if self.se != None:
            return self.se(l1 + l0)
        else:
            return l1 + l0


class PBPNBlock_Re(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, activation='prelu', norm=None, se_block=False, ps_block=False, inception=False, reconstruction=False):
        super(PBPNBlock_Re, self).__init__()

        self.up = UpBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, se_block=se_block, ps_block=ps_block, inception=inception)
        self.down = DownBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, se_block=se_block, inception=inception)
        if reconstruction:
            self.reconstructor = ConvBlock(base_filter, 3, 3, 1, 1, activation=None, norm=None)
            self.conv = ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm)
            self.compress = ConvBlock(base_filter * 2, base_filter, 1, 1, 0, activation=None, norm=None)
        else:
            self.reconstructor = None

    def forward(self, x, num_expand, Bicubic_LR=None):  # [l1, h2, h4, l2]
        if num_expand == 0:
            l1 = x
            l1 = torch.cat((l1, self.conv(l1)), 1)

            return [l1]

        elif num_expand == 1:
            l1 = x
            h2 = self.up(l1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1 = x
            h2 = self.up(l1)
            h4 = self.up(h2)
            
            if self.reconstructor != None:
                reconst_image = self.reconstructor(h4)
                reconst_image = reconst_image + Bicubic_LR
                h4 = torch.cat((h4, self.conv(reconst_image)), 1)
                h4 = self.compress(h4)
            else:
                reconst_image = None

            l2 = self.down(h4)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2, h4, l2], reconst_image


class D_PBPNBlock_Re(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, se_block=False, ps_block=False, inception=False, reconstruction=False):
        super(D_PBPNBlock_Re, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, se_block=se_block, ps_block=ps_block, inception=inception)
        self.down = D_DownBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, se_block=se_block, inception=inception)
        if reconstruction:
            self.reconstructor = ConvBlock(base_filter, 3, 3, 1, 1, activation=None, norm=None)
            self.conv = ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm)
            self.compress = ConvBlock(base_filter * 2, base_filter, 1, 1, 0, activation=None, norm=None)
        else:
            self.reconstructor = None

    def forward(self, x, num_expand, Bicubic_LR=None):
        if num_expand == 0:
            l1 = x[0]
            l1 = torch.cat((l1, self.conv(l1)), 1)
            return [l1]

        elif num_expand == 1:
            l1, h2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1, h2, h4, l2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)

            if self.reconstructor != None:
                h4_recon = self.up(h2)
                reconst_image = self.reconstructor(h4_recon)
                reconst_image = reconst_image + Bicubic_LR
                h4_recon = torch.cat((h4_recon, self.conv(reconst_image)), 1)
                h4_recon = self.compress(h4_recon)
                h4 = torch.cat((h4, h4_recon), 1)
            else:
                h4 = torch.cat((h4, self.up(h2)), 1)
                reconst_image = None

            l2 = torch.cat((l2, self.down(h4)), 1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2, h4, l2], reconst_image


class DL_PBPNBlock_Re(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, se_block=False, ps_block=False, inception=False):
        super(DL_PBPNBlock_Re, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, se_block=se_block, ps_block=ps_block, inception=inception)
        self.conv = ConvBlock(base_filter * stage, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.reconstructor = ConvBlock(base_filter, 3, 3, 1, 1, activation=None, norm=None)

    def forward(self, x, num_expand, Bicubic_LR=None):
        if num_expand == 0:
            return x

        elif num_expand == 1:
            l1, h2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1, h2, h4, l2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)
            h4 = torch.cat((h4, self.up(h2)), 1)

            h4 = self.conv(h4)
            reconst_image = self.reconstructor(h4)
            if Bicubic_LR != None:
                reconst_image = reconst_image + Bicubic_LR

            return [l1, h2, h4, l2], reconst_image


class UNetBlock(torch.nn.Module):
    def __init__(self, base_filter=64, activation=None, norm=None):
        super(UNetBlock, self).__init__()

        if norm == 'none':
            norm = None

        self.conv_blocks = torch.nn.ModuleList([
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
        ])
        self.deconv_blocks = torch.nn.ModuleList([
            DeconvBlock(base_filter*4, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
        ])


    def forward(self, x):
        sources = [] # 1 2 2
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)
           
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
        
        return x

#### DBPN_Block ####
# functions to load pre-trained dbpn model
# these functions are used for dbpn and dbpn-ll
# dbpn-custom doesn't use these functions

class UpBlock_DBPN(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock_DBPN, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)        

    def forward(self, x):
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class D_UpBlock_DBPN(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock_DBPN, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)        

    def forward(self, x):
    	x = self.conv(x)
    	h0 = self.up_conv1(x)
    	l0 = self.up_conv2(h0)
    	h1 = self.up_conv3(l0 - x)
    	return h1 + h0

class DownBlock_DBPN(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock_DBPN, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

class D_DownBlock_DBPN(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock_DBPN, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation=activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation=activation, norm=None)

    def forward(self, x):
    	x = self.conv(x)
    	l0 = self.down_conv1(x)
    	h0 = self.down_conv2(l0)
    	l1 = self.down_conv3(h0 - x)
    	return l1 + l0

#### OLD ######
class PBPNBlock(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(PBPNBlock, self).__init__()

        self.up = UpBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        self.down = DownBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)

    def forward(self, x, num_expand):  # [l1, h2, h4, l2]
        if num_expand == 0:
            l1 = x
            l1 = torch.cat((l1, self.conv(l1)), 1)

            return [l1]

        elif num_expand == 1:
            l1 = x
            h2 = self.up(l1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1 = x
            h2 = self.up(l1)
            h4 = self.up(h2)
            l2 = self.down(h4)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2, h4, l2]


class D_PBPNBlock(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(D_PBPNBlock, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        self.down = D_DownBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)

    def forward(self, x, num_expand):
        if num_expand == 0:
            l1 = x[0]
            l1 = torch.cat((l1, self.conv(l1)), 1)
            return [l1]

        elif num_expand == 1:
            l1, h2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1, h2, h4, l2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)
            h4 = torch.cat((h4, self.up(h2)), 1)
            l2 = torch.cat((l2, self.down(h4)), 1)
            l1 = torch.cat((l1, self.down(h2)), 1)

            return [l1, h2, h4, l2]


class DL_PBPNBlock(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(DL_PBPNBlock, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)

    def forward(self, x, num_expand):
        if num_expand == 0:
            return x

        elif num_expand == 1:
            l1, h2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)

            return [l1, h2]

        elif num_expand == 2:
            l1, h2, h4, l2 = x
            h2 = torch.cat((h2, self.up(l1)), 1)
            h4 = torch.cat((h4, self.up(h2)), 1)
            return [l1, h2, h4, l2]