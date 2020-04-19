import torch
import math
import torch.nn.functional as F

import time
from .base_networks import UpBlock, DownBlock, ConvBlock, D_UpBlock, D_DownBlock

#################################################################################

# PBPN Custom module

#################################################################################

class PBPNBlock_Custom(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False, reconstruction=False):
        super(PBPNBlock_Custom, self).__init__()

        self.up = UpBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        self.down = DownBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        
        if reconstruction:
            self.reconstructor = ConvBlock(base_filter, 3, 3, 1, 1, activation=None, norm=None)
        else:
            self.reconstructor = None

    def forward(self, x, num_expand, resized_img=None):
        feature_maps = [x]

        # upscale
        for _ in range(num_expand):
            feature_maps.append(self.up(feature_maps[-1]))
        
        # reconst image from highest resolution feature maps
        if self.reconstructor != None:
            reconst_img = self.reconst_image(feature_maps[num_expand], resized_img)
        else:
            reconst_img = None
        
        # down scale
        for i in range(num_expand):
            if i == num_expand - 1:
                feature_maps[0] = torch.cat((feature_maps[0], self.down(feature_maps[-1])), 1)
            else:
                feature_maps.append(self.down(feature_maps[-1]))

        return feature_maps, reconst_img

    def reconst_image(self,x,resized_img=None):
        reconst_image = self.reconstructor(x)

        if resized_img != None:
            reconst_image = reconst_image + resized_img

        return reconst_image

class D_PBPNBlock_Custom(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False, reconstruction=False):
        super(D_PBPNBlock_Custom, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        self.down = D_DownBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        
        if reconstruction:
            self.reconstructor = ConvBlock(base_filter * stage, 3, 3, 1, 1, activation=None, norm=None)
        else:
            self.reconstructor = None

    def forward(self, x, num_expand, resized_img=None):
        for i in range(num_expand):
            x[i + 1] = torch.cat((x[i + 1], self.up(x[i])), 1)

        # reconst image from highest resolution feature maps
        if self.reconstructor != None:
            reconst_img = self.reconst_image(x[num_expand], resized_img)
        else:
            reconst_img = None

        for i in range(num_expand):
            if i == num_expand - 1:
                x[0] = torch.cat((x[0], self.down(x[-1])), 1)
            else:
                x[num_expand + i + 1] = torch.cat((x[num_expand + i + 1], self.down(x[num_expand + i])), 1)

        return x, reconst_img

    def reconst_image(self,x,resized_img=None):
        reconst_image = self.reconstructor(x)

        if resized_img != None:
            reconst_image = reconst_image + resized_img

        return reconst_image

class DL_PBPNBlock_Custom(torch.nn.Module):
    def __init__(self, base_filter, kernel, stride, padding, stage, activation='prelu', norm=None, up_mode='deconv', se_block=False, inception=False):
        super(DL_PBPNBlock_Custom, self).__init__()

        self.up = D_UpBlock(base_filter, kernel, stride, padding, stage, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception)
        self.conv = ConvBlock(base_filter * stage, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.reconstructor = ConvBlock(base_filter, 3, 3, 1, 1, activation=None, norm=None)

    def forward(self, x, num_expand, resized_img=None):
        for i in range(num_expand):
            x[i + 1] = torch.cat((x[i + 1], self.up(x[i])), 1)

        reconst_img = self.conv(x[num_expand])
        reconst_img = self.reconstructor(reconst_img)
        if resized_img != None:
            reconst_img = reconst_img + resized_img

        return x, reconst_img

#################################################################################

# PBPN module

#################################################################################
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