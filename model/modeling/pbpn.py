import torch
import torch.nn as nn

from .base_networks import *


class PBPN_Re(nn.Module):
    def __init__(self, cfg):
        super(PBPN_Re, self).__init__()

        kernel = 8
        stride = 4
        padding = 2

        feat = 256
        base_filter = 64

        activation = cfg.MODEL.GEN_ACTIVATION
        norm = cfg.MODEL.GEN_NORM
        if norm == 'none':
            norm = None
        num_stages = cfg.MODEL.NUM_STAGES
        
        se_block = cfg.MODEL.SE_MODULE
        ps_block = cfg.MODEL.PS_MODULE
        inception_block = cfg.MODEL.INCEPTION_MODULE
        reconstruction = cfg.MODEL.RECONSTRUCTION
        if reconstruction:
            self.bicubic = torch.nn.Upsample(scale_factor=16, mode='bicubic')
        else:
            self.bicubic = None

        if cfg.MODEL.INIT_UNET == True:
            self.init_block = nn.ModuleList([ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(base_filter, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        UNetBlock(base_filter=base_filter, activation=activation, norm=norm)])
        else:
            self.init_block = nn.ModuleList([ConvBlock(3, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, base_filter, 1, 1, 0, activation=activation, norm=norm)])

        layers = []
        for i in range(1, num_stages):
            if i == 1:
                layers.append(PBPNBlock_Re(base_filter, kernel, stride, padding, activation=activation, norm=norm, se_block=se_block, ps_block=ps_block,inception=inception_block, reconstruction=reconstruction))
            else:
                layers.append(D_PBPNBlock_Re(base_filter, kernel, stride, padding, i, activation=activation, norm=norm, se_block=se_block, ps_block=ps_block, inception=inception_block, reconstruction=reconstruction))
        layers.append(DL_PBPNBlock_Re(base_filter, kernel, stride, padding, num_stages, activation=activation, norm=norm, ps_block=ps_block, inception=inception_block))

        self.pbpn = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # self.feat0.apply(weights_init)
        # self.feat1.apply(weights_init)
        self.init_block.apply(weights_init)
        self.pbpn.apply(weights_init)

    def forward(self, x, num_expand=2, only_last=False):
        if self.bicubic != None:
            bicubic = self.bicubic(x)
        else:
            bicubic = None

        for layer in self.init_block:
            x = layer(x)

        # x = self.feat0(x)
        # x = self.feat1(x)
        output_list = []

        for layer in self.pbpn:
            x, reconst_image = layer(x, num_expand, bicubic)
            if reconst_image != None:
                output_list.append(reconst_image)
        
        if only_last:
            return output_list[-1]
        else:
            return output_list

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(fix_model_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage)))


class PBPN(nn.Module):
    def __init__(self, cfg):
        super(PBPN, self).__init__()

        kernel = 8
        stride = 4
        padding = 2

        feat = 256
        base_filter = 64

        activation = cfg.MODEL.GEN_ACTIVATION
        norm = cfg.MODEL.GEN_NORM
        if norm == 'none':
            norm = None
        num_stages = cfg.MODEL.NUM_STAGES
        
        up_mode = cfg.MODEL.UPSAMPLE_MODE
        se_block = cfg.MODEL.SE_MODULE
        inception_block = cfg.MODEL.INCEPTION_MODULE

        if cfg.MODEL.INIT_UNET == True:
            self.init_block = nn.ModuleList([ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(base_filter, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        UNetBlock(base_filter=base_filter, activation=activation, norm=norm)])
        else:
            self.init_block = nn.ModuleList([ConvBlock(3, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, base_filter, 1, 1, 0, activation=activation, norm=norm)])

        # self.feat0 = ConvBlock(3, feat, 3, 1, 1, activation=activation, norm=norm)
        # self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation=activation, norm=norm)

        layers = []
        for i in range(1, num_stages):
            if i == 1:
                layers.append(PBPNBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block))
            else:
                layers.append(D_PBPNBlock(base_filter, kernel, stride, padding, i, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block))
        layers.append(DL_PBPNBlock(base_filter, kernel, stride, padding, num_stages, activation=activation, norm=norm, up_mode=up_mode, inception=inception_block))

        self.pbpn = nn.ModuleList(layers)

        # if cfg.MODEL.COMPRESS == True:
        #     recon_input_channel = base_filter
        #     self.compress = ConvBlock(base_filter*cfg.MODEL.NUM_STAGES, recon_input_channel, 1, 1, 0, activation=activation, norm=norm)
        # else:
        #     recon_input_channel = base_filter*cfg.MODEL.NUM_STAGES
        #     self.compress = None
        
        recon_input_channel = base_filter*cfg.MODEL.NUM_STAGES
        self.compress = None

        self.reconstructor16 = ConvBlock(recon_input_channel, 3, 3, 1, 1, activation=None, norm=None)
        self.reconstructor4 = None

        self.reset_parameters()

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # self.feat0.apply(weights_init)
        # self.feat1.apply(weights_init)
        self.init_block.apply(weights_init)
        self.pbpn.apply(weights_init)
        self.reconstructor16.apply(weights_init)
        if self.reconstructor4 != None:
            self.reconstructor4.apply(weights_init)
        if self.compress != None:
            self.compress.apply(weights_init)

    def forward(self, x, num_expand=2, only_last=None):
        for layer in self.init_block:
            x = layer(x)

        # x = self.feat0(x)
        # x = self.feat1(x)

        for layer in self.pbpn:
            x = layer(x, num_expand)

        if self.compress != None:
            x[num_expand] = self.compress(x[num_expand])
            if self.reconstructor4 != None:
                x[num_expand-1] = self.compress(x[num_expand-1])

        if self.reconstructor4 != None:
            sr_images4 = self.reconstructor4(x[num_expand-1])
            sr_images16 = self.reconstructor16(x[num_expand])
            return sr_images4, sr_images16
        else:
            sr_images16 = self.reconstructor16(x[num_expand])
        
        if only_last == True:
            return sr_images16
        else:
            return [sr_images16]

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(fix_model_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage)))



