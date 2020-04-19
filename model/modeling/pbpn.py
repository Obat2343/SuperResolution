import torch
import torch.nn as nn

from .base_networks import ConvBlock, UNetBlock
from .pbpn_module import PBPNBlock, PBPNBlock_Custom, D_PBPNBlock, D_PBPNBlock_Custom, DL_PBPNBlock, DL_PBPNBlock_Custom

class PBPN_Custom(nn.Module):
    def __init__(self, cfg):
        super(PBPN_Custom, self).__init__()

        # check size
        if cfg.SCALE_FACTOR != cfg.GEN.SCALE_FACTOR ** cfg.GEN.NUM_UPSCALE:
            raise ValueError('please define correct scale: cfg.SCALE_FACTOR != cfg.GEN.SCALE_FACTOR ** cfg.GEN.NUM_UPSCALE')
        
        # define kernel size and etc.
        if cfg.GEN.SCALE_FACTOR == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif cfg.GEN.SCALE_FACTOR == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif cfg.GEN.SCALE_FACTOR ==8:
            kernel = 12
            stride = 8
            padding = 2

        # define filter size and num of stages
        feat = cfg.GEN.FEAT   #default 256
        base_filter = cfg.GEN.BASE_FILTER     #default 64
        num_stages = cfg.GEN.NUM_STAGES
        up_mode = cfg.GEN.UPSAMPLE

        # define activation and norm
        activation = cfg.GEN.ACTIVATION
        norm = cfg.GEN.NORM
        if norm == 'none':
            norm = None
        
        # define option
        se_block = cfg.GEN.OPTION.SE
        inception_block = cfg.GEN.OPTION.INCEPTION
        reconstruction = cfg.GEN.OPTION.RECONSTRUCTION

        if cfg.GEN.OPTION.RESIZE_RESIDUAL != 'none':
            self.resize = torch.nn.Upsample(scale_factor=cfg.SCALE_FACTOR, mode=cfg.GEN.OPTION.RESIZE_METHOD) # TODO
        else:
            self.resize = None

        # define init module
        if cfg.GEN.OPTION.INIT_MODULE == 'unet':
            self.init_block = nn.ModuleList([ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(base_filter, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        UNetBlock(base_filter=base_filter, activation=activation, norm=norm)])
        else:
            self.init_block = nn.ModuleList([ConvBlock(3, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, base_filter, 1, 1, 0, activation=activation, norm=norm)])

        # define pbpn block
        layers = []
        for i in range(1, num_stages):
            if i == 1:
                layers.append(PBPNBlock_Custom(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block, reconstruction=reconstruction))
            else:
                layers.append(D_PBPNBlock_Custom(base_filter, kernel, stride, padding, i, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block, reconstruction=reconstruction))
        layers.append(DL_PBPNBlock_Custom(base_filter, kernel, stride, padding, num_stages, activation=activation, norm=norm, up_mode=up_mode, inception=inception_block))
        self.pbpn = nn.ModuleList(layers)

        # initialize all wight
        self.reset_parameters()

    def forward(self, x, num_expand=2, only_last=False):
        # get resized_img
        if self.resize != None:
            resized_img = self.resize(x)
        else:
            resized_img = None

        # apply init block
        for layer in self.init_block:
            x = layer(x)

        # apply pbpn block
        output_list = []
        for layer in self.pbpn:
            x, reconst_image = layer(x, num_expand, resized_img)
            if reconst_image != None:
                output_list.append(reconst_image)
        
        # output
        if only_last:
            return output_list[-1]
        else:
            return output_list

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path):
        self.load_state_dict(fix_model_state_dict(torch.load(model_path, map_location=lambda storage, loc:storage)))

    def reset_parameters(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.init_block.apply(weights_init)
        self.pbpn.apply(weights_init)

class PBPN(nn.Module):
    def __init__(self, cfg):
        super(PBPN, self).__init__()

        kernel = 8
        stride = 4
        padding = 2

        feat = 256
        base_filter = 64

        activation = cfg.GEN.ACTIVATION
        norm = cfg.GEN.NORM
        if norm == 'none':
            norm = None
        num_stages = cfg.GEN.NUM_STAGES
        
        up_mode = cfg.GEN.UPSAMPLE
        se_block = cfg.GEN.OPTION.SE
        inception_block = cfg.GEN.OPTION.INCEPTION

        if cfg.GEN.OPTION.INIT_MODULE == 'unet':
            self.init_block = nn.ModuleList([ConvBlock(3, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(base_filter, base_filter, 3, 1, 1, activation=activation, norm=norm),
                                        UNetBlock(base_filter=base_filter, activation=activation, norm=norm)])
        else:
            self.init_block = nn.ModuleList([ConvBlock(3, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, feat, 3, 1, 1, activation=activation, norm=norm),
                                        ConvBlock(feat, base_filter, 1, 1, 0, activation=activation, norm=norm)])

        layers = []
        for i in range(1, num_stages):
            if i == 1:
                layers.append(PBPNBlock(base_filter, kernel, stride, padding, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block))
            else:
                layers.append(D_PBPNBlock(base_filter, kernel, stride, padding, i, activation=activation, norm=norm, up_mode=up_mode, se_block=se_block, inception=inception_block))
        layers.append(DL_PBPNBlock(base_filter, kernel, stride, padding, num_stages, activation=activation, norm=norm, up_mode=up_mode, inception=inception_block))

        self.pbpn = nn.ModuleList(layers)
        
        recon_input_channel = base_filter*cfg.GEN.NUM_STAGES
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