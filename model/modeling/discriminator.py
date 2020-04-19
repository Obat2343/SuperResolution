import torch
import torch.nn as nn
import numpy as np
from .base_networks import ConvBlock, DenseBlock, Flatten, DeconvBlock, PSBlock
from torchvision.transforms import RandomCrop


class Discriminator(nn.Module):
    # receptive field : 61 * 61 px
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(Discriminator, self).__init__()
        self.image_size = cfg.BASIC.PATCH_SIZE * cfg.BASIC.SCALE_FACTOR
        
        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None
        
        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation=activation, norm =norm), # 1/2
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation=activation, norm =norm), # 1/4
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation=activation, norm =norm), # 1/8
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation=activation, norm =norm), # 1/16
        )
        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 8 * self.image_size // 16 * self.image_size // 16, base_filter * 16, activation=activation, norm=norm),
            DenseBlock(base_filter * 16, 1, activation=None, norm=None)
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()


    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out
        

class LargeDiscriminator(nn.Module):
    # receptive field : 253 * 253 px
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(LargeDiscriminator, self).__init__()
        self.image_size = cfg.BASIC.PATCH_SIZE * cfg.BASIC.SCALE_FACTOR
        
        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation=activation, norm =norm),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 32, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 32, base_filter * 32, 3, 2, 1, activation=activation, norm =norm),
        )
        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 32 * self.image_size // 64 * self.image_size // 64, base_filter * 16, activation=activation,norm=norm),
            DenseBlock(base_filter * 16, 1, activation=None, norm=None)
        )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()


    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out

class Discriminator128(nn.Module):
    # receptive field : 128 * 128 px
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(Discriminator128, self).__init__()
        self.image_size = cfg.BASIC.PATCH_SIZE * cfg.BASIC.SCALE_FACTOR
        np.random.seed(seed=cfg.SEED)
        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation=activation, norm =norm), # 1/2
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation=activation, norm =norm), # 1/4
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation=activation, norm =norm), # 1/8
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation=activation, norm =norm), # 1/16
            ConvBlock(base_filter * 8, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/32
        )

        self.dense_layers = nn.Sequential(
                ConvBlock(base_filter * 16, base_filter, 1, 1, 0, activation=activation, norm =norm),
                ConvBlock(base_filter,1, 4, 1, 0, activation=None, norm = None)
        )

        self.flat = Flatten()
        self.crop = RandomCrop(128)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def crop_images(self,x,size):
        B,C,H,W = x.shape
        max_left = int(W) - size
        max_top = int(H) - size
        left = np.random.randint(max_left)
        top = np.random.randint(max_top)
        return x[:,:,top:top+size,left:left+size]

    def forward(self, x):
        if (x.shape[2] != 128) or (x.shape[3] != 128):
            x = self.crop_images(x,128)
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = self.dense_layers(out)
        out = self.flat(out)
        return out

class Discriminator256(nn.Module):
    # receptive field : 256 * 256 px
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(Discriminator256, self).__init__()
        self.image_size = cfg.BASIC.PATCH_SIZE * cfg.BASIC.SCALE_FACTOR
        np.random.seed(seed=cfg.SEED)
        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation=activation, norm =norm), # 1/2
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation=activation, norm =norm), # 1/4
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation=activation, norm =norm), # 1/8
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation=activation, norm =norm), # 1/16
            ConvBlock(base_filter * 8, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/32
            ConvBlock(base_filter * 16, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/64
        )

        self.dense_layers = nn.Sequential(
                ConvBlock(base_filter * 16, base_filter, 1, 1, 0, activation=activation, norm =norm),
                ConvBlock(base_filter,1, 4, 1, 0, activation=None, norm = None)
        )

        self.flat = Flatten()

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()

    def crop_images(self,x,size):
        B,C,H,W = x.shape
        max_left = int(W) - size
        max_top = int(H) - size
        left = np.random.randint(max_left)
        top = np.random.randint(max_top)
        return x[:,:,top:top+size,left:left+size]

    def forward(self, x):
        if (x.shape[2] != 256) or (x.shape[3] != 256):
            x = self.crop_images(x,256)
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = self.dense_layers(out)
        out = self.flat(out)
        return out

class Discriminator512(nn.Module):
    # receptive field : 512 * 512 px
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(Discriminator512, self).__init__()
        self.image_size = cfg.BASIC.PATCH_SIZE * cfg.BASIC.SCALE_FACTOR
        
        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation=activation, norm =norm), # 1/2
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation=activation, norm =norm), # 1/4
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation=activation, norm =norm), # 1/8
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation=activation, norm =norm), # 1/16
            ConvBlock(base_filter * 8, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/32
            ConvBlock(base_filter * 16, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/64
            ConvBlock(base_filter * 16, base_filter * 16, 3, 1, 1, activation=activation, norm =norm),
            ConvBlock(base_filter * 16, base_filter * 16, 3, 2, 1, activation=activation, norm =norm), # 1/128
        )

        self.dense_layers = nn.Sequential(
                ConvBlock(base_filter * 16, base_filter, 1, 1, 0, activation=activation, norm =norm),
                ConvBlock(base_filter,1, 4, 1, 0, activation=None, norm = None)
        )

        self.flat = Flatten()

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()


    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = self.dense_layers(out)
        out = self.flat(out)
        return out


class UNetDiscriminator(nn.Module):
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(UNetDiscriminator, self).__init__()

        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.init_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*8, 3, 1, 1, activation=activation, norm=norm),
        ])
        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(base_filter*8, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*4, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
        ])


        if cfg.LOSS.GANLOSS_FN == 'MSE':
            self.out_conv = nn.ModuleList([
                ConvBlock(base_filter*8, 1, 1, 1, 0, activation='sigmoid', norm=None),
                ConvBlock(base_filter*1, 1, 1, 1, 0, activation='sigmoid', norm=None),
            ])
        else:
            self.out_conv = nn.ModuleList([
                ConvBlock(base_filter*8, 1, 1, 1, 0, activation=None, norm=None),
                ConvBlock(base_filter*1, 1, 1, 1, 0, activation=None, norm=None),
            ])

    def forward(self, x):
        x = self.init_conv(x)
        
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        bottleneck_feature = self.out_conv[0](x)
           
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)

        x = self.out_conv[1](x)
        
        return [x, bottleneck_feature]

        

class AllUNetDiscriminator(nn.Module):
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(AllUNetDiscriminator, self).__init__()

        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.init_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*8, 3, 1, 1, activation=activation, norm=norm),
        ])
        self.deconv_blocks = nn.ModuleList([
            DeconvBlock(base_filter*8, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*4, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*2, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            DeconvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
        ])


        self.out_conv = nn.ModuleList([
            ConvBlock(base_filter*8, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*4, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*4, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*2, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*2, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*1, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*1, 1, 1, 1, 0, activation=None, norm=None),
        ])

    def forward(self, x):
        x = self.init_conv(x)
        
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        predictions = []
        predictions.append(self.out_conv[0](x))
        j = 1
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
            else:
                predictions.append(self.out_conv[j](x))
                j += 1
        
        return predictions

class AllUNetDiscriminatorPixelshuffle(nn.Module):
    def __init__(self, cfg, num_channels=3, base_filter=64):
        super(AllUNetDiscriminatorPixelshuffle, self).__init__()

        activation = cfg.DIS.ACTIVATION
        norm = cfg.DIS.NORM
        if norm == 'none':
            norm = None

        self.init_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=activation, norm=norm)
        self.conv_blocks = nn.ModuleList([
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*1, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*1, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*2, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*4, 2, 2, 0, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*8, 3, 1, 1, activation=activation, norm=norm),
        ])
        self.deconv_blocks = nn.ModuleList([
            PSBlock(base_filter*8, base_filter*4, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            PSBlock(base_filter*4, base_filter*4, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*8, base_filter*4, 3, 1, 1, activation=activation, norm=norm),
            PSBlock(base_filter*4, base_filter*2, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            PSBlock(base_filter*2, base_filter*2, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*4, base_filter*2, 3, 1, 1, activation=activation, norm=norm),
            PSBlock(base_filter*2, base_filter*1, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
            PSBlock(base_filter*1, base_filter*1, 2, 3, 1, 1, activation=activation, norm=norm),
            ConvBlock(base_filter*2, base_filter*1, 3, 1, 1, activation=activation, norm=norm),
        ])


        self.out_conv = nn.ModuleList([
            ConvBlock(base_filter*8, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*4, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*4, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*2, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*2, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*1, 1, 1, 1, 0, activation=None, norm=None),
            ConvBlock(base_filter*1, 1, 1, 1, 0, activation=None, norm=None),
        ])

    def forward(self, x):
        x = self.init_conv(x)
        
        sources = []
        for i in range(len(self.conv_blocks)):
            if i % 2 == 0 and i != len(self.conv_blocks)-1 :
                sources.append(x)
            x = self.conv_blocks[i](x)

        predictions = []
        predictions.append(self.out_conv[0](x))
        j = 1
        for i in range(len(self.deconv_blocks)):
            x = self.deconv_blocks[i](x)
            if i % 2 == 0 and len(sources) != 0:
                x = torch.cat((x, sources.pop(-1)), 1)
            else:
                predictions.append(self.out_conv[j](x))
                j += 1
        
        return predictions