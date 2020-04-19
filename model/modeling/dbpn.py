import torch
import torch.nn as nn

from .base_networks import ConvBlock, UpBlock, D_UpBlock, DownBlock, D_DownBlock
from .dbpn_module import UpBlock_DBPN, D_UpBlock_DBPN, DownBlock_DBPN, D_DownBlock_DBPN

class DBPN(nn.Module):
    def __init__(self, cfg):
        super(DBPN, self).__init__()
        
        num_channels = 3 # change here
        base_filter = 64 # change here
        feat = 256 # change here
        num_stages = 7
        scale_factor = 8

        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock_DBPN(base_filter, kernel, stride, padding)
        self.down1 = DownBlock_DBPN(base_filter, kernel, stride, padding)
        self.up2 = UpBlock_DBPN(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, only_last=None):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1),1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down3(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down4(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down5(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down6(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        x = self.output_conv(concat_h)
        
        return x

class DBPN_LL(nn.Module):
    def __init__(self, cfg):
        super(DBPN_LL, self).__init__()
        
        num_channels = 3 # change here
        base_filter = 64 # change here
        feat = 256 # change here
        num_stages = 10
        scale_factor = 8

        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock_DBPN(base_filter, kernel, stride, padding)
        self.down1 = DownBlock_DBPN(base_filter, kernel, stride, padding)
        self.up2 = UpBlock_DBPN(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 8)
        self.up9 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 8)
        self.down9 = D_DownBlock_DBPN(base_filter, kernel, stride, padding, 9)
        self.up10 = D_UpBlock_DBPN(base_filter, kernel, stride, padding, 9)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, only_last=None):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1),1)
        l = self.down2(concat_h)
        
        concat_l = torch.cat((l, l1),1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down3(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down4(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down5(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down6(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down7(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up8(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down8(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up9(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        l = self.down9(concat_h)
        
        concat_l = torch.cat((l, concat_l),1)
        h = self.up10(concat_l)
        
        concat_h = torch.cat((h, concat_h),1)
        x = self.output_conv(concat_h)
        return x

class DBPN_Custom(nn.Module):
    def __init__(self, cfg):
        super(DBPN_Custom, self).__init__()
        
        num_channels = 3 # change here
        base_filter = 64 # change here
        feat = 256 # change here
        num_stages = 5
        scale_factor = 8

        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        elif scale_factor == 16:
            raise ValueError('scale factor 16 is not available now')
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(3, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        
        up_layers = []
        down_layers = []

        for i in range(1, num_stages):
            if i == 1:
                up_layers.append(UpBlock(base_filter, kernel, stride, padding))
                down_layers.append(DownBlock(base_filter, kernel, stride, padding))
            else:
                up_layers.append(D_UpBlock(base_filter, kernel, stride, padding, i))
                down_layers.append(D_DownBlock(base_filter, kernel, stride, padding, i))
        
        self.up_layers = nn.ModuleList(up_layers)
        self.down_layers = nn.ModuleList(down_layers)

        self.last_up = D_UpBlock(base_filter, kernel, stride, padding, num_stages)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, only_last=None):
        x = self.feat0(x)
        x = self.feat1(x)
        concat_l = None
        concat_h = None
        
        for stage, (up, down) in enumerate(zip(self.up_layers,self.down_layers),1):
            if stage == 1:
                h1 = up(x)
                l = down(h1)
                concat_l = torch.cat((l, x),1)
            else:
                h = up(concat_l)
                if concat_h == None:
                    concat_h = torch.cat((h, h1),1)
                else:
                    concat_h = torch.cat((h, concat_h),1)
                
                l = down(concat_h)
                concat_l = torch.cat((l, concat_l),1)

        h = self.last_up(concat_l)
        concat_h = torch.cat((h, concat_h),1)

        x = self.output_conv(concat_h)
        
        return x