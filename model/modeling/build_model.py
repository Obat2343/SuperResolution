import torch
import torch.nn as nn
from torch.autograd import grad as torch_grad

from .pbpn import PBPN, PBPN_Custom
from .dbpn import DBPN, DBPN_LL, DBPN_Custom
from .discriminator import Discriminator, LargeDiscriminator, UNetDiscriminator, Discriminator128, Discriminator256, Discriminator512, AllUNetDiscriminator, AllUNetDiscriminatorPixelshuffle

class MODEL(nn.Module):
    def __init__(self, cfg, sr_model, discriminator=None):
        '''
        discriminator is modulelist for using multiple discriminators
        e.g.
        discriminator = [Discriminator128, Discriminator256, Discriminator512]
        '''
        super(MODEL, self).__init__()
        
        self.sr_model = sr_model
        self.discriminators = discriminator

    def forward(self, lr_images, hr_images=None, edge_hr_images=None, mixup=False, device=False, only_last=False, grad_penalty=False):
        if grad_penalty:
            return self.calc_gradient_penalty(lr_images, hr_images)
        
        sr_images = self.sr_model(lr_images, only_last=only_last)
        if edge_hr_images != None:
            edge_hr_images = torch.unsqueeze(edge_hr_images, 1)
            sr_images, hr_images = [torch.cat((sr_images[0], edge_hr_images), 1)], torch.cat((hr_images, edge_hr_images), 1)

        if self.discriminators is None or not self.training:
            return sr_images
        elif mixup == True:
            mix_predictions = []
            mix_images, labels = self.mix_data(sr_images, hr_images, device)
            for discriminator in self.discriminators:
                mix_predictions.append(discriminator(mix_images))
            return mix_images, mix_predictions, labels
        else:
            fake_predictions = []
            real_predictions = []
            for discriminator in self.discriminators:
                fake_predictions.append(discriminator(sr_images[-1]))
                real_predictions.append(discriminator(hr_images))
            # predictions = self.discriminator(torch.cat((sr_images, hr_images), 0))
            if edge_hr_images != None:
                sr_images = [sr_image[:,:3] for sr_image in sr_images]
            return sr_images[-1], fake_predictions, real_predictions
            # return sr_images, predictions

        

    def mix_data(self, sr_images, hr_images, device):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        batch_size = sr_images.size()[0]
        alphas = torch.rand(batch_size,1).to(device)
        mixed_image = torch.zeros(sr_images.size()).to(device)
        for i in range(len(alphas)):
            mixed_image[i] = (alphas[i] * hr_images[i]) + ((1 - alphas[i]) * sr_images[i])
        return mixed_image, alphas

    def debug_forward(self, lr_images, hr_images=None, mixup=False, device=False):
        sr_images = self.sr_model(lr_images)

        if mixup == True:
            mix_predictions = []
            mix_images, labels = self.mix_data(sr_images, hr_images, device)
            for discriminator in self.discriminators:
                mix_predictions.append(discriminator(mix_images))
            return mix_images, mix_predictions, labels
        else:
            fake_predictions = []
            real_predictions = []
            for discriminator in self.discriminators:
                fake_predictions.append(discriminator(sr_images))
                real_predictions.append(discriminator(hr_images))
            # predictions = self.discriminator(torch.cat((sr_images, hr_images), 0))
            return sr_images, fake_predictions, real_predictions
            # return sr_images, predictions

    def calc_gradient_penalty(self, sr_images, hr_images):
        device = hr_images.device
        batch_size = hr_images.size()[0]

        alpha = torch.rand(batch_size, 1 ,1 ,1)
        alpha = alpha.expand_as(hr_images).to(device)

        interpolated = alpha * hr_images.data + (1 - alpha) * sr_images.data
        interpolated = interpolated.to(device)
        interpolated.requires_grad = True

        gradients_norms = []
        for discriminator in self.discriminators:
            prob_interpolated = discriminator(interpolated)

            for prob in prob_interpolated:
                # print(prob)
                gradients = torch_grad(outputs=prob, inputs=interpolated,
                                        grad_outputs=torch.ones(prob.size()).to(device),
                                        create_graph=True, retain_graph=True)[0]

                gradients = gradients.view(batch_size, -1)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                gradients_norm = ((gradients_norm - 1) ** 2).mean()

                gradients_norms.append(gradients_norm)

        return sum(gradients_norms) / len(gradients_norms)


def build_model(cfg, pretrain=False):
    if cfg.GEN.NAME == 'pbpn':
        generator = PBPN(cfg)
    elif cfg.GEN.NAME == 'pbpn-custom':
        generator = PBPN_Custom(cfg)
    elif cfg.GEN.NAME == 'dbpn-custom':
        generator = DBPN_Custom(cfg)
    elif cfg.GEN.NAME == 'dbpn-ll':
        generator = DBPN_LL(cfg)
    elif cfg.GEN.NAME == 'dbpn':
        generator = DBPN(cfg)

    if pretrain:
        print('generator: {}'.format(cfg.GEN.NAME))
        return MODEL(cfg, generator)
    else:
        discriminator_name_list = list(cfg.DIS.NAME)
        discriminator_name_list.sort()
        discriminator_list = []
        print(discriminator_name_list)
        if cfg.DATASETS.EDGE == True:
            num_channels = 4
        else:
            num_channels = 3

        if 'UNet' in discriminator_name_list:
            discriminator_list.append(UNetDiscriminator(cfg, num_channels=num_channels))
        if 'small' in discriminator_name_list:
            discriminator_list.append(Discriminator(cfg, num_channels=num_channels))
        if 'large' in discriminator_name_list:
            discriminator_list.append(LargeDiscriminator(cfg, num_channels=num_channels))
        if 'size128' in discriminator_name_list:
            discriminator_list.append(Discriminator128(cfg, num_channels=num_channels))
        if 'size256' in discriminator_name_list:
            discriminator_list.append(Discriminator256(cfg, num_channels=num_channels))
        if 'size512' in discriminator_name_list:
            discriminator_list.append(Discriminator512(cfg, num_channels=num_channels))
        if 'AllUNet' in discriminator_name_list:
            discriminator_list.append(AllUNetDiscriminator(cfg, num_channels=num_channels))
        if 'AllUNetPix' in discriminator_name_list:
            discriminator_list.append(AllUNetDiscriminatorPixelshuffle(cfg, num_channels=num_channels))

        print('generator: {}'.format(cfg.GEN.NAME))
        print('discriminator list')
        print(discriminator_list)

        return MODEL(cfg, generator, discriminator=nn.ModuleList(discriminator_list))
