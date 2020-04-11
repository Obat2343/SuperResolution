import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from model.utils.misc import _sigmoid


# class Generator_Loss(nn.Module):
#     def __init__(self, args, cfg, device, Discriminator_Loss):
#         super(Generator_Loss, self).__init__()
#         self.real_label = torch.ones(cfg.SOLVER.BATCH_SIZE, 1).to(device)
#         self.device = device

#         if cfg.SOLVER.GEN_L1:
#             self.recon_loss_fn = self.L1
#             # self.recon_loss_fn = nn.L1Loss().to(device)
#         else:
#             self.recon_loss_fn = None

#         if cfg.SOLVER.GEN_VGG != 'none':
#             self.vgg = VGGLoss(args, cfg, device)
#         else:
#             self.vgg = None

#         self.gan_loss = Discriminator_Loss
#         self.w1 = cfg.SOLVER.GANLOSS_WEIGHT

#     def forward(self, sr_images, hr_images, fake_prediction, real_prediction, skip_recon=False):
#         if (skip_recon == False) and (self.recon_loss_fn != None):
#             recon_losses = self.recon_loss_fn(sr_images, hr_images)
#         else:
#             recon_losses = [torch.tensor(0,).to(self.device)] * len(sr_images)

#         if (skip_recon == False) and (self.vgg != None):
#             vgg_loss = self.vgg(sr_images, hr_images)
#         else:
#             vgg_loss = torch.tensor(0,).to(self.device)

#         real_loss, fake_loss = self.gan_loss(real_prediction, fake_prediction)
#         gan_loss = (real_loss + fake_loss) / 2

#         return recon_losses, vgg_loss, gan_loss
    
#     def L1(self, hr, sr):
#         return torch.mean(torch.abs(hr.view(hr.size(0), -1) - sr.view(sr.size(0), -1)), -1)

#     def MSE(self, hr, sr):
#         return torch.mean(torch.pow(hr.view(hr.size(0), -1) - sr.view(sr.size(0), -1),2), -1)

# class Discriminator_Loss(nn.Module):
#     def __init__(self, args, cfg, device):
#         super(Discriminator_Loss, self).__init__()

#         self.real_label = torch.ones(cfg.SOLVER.BATCH_SIZE, 1).to(device)
#         self.fake_label = torch.zeros(cfg.SOLVER.BATCH_SIZE, 1).to(device)

#         self.BCE = nn.BCEWithLogitsLoss().to(device)
#         self.MSE = nn.MSELoss().to(device)

#         if cfg.SOLVER.GANLOSS_FN == "BCE":
#             print('Loss is BCE')
#             self.gan_loss_fn = self.Standard_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "MSE":
#             print('Loss is MSE')
#             self.gan_loss_fn = self.Standard_MSE_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "RelativeLoss":
#             print('Loss is RelativeLoss')
#             self.gan_loss_fn = self.Relative_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "RelativeAverageLoss":
#             print('Loss is RelativeAverageLoss')
#             self.gan_loss_fn = self.Relative_Average_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "RelativeHingeLoss":
#             print('Loss is RelativeHingeLoss')
#             self.gan_loss_fn = self.Relative_Hinge_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "RelativeAverageHingeLoss":
#             print('Loss is RelativeAverageHingeLoss')
#             self.gan_loss_fn = self.Relative_Average_Hinge_Loss
#         elif cfg.SOLVER.GANLOSS_FN == "FocalLoss":
#             self.Focal = FocalLoss(cfg.SOLVER.FOCAL_GAMMA)
#             self.gan_loss_fn = self.Focal_Loss
    
#     def forward(self, fake_prediction, real_prediction):
#         return self.gan_loss_fn(fake_prediction, real_prediction)
        
#     def Standard_Loss(self, fake_prediction, real_prediction):
#         real_loss = self.BCE(real_prediction, self.real_label)
#         fake_loss = self.BCE(fake_prediction, self.fake_label)
#         return real_loss, fake_loss
    
#     def Standard_MSE_Loss(self, fake_prediction, real_prediction):
#         real_loss = self.MSE(real_prediction, self.real_label)
#         fake_loss = self.MSE(fake_prediction, self.fake_label)
#         return real_loss, fake_loss
    
#     def Relative_Loss(self, fake_prediction, real_prediction):
#         loss = self.BCE(real_prediction - fake_prediction, self.real_label)
#         return loss, loss # return same loss just for coding problem

#     def Relative_Average_Loss(self, fake_prediction, real_prediction):
#         real_loss = self.BCE(real_prediction - torch.mean(fake_prediction), self.real_label)
#         fake_loss = self.BCE(fake_prediction - torch.mean(real_prediction), self.fake_label)
#         return real_loss, fake_loss

#     def Relative_Hinge_Loss(self, fake_prediction, real_prediction):
#         real_loss = torch.mean(torch.nn.ReLU()(1.0 - (real_prediction - fake_prediction)))
#         fake_loss = torch.mean(torch.nn.ReLU()(1.0 + (fake_prediction - real_prediction)))
#         return real_loss, fake_loss
    
#     def Relative_Average_Hinge_Loss(self, fake_prediction, real_prediction):
#         real_loss = torch.mean(torch.nn.ReLU()(1.0 - (real_prediction - torch.mean(fake_prediction))))
#         fake_loss = torch.mean(torch.nn.ReLU()(1.0 + (fake_prediction - torch.mean(real_prediction))))
#         return real_loss, fake_loss
    
#     def Focal_Loss(self, fake_prediction, real_prediction):
#         real_loss = self.Focal(real_prediction, self.real_label)
#         fake_loss = self.Focal(fake_prediction, self.fake_label)
#         return real_loss, fake_loss


class VariationRegularizationLoss(nn.Module):
    def __init__(self):
        super(VariationRegularizationLoss, self).__init__()

    def forward(self, y):
        reg_loss = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

        return reg_loss


class VGGloss(nn.Module):
    def __init__(self):
        super(VGGloss, self).__init__()
        
        self.feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True), [9,18,27,36])
        self.mse_loss_fn = nn.MSELoss()

    def forward(self,fake_images, real_images):
        real_features = self.feature_extractor(real_images)
        fake_features = self.feature_extractor(fake_images)

        return sum([self.mse_loss_fn(fake_features[i], real_features[i].detach()) for i in range(len(real_features))])


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

        self.feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True), [9,18,27,36])
        self.mse_loss_fn = nn.MSELoss()

    
    def forward(self, fake_images, real_images):
        real_features = self.feature_extractor(real_images)
        fake_features = self.feature_extractor(fake_images)

        return sum([self.mse_loss_fn(self.gram_matrix(fake_features[i]), self.gram_matrix(real_features[i]).detach()) for i in range(len(real_features))])

    
    def gram_matrix(self, x):
        a, b, c, d = x.size()

        features = x.view(a * b, c * d)
        G = torch.mm(features, features.t())

        return G.div(a * b * c * d)


class FeatureExtractor(nn.Module):
    def __init__(self, netVGG, feature_layer):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children()))
        self.feature_layer = feature_layer

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in self.feature_layer:
                results.append(x)
        return results


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, fake_prediction, real_prediction):
        real_label = torch.ones(real_prediction.shape).to(real_prediction.device)
        fake_label = torch.zeros(fake_prediction.shape).to(fake_prediction.device)

        real_loss = self.loss_fn(real_prediction, real_label)  
        fake_loss = self.loss_fn(fake_prediction, fake_label)

        return fake_loss, real_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, fake_prediction, real_prediction):
        real_label = torch.ones(real_prediction.shape).to(real_prediction.device)
        fake_label = torch.zeros(fake_prediction.shape).to(fake_prediction.device)

        real_loss = self.loss_fn(real_prediction, real_label)  
        fake_loss = self.loss_fn(fake_prediction, fake_label)

        return fake_loss, real_loss


class RelativeLoss(nn.Module):
    def __init__(self):
        super(RelativeLoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, fake_predictions, real_predictions):
        if type(fake_predictions) != type(list):
            fake_predictions = [fake_predictions]
            real_predictions = [real_predictions]

        fake_loss, real_loss = 0, 0
        for fake_prediction, real_prediction in zip(fake_predictions, real_predictions):
            real_label = torch.ones(real_prediction.shape).to(real_prediction.device)
            fake_label = torch.zeros(fake_prediction.shape).to(fake_prediction.device)

            real_loss += self.loss_fn(_sigmoid(real_prediction - fake_prediction), real_label)
            fake_loss += self.loss_fn(_sigmoid(fake_prediction - real_prediction), fake_label)

        return fake_loss, real_loss


class RelativeAverageLoss(nn.Module):
    def __init__(self):
        super(RelativeAverageLoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, fake_predictions, real_predictions):
        if type(fake_predictions) != type(list):
            fake_predictions = [fake_predictions]
            real_predictions = [real_predictions]

        fake_loss, real_loss = 0, 0
        for fake_prediction, real_prediction in zip(fake_predictions, real_predictions):
            real_label = torch.ones(real_prediction.shape).to(real_prediction.device)
            fake_label = torch.zeros(fake_prediction.shape).to(fake_prediction.device)

            real_loss += self.loss_fn(_sigmoid(real_prediction - torch.mean(fake_prediction, dim=0)), real_label)       
            fake_loss += self.loss_fn(_sigmoid(fake_prediction - torch.mean(real_prediction, dim=0)), fake_label)

        return fake_loss, real_loss


class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, fake_predictions, real_predictions):
        real_loss = -torch.mean(real_predictions)
        fake_loss = torch.mean(fake_predictions)

        return fake_loss, real_loss


class HingeLoss(nn.Module):
    def __init__(self, g_train=False):
        super(HingeLoss, self).__init__()
        self.relu = nn.ReLU()
        self.g_train = g_train

    def forward(self, fake_predictions, real_predictions):
        if self.g_train:
            real_loss = torch.mean(-real_predictions)
            fake_loss = torch.mean(fake_predictions)
        else:
            real_loss = torch.mean(self.relu(1.0 - real_predictions))
            fake_loss = torch.mean(self.relu(1.0 + fake_predictions))

        return real_loss, fake_loss


class RelativeHingeLossFix(nn.Module):
    def __init__(self, g_train=False):
        super(RelativeHingeLossFix, self).__init__()
        self.relu = nn.ReLU()
        self.g_train = g_train

    def forward(self, fake_predictions, real_predictions):
        if self.g_train:
            real_loss = torch.mean(fake_predictions - real_predictions)
            fake_loss = torch.mean(fake_predictions - real_predictions)

        else:
            real_loss = torch.mean(self.relu(1.0 - (real_predictions - fake_predictions)))
            fake_loss = torch.mean(self.relu(1.0 + (fake_predictions - real_predictions)))

        return real_loss, fake_loss


class RelativeHingeLoss(nn.Module):
    def __init__(self):
        super(RelativeHingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, fake_predictions, real_predictions):
        real_loss = torch.mean(self.relu(1.0 - (real_predictions - fake_predictions)))
        fake_loss = torch.mean(self.relu(1.0 + (fake_predictions - real_predictions)))

        return real_loss, fake_loss


class RelativeAverageHingeLoss(nn.Module):
    def __init__(self):
        super(RelativeAverageHingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, fake_predictions, real_predictions):
        real_loss = torch.mean(self.relu(1.0 - (real_predictions - torch.mean(fake_predictions, dim=0))))
        fake_loss = torch.mean(self.relu(1.0 + (fake_predictions - torch.mean(real_predictions, dim=0))))

        return real_loss, fake_loss


class Generator_Loss(nn.Module):
    def __init__(self, args, cfg):
        super(Generator_Loss, self).__init__()

        if cfg.SOLVER.RECONSTRUCTION_LOSS:
            self.recon_loss_fn = nn.L1Loss()
        else:
            self.recon_loss_fn = None

        if cfg.SOLVER.VGG_LOSS:
            self.vgg_loss_fn = VGGloss()
        else:
            self.vgg_loss_fn = None

        if cfg.SOLVER.STYLE_LOSS:
            self.style_loss_fn = StyleLoss()
        else:
            self.style_loss_fn = None

        if cfg.SOLVER.VARIATION_REGURALIZATION:
            self.variation_reg_fn = VariationRegularizationLoss()
        else:
            self.variation_reg_fn = None
            
        if cfg.SOLVER.GANLOSS_FN == 'BCE':
            self.gan_loss_fn = BCELoss()
        elif cfg.SOLVER.GANLOSS_FN == 'MSE':
            self.gan_loss_fn = MSELoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeLoss':
            self.gan_loss_fn = RelativeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeAverageLoss':
            self.gan_loss_fn = RelativeAverageLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'HingeLoss':
            self.gan_loss_fn = HingeLoss(g_train=True)
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeHingeLoss':
            self.gan_loss_fn = RelativeHingeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeAverageHingeLoss':
            self.gan_loss_fn = RelativeAverageHingeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeHingeLossFix':
            self.gan_loss_fn = RelativeHingeLossFix(g_train=True)
        elif cfg.SOLVER.GANLOSS_FN == 'WassersteinLoss':
            self.gan_loss_fn = WassersteinLoss()
        else:
            self.gan_loss_fn = None    

        self.recon_loss_weight = cfg.SOLVER.RECONLOSS_WEIGHT
        self.vgg_loss_weight = cfg.SOLVER.VGGLOSS_WEIGHT
        self.gan_loss_weight = cfg.SOLVER.GANLOSS_WEIGHT
        self.style_loss_weight = cfg.SOLVER.STYLELOSS_WEIGHT
        self.variation_reg_weight = cfg.SOLVER.VARIATION_REG_WEIGHT


    def forward(self, sr_images, hr_images, fake_predictions, real_predictions):
        total_loss = 0
        loss_dict = {}
        if self.recon_loss_fn is not None:
            recon_loss = self.recon_loss_fn(sr_images, hr_images)
            total_loss += self.recon_loss_weight * recon_loss
            loss_dict['recon_loss'] = recon_loss.item()

        if self.vgg_loss_fn is not None:
            vgg_loss = self.vgg_loss_fn(sr_images, hr_images)
            total_loss += self.vgg_loss_weight * vgg_loss
            loss_dict['vgg_loss'] = vgg_loss.item()

        if self.style_loss_fn is not None:
            style_loss = self.style_loss_fn(sr_images, hr_images)
            total_loss += self.style_loss_weight * style_loss
            loss_dict['style_loss'] = style_loss.item()
            # print(style_loss)

        if self.variation_reg_fn is not None:
            reg_loss = self.variation_reg_fn(sr_images)
            total_loss += self.variation_reg_weight * reg_loss
            loss_dict['variation_reg'] = reg_loss.item()

        if self.gan_loss_fn is not None:
            gan_loss = 0
            gan_loss_stats = []
            for fake_prediction, real_prediction in zip(fake_predictions[0], real_predictions[0]):
                fake_loss, real_loss = self.gan_loss_fn(real_prediction, fake_prediction)
                gan_loss += real_loss / len(real_predictions)
                gan_loss_stats.append(real_loss/len(real_predictions))
            total_loss += self.gan_loss_weight * gan_loss
            loss_dict['gan_loss'] = gan_loss_stats

        # return total_loss
        return total_loss, loss_dict
        # return recon_loss + self.w1*gan_loss
        # return self.w1*gan_loss


class Discriminator_Loss(nn.Module):
    def __init__(self, args, cfg):
        super(Discriminator_Loss, self).__init__()
        if cfg.SOLVER.GANLOSS_FN == 'BCE':
            self.gan_loss_fn = BCELoss()
        elif cfg.SOLVER.GANLOSS_FN == 'MSE':
            self.gan_loss_fn = MSELoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeLoss':
            self.gan_loss_fn = RelativeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'HingeLoss':
            self.gan_loss_fn = HingeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeAverageLoss':
            self.gan_loss_fn = RelativeAverageLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeHingeLoss':
            self.gan_loss_fn = RelativeHingeLoss()
        elif cfg.SOLVER.GANLOSS_FN == 'RelativeHingeLossFix':
            self.gan_loss_fn = RelativeHingeLossFix()
        elif cfg.SOLVER.GANLOSS_FN == 'WassersteinLoss':
            self.gan_loss_fn = WassersteinLoss()

    def forward(self, fake_predictions, real_predictions):
        total_fake_loss, total_real_loss = 0, 0
        for fake_prediction, real_prediction in zip(fake_predictions, real_predictions):
            fake_loss, real_loss = self.gan_loss_fn(fake_prediction, real_prediction)
            total_fake_loss += fake_loss / len(fake_predictions)
            total_real_loss += real_loss / len(real_predictions)

        # return self.gan_loss_fn(predictions, self.label)

        return total_real_loss, total_fake_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class L1_move_Loss(nn.Module):
    def __init__(self, move=0):
        super(L1_move_Loss, self).__init__()
        self.move = move

    def forward(self, hr, sr):
        L1_min = self.L1(hr, sr)
        if self.move == 0:
            return sum(L1_min) / len(L1_min)
        elif self.move > 0:
            # move[0,1]
            temp_sr = sr[:,:,:,1:]
            temp_hr = hr[:,:,:,:-1]
            L1_min = torch.min(L1_min, self.L1(temp_hr,temp_sr))

            # move[1,0]
            temp_sr = sr[:,:,1:,:]
            temp_hr = hr[:,:,:-1,:]
            L1_min = torch.min(L1_min, self.L1(temp_hr,temp_sr))

            for i in range(1, self.move + 1):
                for j in range(1, self.move + 1):
                    temp_sr = sr[:,:,i:,j:]
                    temp_hr = hr[:,:,:-i,:-j]
                    L1_min = torch.min(L1_min, self.L1(temp_hr,temp_sr))
            return sum(L1_min) / len(L1_min)
    
    def L1(self, hr, sr):
        return torch.mean(torch.abs(hr.reshape(hr.size(0), -1) - sr.reshape(sr.size(0), -1)), -1)


