from yacs.config import CfgNode as CN

_C = CN()

#########################################################################

# BASIC

#########################################################################

_C.SCALE_FACTOR = 16
_C.PATCH_SIZE = 32
_C.NUM_GPUS = 4
_C.NUM_WORKERS = 24
_C.BATCH_SIZE = 4
_C.MAX_ITER = 1000000

_C.OUTPUT_DIR = 'output'
_C.PRETRAIN_MODEL = 'weights/base_model.pth'
_C.PRETRAIN_D_MODEL = ''
_C.SEED = 123
_C.DEVICE = 'cuda'

#########################################################################

# GENERATOR

#########################################################################

_C.GEN = CN()

# name of generator
_C.GEN.NAME = 'dbpn-ll' # pbpn, pbpn-custom, dbpn-custom, dbpn-ll, dbpn

# generator basic option
_C.GEN.ACTIVATION = 'prelu' # relu, prelu, lrelu, tanh, sigmoid
_C.GEN.NORM = 'none' # batch, instance, group, spectral
_C.GEN.UPSAMPLE = 'pixelshuffle' # pixelshuffle, deconv, rconv

# generator structure option
_C.GEN.SCALE_FACTOR = 4
_C.GEN.NUM_UPSCALE = 2      # size = lr_size * (scale_factor ^ num_upscale)
_C.GEN.NUM_STAGES = 10
_C.GEN.FEAT = 256
_C.GEN.BASE_FILTER = 64

# generator additional module option
_C.GEN.OPTION = CN()
_C.GEN.OPTION.SE = False
_C.GEN.OPTION.INCEPTION = False
_C.GEN.OPTION.RECONSTRUCTION = False
_C.GEN.OPTION.INIT_MODULE = 'unet'      #normal, unet
_C.GEN.OPTION.RESIZE_RESIDUAL = 'none'    # 'none', 'all', 'last'
_C.GEN.OPTION.RESIZE_METHOD = 'bicubic'   # 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'

# generator solver
_C.GEN.SOLVER = CN()
_C.GEN.SOLVER.METHOD = 'adam'     # adam, sgd
_C.GEN.SOLVER.LR = 1e-4
_C.GEN.SOLVER.BETA1 = 0.5
_C.GEN.SOLVER.BETA2 = 0.9

# generator scheduler
_C.GEN.SCHEDULER = CN()
_C.GEN.SCHEDULER.STEP = 10000
_C.GEN.SCHEDULER.GAMMA = 0.1

# generator option for gan
_C.GEN.SOLVER.TRAIN_RATIO = 1
_C.GEN.TRAIN_START = 0

#########################################################################

# DISCRIMINATOR

#########################################################################

_C.DIS = CN()

# name of discriminator
_C.DIS.NAME = ('UNet',)

# discriminator basic option
_C.DIS.ACTIVATION = 'lrelu'     # relu, prelu, lrelu, tanh, sigmoid
_C.DIS.NORM = 'spectral'        # batch, instance, group, spectral

# discriminator solver
_C.DIS.SOLVER = CN()
_C.DIS.SOLVER.METHOD = 'adam'   # adam, sgd
_C.DIS.SOLVER.LR = 1e-4
_C.DIS.SOLVER.BETA1 = 0.9
_C.DIS.SOLVER.BETA2 = 0.999
_C.DIS.SOLVER.TRAIN_RATIO = 1

# discriminator scheduler
_C.DIS.SCHEDULER = CN()
_C.DIS.SCHEDULER.STEP = 10000
_C.DIS.SCHEDULER.GAMMA = 0.1

#########################################################################

# LOSS

#########################################################################

_C.LOSS = CN()

# weight of loss
_C.LOSS.RECONLOSS_WEIGHT = 1.0
_C.LOSS.VGGLOSS_WEIGHT = 1.0
_C.LOSS.GANLOSS_WEIGHT = 0.001
_C.LOSS.STYLELOSS_WEIGHT = 0.001
_C.LOSS.VARIATION_REG_WEIGHT = 0.001
_C.LOSS.GRADIENT_PENALTY_WEIGHT = 0

# wight option
_C.LOSS.GANLOSS_FN = 'BCE'      # BCE, MSE, RelativeLoss, HingeLoss, RelativeAverageLoss, RelativeHingeLoss, RelativeHingeLossFix, WassersteinLoss
_C.LOSS.GEN_L1 = True
_C.LOSS.RECONSTRUCTION_LOSS = True
_C.LOSS.VGG_LOSS = False
_C.LOSS.GEN_VGG = 'before'      # TODO
_C.LOSS.STYLE_LOSS = False
_C.LOSS.VARIATION_REGURALIZATION = False

#########################################################################

# DATASET

#########################################################################

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ('div8k_train',)
_C.DATASETS.EVAL = ('div8k_minival',)
_C.DATASETS.TEST = ('div8k_test',)
_C.DATASETS.VIS = ('div8k_minitrain',)
_C.DATASETS.EDGE = False
_C.DATASETS.BLUR_AUG = False