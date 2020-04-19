from .transforms import *


class TrainAugmentation:
    def __init__(self, cfg):
        if cfg.DATASETS.BLUR_AUG:
            self.transform = Compose([
                ConvertFromInts(),
                RandomCrop(cfg.PATCH_SIZE*cfg.SCALE_FACTOR,16,seed=cfg.SEED),
                Resize(cfg.SCALE_FACTOR),
                RandomFlip(seed=cfg.SEED),
                RandomRotate(seed=cfg.SEED),
                GaussianBlur(kernel_range=(0, 5)),
                Normalize(),
                ToTensor(),
            ])
        else:
            self.transform = Compose([
                ConvertFromInts(),
                RandomCrop(cfg.PATCH_SIZE*cfg.SCALE_FACTOR,16,seed=cfg.SEED),
                Resize(cfg.SCALE_FACTOR),
                RandomFlip(seed=cfg.SEED),
                RandomRotate(seed=cfg.SEED),
                Normalize(),
                ToTensor(),
            ])

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) == type(None):
            hr_img, lr_img, filename = self.transform(hr_img, lr_img, filename=filename)
            return hr_img, lr_img, filename
        else:
            hr_img, lr_img, edge_hr_img, edge_lr_img, filename = self.transform(hr_img, lr_img, edge_hr_img, edge_lr_img, filename)
            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename

class MiniTrainAugmentation:
    def __init__(self, cfg):
        self.transform = Compose([
            ConvertFromInts(),
            CenterCrop(crop_size=512),
            Resize(cfg.SCALE_FACTOR),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) == type(None):
            hr_img, lr_img, filename = self.transform(hr_img, lr_img, filename=filename)
            return hr_img, lr_img, filename
        else:
            hr_img, lr_img, edge_hr_img, edge_lr_img, _ = self.transform(hr_img, lr_img, edge_hr_img, edge_lr_img)
            return hr_img, lr_img, edge_hr_img, edge_lr_img, _

class EvaluateAugmentation:
    def __init__(self, cfg):
        self.transform = Compose([
            ConvertFromInts(),
            CenterCrop(),
            # ShapingCrop(cfg.SCALE_FACTOR),
            # Resize(cfg.SCALE_FACTOR, eval=True),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, lr_img):
        lr_img, _, _ = self.transform(lr_img)
        return lr_img

class EvaluateAugmentationFull:
    def __init__(self, cfg):
        self.transform = Compose([
            ConvertFromInts(),
            # ShapingCrop(cfg.SCALE_FACTOR),
            # Resize(cfg.SCALE_FACTOR, eval=True),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, lr_img):
        lr_img, _, _ = self.transform(lr_img)
        return lr_img

class VisualizeAugmentation:
    def __init__(self, cfg):
        self.transform = Compose([
            ConvertFromInts(),
            # ShapingCrop(cfg.SCALE_FACTOR),
            CenterCrop(crop_size=512),
            Resize(cfg.SCALE_FACTOR, eval=True),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        hr_img, lr_img, filename = self.transform(hr_img, lr_img, filename=filename)
        return hr_img, lr_img, filename

class VisualizeAugmentationFull:
    def __init__(self, cfg):
        self.transform = Compose([
            ConvertFromInts(),
            ShapingCrop(16),
            # CenterCrop(crop_size=512),
            Resize(cfg.SCALE_FACTOR, eval=True),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        hr_img, lr_img, filename = self.transform(hr_img, lr_img, filename=filename)
        return hr_img, lr_img, filename

