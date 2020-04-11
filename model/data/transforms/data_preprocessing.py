from .transforms import *


class TrainAugmentation:
    def __init__(self, cfg):
        if cfg.SOLVER.BLUR_AUG:
            self.transform = Compose([
                ConvertFromInts(),
                RandomCrop(cfg.INPUT.PATCH_SIZE*cfg.MODEL.SCALE_FACTOR,16,seed=cfg.SEED),
                Resize(cfg.MODEL.SCALE_FACTOR),
                RandomFlip(seed=cfg.SEED),
                RandomRotate(seed=cfg.SEED),
                GaussianBlur(kernel_range=(0, 5)),
                Save_image(cfg.OUTPUT_DIR,skip=cfg.DEBUG.SKIP_SAVE_IMAGE),
                Normalize(),
                ToTensor(),
            ])
        else:
            self.transform = Compose([
                ConvertFromInts(),
                RandomCrop(cfg.INPUT.PATCH_SIZE*cfg.MODEL.SCALE_FACTOR,16,seed=cfg.SEED),
                Resize(cfg.MODEL.SCALE_FACTOR),
                RandomFlip(seed=cfg.SEED),
                RandomRotate(seed=cfg.SEED),
                Save_image(cfg.OUTPUT_DIR,skip=cfg.DEBUG.SKIP_SAVE_IMAGE),
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
            Resize(cfg.MODEL.SCALE_FACTOR),
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
            # ShapingCrop(cfg.MODEL.SCALE_FACTOR),
            # Resize(cfg.MODEL.SCALE_FACTOR, eval=True),
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
            # ShapingCrop(cfg.MODEL.SCALE_FACTOR),
            # Resize(cfg.MODEL.SCALE_FACTOR, eval=True),
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
            # ShapingCrop(cfg.MODEL.SCALE_FACTOR),
            CenterCrop(crop_size=512),
            Resize(cfg.MODEL.SCALE_FACTOR, eval=True),
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
            Resize(cfg.MODEL.SCALE_FACTOR, eval=True),
            Normalize(),
            ToTensor(),
        ])

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        hr_img, lr_img, filename = self.transform(hr_img, lr_img, filename=filename)
        return hr_img, lr_img, filename

