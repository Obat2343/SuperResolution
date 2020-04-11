import torch
import PIL
from PIL import Image
import numpy as np
import cv2
import os
import math
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            for t in self.transforms:
                hr_img, lr_img, filename = t(hr_img, lr_img, filename=filename)
            return hr_img, lr_img, filename
        else:
            for t in self.transforms:
                hr_img, lr_img, edge_hr_img, edge_lr_img, filename = t(hr_img, lr_img, edge_hr_img, edge_lr_img, filename)
            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename


class ConvertFromInts(object):
    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            return hr_img.astype(np.float32), lr_img, filename
        else:
            return hr_img.astype(np.float32), lr_img, edge_hr_img.astype(np.float32), edge_lr_img, filename


class ToTensor(object):
    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            if lr_img is None:
                return torch.from_numpy(hr_img.astype(np.float32)).permute(2, 0, 1), lr_img, filename
            else:
                return torch.from_numpy(hr_img.astype(np.float32)).permute(2, 0, 1), torch.from_numpy(lr_img.astype(np.float32)).permute(2, 0, 1), filename
        else:
            hr_img = torch.from_numpy(hr_img.astype(np.float32)).permute(2, 0, 1)
            edge_hr_img = torch.from_numpy(edge_hr_img.astype(np.float32))
            if lr_img is not None:
                lr_img = torch.from_numpy(lr_img.astype(np.float32)).permute(2, 0, 1)
                edge_lr_img = torch.from_numpy(edge_lr_img.astype(np.float32))
            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename


class RandomCrop(object):
    def __init__(self, size, max_diff=None, seed=123):
        self.size = size
        np.random.seed(seed=seed)
        if max_diff != None:
            self.edge_img_max_diff = max_diff
            print('#############################################')
            print('max_diff between edge and RGB is {}pix'.format(self.edge_img_max_diff))

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        assert lr_img == None, 'lr_img should be None'
        assert edge_lr_img == None, 'edge_lr_img should be None'
        height, width, _ = hr_img.shape

        max_left = width - self.size
        max_top = height - self.size
        left = np.random.randint(max_left)
        top = np.random.randint(max_top)

        if type(edge_hr_img) is type(None):
            return hr_img[top:top+self.size, left:left+self.size], lr_img, filename
        else:
            while True:
                add_left = np.random.randint(-self.edge_img_max_diff, self.edge_img_max_diff + 1)
                add_top = np.random.randint(-self.edge_img_max_diff, self.edge_img_max_diff + 1)
                edge_top = top + add_top
                edge_left = left + add_left            
                if (edge_top <= max_top) and (edge_top >= 0) and (edge_left <= max_left) and (edge_left >= 0):
                    return hr_img[top:top+self.size, left:left+self.size], lr_img, edge_hr_img[edge_top:edge_top+self.size, edge_left:edge_left+self.size], edge_lr_img, filename
            

class ShapingCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        assert lr_img == None, 'lr_img should be None'

        height, width, _ = hr_img.shape
        # print('before: {} {}'.format(height, width))
        height = math.floor(height/self.size) * self.size
        width = math.floor(width/self.size) * self.size
        # print('after: {} {}'.format(height, width))
        if type(edge_hr_img) is type(None):
            return hr_img[:height, :width], lr_img, filename
        else:
            return hr_img[:height, :width], lr_img, edge_hr_img[:height, :width], edge_lr_img, filename


class RandomFlip(object):
    def __init__(self,seed=123):
        np.random.seed(seed=seed)

    def __call__(self, hr_img, lr_img, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            if np.random.randint(2):
                hr_img = hr_img[:, ::-1]
                lr_img = lr_img[:, ::-1]

            return hr_img, lr_img, filename
        else:
            if np.random.randint(2):
                hr_img = hr_img[:, ::-1]
                lr_img = lr_img[:, ::-1]
                edge_hr_img = edge_hr_img[:, ::-1]
                edge_lr_img = edge_lr_img[:, ::-1]

            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename

    
class RandomRotate(object):
    def __init__(self,seed=123):
        np.random.seed(seed=seed)
        
    def __call__(self, hr_img, lr_img, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            angle = np.random.randint(4)
            return np.rot90(hr_img, k=angle), np.rot90(lr_img, k=angle), filename
        else:
            angle = np.random.randint(4)
            return np.rot90(hr_img, k=angle), np.rot90(lr_img, k=angle), np.rot90(edge_hr_img, k=angle), np.rot90(edge_lr_img, k=angle), filename


class Resize(object):
    def __init__(self, factor, eval=False):
        self.factor = factor
        # self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.interp_methods = [PIL.Image.BICUBIC, PIL.Image.BILINEAR, PIL.Image.LANCZOS]
        self.eval = eval

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        assert lr_img == None, 'lr_img should be None'
        assert edge_lr_img == None, 'edge_lr_img should be None'

        height, width, _ = hr_img.shape
        size = (int(width/self.factor), int(height/self.factor))
        if self.eval:
            interp_method = self.interp_methods[2]
        else:
            interp_method = self.interp_methods[np.random.randint(3)]
            # interp_method = self.interp_methods[2]

        hr_img_pil = Image.fromarray(hr_img.astype(np.uint8))
        lr_img_pil = hr_img_pil.resize(size, interp_method)
        lr_img = np.array(lr_img_pil, dtype=np.float32)
        # lr_img = cv2.resize(hr_img, size, interpolation=interp_method)

        if type(edge_hr_img) is type(None):
            return hr_img, lr_img, filename
        else:
            edge_hr_img_pil = Image.fromarray(edge_hr_img.astype(np.uint8))
            edge_lr_img_pil = edge_hr_img_pil.resize(size, interp_method)
            edge_lr_img = np.array(edge_lr_img_pil, dtype=np.float32)
            # edge_lr_img = cv2.resize(edge_hr_img, size, interpolation=interp_method)
            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename


class Normalize(object):
    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if type(edge_hr_img) is type(None):
            if lr_img is None:
                return hr_img/255, lr_img, filename
            else:
                return hr_img/255, lr_img/255, filename
        else:
            if lr_img is None:
                return hr_img/255, lr_img, edge_hr_img/255, edge_lr_img, filename
            else:
                return hr_img/255, lr_img/255, edge_hr_img/255, edge_lr_img/255, filename


class CenterCrop(object):
    def __init__(self, crop_size=64):
        self.crop_size = crop_size

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        height, width, _ = hr_img.shape
        crop_area = [(height - self.crop_size)/2, (height + self.crop_size)/2, (width - self.crop_size)/2, (width + self.crop_size)/2]
        crop_area = [int(i) for i in crop_area]
        
        if type(edge_hr_img) is type(None):
            return hr_img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], lr_img, filename
        else:
            return hr_img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], lr_img, edge_hr_img[crop_area[0]:crop_area[1], crop_area[2]:crop_area[3]], edge_lr_img, filename

class Save_image(object):
    def __init__(self,save_path,skip=True):
        self.skip = skip

        if self.skip == False:
            self.save_path = os.path.join(save_path,'save_image')
            os.makedirs(self.save_path, exist_ok=True)
    
    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        if self.skip == False:
            filename = os.path.basename(filename)
            head, ext = os.path.splitext(filename)
            save_hr_image_path = os.path.join(self.save_path, head + '_hr' + ext)
            save_lr_image_path = os.path.join(self.save_path, head + '_lr' + ext)
            cv2.imwrite(save_hr_image_path,hr_img[:,:,[2,1,0]])
            cv2.imwrite(save_lr_image_path,lr_img[:,:,[2,1,0]])
        
        if type(edge_hr_img) is type(None):
            return hr_img, lr_img, filename
        else:
            return hr_img, lr_img, edge_hr_img, edge_lr_img, filename


class NoisedResize(object):
    def __init__(self, factor, mean=0, std=10, eval=False):
        self.factor = factor
        self.interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        self.eval = eval

        self.aug = iaa.AdditiveGaussianNoise(scale=(mean, std))

    def __call__(self, hr_img, lr_img=None):
        assert lr_img == None, 'lr_img should be None'

        lr_img = copy.deepcopy(hr_img)
        lr_img = self.aug.augment_image(lr_img)

        height, width, _ = lr_img.shape
        size = (int(width/self.factor), int(height/self.factor))

        if self.eval:
            interp_method = self.interp_methods[0]
        else:
            interp_method = self.interp_methods[np.random.randint(5)]
        lr_img = cv2.resize(lr_img, size, interpolation=interp_method)[:, :, np.newaxis]

        return hr_img, lr_img


class GaussianBlur(object):
    def __init__(self, kernel_range):
        self.kernel_range = kernel_range

    def __call__(self, hr_img, lr_img=None, edge_hr_img=None, edge_lr_img=None, filename=None):
        kernel_size = 2*random.randint(self.kernel_range[0], self.kernel_range[1])+1
        if type(edge_hr_img) is type(None):
            if lr_img is not None:
                lr_img = cv2.GaussianBlur(lr_img, (kernel_size, kernel_size), 0)

                return hr_img, lr_img, filename

            else:
                hr_img = cv2.GaussianBlur(hr_img, (kernel_size, kernel_size), 0)

                return hr_img, lr_img, filename

        else:
            if lr_img is not None:
                lr_img = cv2.GaussianBlur(lr_img, (kernel_size, kernel_size), 0)

                return hr_img, lr_img, edge_hr_img/255, edge_lr_img, filename

            else:
                hr_img = cv2.GaussianBlur(hr_img, (kernel_size, kernel_size), 0)

                return hr_img, lr_img, edge_hr_img/255, edge_lr_img/255, filename

