import PIL
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm


def make_lr(root_dir,target_dir,ratio=1/16):
    '''
    method list: [PIL.Image.BICUBIC, PIL.Image.BILINEAR, PIL.Image.LANCZOS]
    '''
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    file_list = os.listdir(root_dir)
    for filename in tqdm(file_list):
        filepath = os.path.join(root_dir,filename)
        head, ext = os.path.splitext(filepath)
        if ext in ['.png','.jpg','.JPG','.jpeg','.webp']:
            img = Image.open(filepath)
            img_resize = img.resize((int(img.width * ratio), int(img.height * ratio)),PIL.Image.BICUBIC)
            new_filepath = os.path.join(target_dir,filename)
            img_resize.save(new_filepath)

def make_hr(root_dir,target_dir,ratio=1/16):
    '''
    method list: [PIL.Image.BICUBIC, PIL.Image.BILINEAR, PIL.Image.LANCZOS]
    '''
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    file_list = os.listdir(root_dir)
    for filename in tqdm(file_list):
        filepath = os.path.join(root_dir,filename)
        head, ext = os.path.splitext(filepath)
        if ext in ['.png','.jpg','.JPG','.jpeg','.webp']:
            img = Image.open(filepath)
            img_resize = img.resize((int(img.width * ratio), int(img.height * ratio)),PIL.Image.BICUBIC)
            img_resize = img_resize.resize((img.width, img.height),PIL.Image.BICUBIC)
            new_filepath = os.path.join(target_dir,filename)
            img_resize.save(new_filepath)

if __name__ == '__main__':
    make_hr(sys.argv[1],sys.argv[2])