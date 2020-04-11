import os
import numpy as np
from PIL import Image
import torch.utils.data as data


class DIV2KDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DIV2KDataset, self).__init__()
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.transform = transform

    def __getitem__(self, index):
        hr_img = self._load_img(index)
        lr_img, hr_img = self.transform(hr_img)

        return (lr_img, hr_img)

    def _load_img(self, img_id):
        return np.array(Image.open(self.img_filenames[img_id]).convert('RGB'))

    def __len__(self):
        return len(self.img_filenames)
