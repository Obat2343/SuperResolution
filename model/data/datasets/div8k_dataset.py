import os
import numpy as np
from PIL import Image
import torch.utils.data as data


class SRDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, edge=None):
        super(SRDataset, self).__init__()
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.transform = transform
        self.edge = edge
        self.img_filenames.sort()

        if self.edge != None:
            self.edge_filenames = [os.path.join(data_dir, x) for x in os.listdir(edge)]
            self.edge_filenames.sort()

    def __getitem__(self, index):
        hr_img = self._load_img(index)
        lr_img, hr_img = self.transform(hr_img)

        return (lr_img, hr_img)

    def _load_img(self, img_id):
        return np.array(Image.open(self.img_filenames[img_id]).convert('RGB'))

    def __len__(self):
        return len(self.img_filenames)
