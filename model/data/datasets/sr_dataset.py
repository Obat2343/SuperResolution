import os
import numpy as np
from PIL import Image
import torch.utils.data as data


class TrainDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, edge=None):
        super(TrainDataset, self).__init__()
        print('set train dataset')
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.img_filenames.sort()
        self.transform = transform
        if edge == False:
            edge = None
        self.edge = edge

    def __getitem__(self, index):
        hr_img = self._load_img(index)
        filename = self._get_img_name(index)

        if self.edge == None:
            hr_img, lr_img, filename = self.transform(hr_img, filename=filename)
            return (lr_img, hr_img, filename)
        else:
            edge_hr_img = self._load_edge_img(index)
            hr_img, lr_img, edge_hr_img, edge_lr_img, filename = self.transform(hr_img,None,edge_hr_img,None, filename)
            return (lr_img, hr_img, filename, edge_lr_img, edge_hr_img)


    def _load_img(self, img_id):
        return np.array(Image.open(self.img_filenames[img_id]).convert('RGB'))

    def _get_img_name(self, img_id):
        return os.path.basename(self.img_filenames[img_id])

    def _load_edge_img(self, img_id):
        img_file_path = self.img_filenames[img_id]
        edge_file_path = img_file_path.replace('train','train_edge')
        return np.array(Image.open(edge_file_path).convert('L'))

    def __len__(self):
        return len(self.img_filenames)

class MiniTrainDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, edge=None):
        super(MiniTrainDataset, self).__init__()
        print('set mini train dataset')
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.img_filenames.sort()
        self.transform = transform
        if edge == False:
            edge = None
        self.edge = edge

    def __getitem__(self, index):
        hr_img = self._load_img(index)
        filename = self._get_img_name(index)

        if self.edge == None:
            hr_img, lr_img, filename = self.transform(hr_img, filename=filename)
            return (lr_img, hr_img, filename)
        else:
            edge_hr_img = self._load_edge_img(index)
            hr_img, lr_img, edge_hr_img, edge_lr_img, _ = self.transform(hr_img,None,edge_hr_img,None)
            return (lr_img, hr_img, filename, edge_lr_img, edge_hr_img)

    def _load_img(self, img_id):
        return np.array(Image.open(self.img_filenames[img_id]).convert('RGB'))

    def _get_img_name(self, img_id):
        return os.path.basename(self.img_filenames[img_id])

    def _load_edge_img(self, img_id):
        img_file_path = self.img_filenames[img_id]
        edge_file_path = img_file_path.replace('train','train_edge')
        return np.array(Image.open(edge_file_path).convert('L'))

    def __len__(self):
        return len(self.img_filenames)


class ValDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, edge=None):
        super(ValDataset, self).__init__()
        self.img_filenames = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.transform = transform

    def __getitem__(self, index):
        lr_img = self._load_img(index)
        lr_img = self.transform(lr_img)
        filename = self._get_img_name(index)

        return (lr_img, filename)

    def _load_img(self, img_id):
        return np.array(Image.open(self.img_filenames[img_id]).convert('RGB'))

    def _get_img_name(self, img_id):
        return os.path.basename(self.img_filenames[img_id])

    def __len__(self):
        return len(self.img_filenames)