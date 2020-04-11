import os
import sys
import PIL
from PIL import Image
import torch
import torchvision
import csv
from tqdm import tqdm
from model.utils.misc import save_heatmap_overlay, save_torch_img

def PSNR(hr_images, sr_images):
    hr_images, sr_images = hr_images*255, sr_images*255
    hr_images, sr_images = torch.clamp(hr_images, 0, 255), torch.clamp(sr_images, 0, 255)
    if hr_images.size(0) == 1:
        mse = torch.nn.functional.mse_loss(hr_images, sr_images)
        mse = torch.unsqueeze(mse, 0)
    elif hr_images.size(0) > 1:
        mse = torch.mean(torch.pow(hr_images.view(hr_images.size(0), -1) - sr_images.view(sr_images.size(0), -1),2), -1)
    psnr = 10 * torch.log10(255 * 255 / mse)
    return psnr


def compute_PSNR(hr_root, sr_root, save_dir):
    totensor = torchvision.transforms.ToTensor()
    hr_list = os.listdir(hr_root)
    hr_list.sort()
    sr_list = os.listdir(sr_root)
    sr_list.sort()
    result_list = []
    save_dir = os.path.abspath(save_dir)

    for hr_file_name in tqdm(hr_list):
        head, ext = os.path.splitext(hr_file_name)
        if ext in ['.png','.jpg','.JPG','.jpeg','.webp']:            
            hr_file_path = os.path.join(hr_root,hr_file_name)
            hr_image = Image.open(hr_file_path)
            hr_image = totensor(hr_image)
            hr_image = torch.unsqueeze(hr_image, 0)

            for sr_file_name in sr_list:
                if head == sr_file_name[:4]:
                    sr_file_path = os.path.join(sr_root, sr_file_name)
                    sr_image = Image.open(sr_file_path)
                    sr_image = totensor(sr_image)
                    sr_image = torch.unsqueeze(sr_image, 0)
                    psnr = PSNR(hr_image,sr_image)
                    psnr = psnr[0].item()
                    result_list.append([hr_file_name,psnr])

                    heatmap = torch.abs((hr_image - sr_image))
                    heatmap = heatmap[:, 0, :, :] + heatmap[:, 1, :, :] + heatmap[:, 2, :, :]
                    save_path = os.path.join(save_dir, 'heatmap_full', '{}_{}{}'.format(head,psnr,ext))
                    save_heatmap_overlay(sr_image, heatmap, save_path)

                    os.makedirs(os.path.join(save_dir,'sr_image_full'), exist_ok=True)
                    os.symlink(sr_file_path, os.path.join(save_dir,'sr_image_full','{}_{}{}'.format(head,psnr,ext)))
    
    mean_psnr = sum([x[1] for x in result_list]) / len(result_list)
    result_list.append(['mean', mean_psnr])

    with open(os.path.join(save_dir,'psnr.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result_list)

if __name__ == '__main__':
    save_path = os.path.join('Results/visualize', sys.argv[3])
    compute_PSNR(sys.argv[1], sys.argv[2], save_path)



