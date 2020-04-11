import argparse
import os
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader

from model.modeling.pbpn import PBPN, PBPN_Re
from model.config import cfg
from model.data.transforms.data_preprocessing import VisualizeAugmentation, VisualizeAugmentationFull
from model.utils.misc import fix_model_state_dict, chop_forward, save_heatmap_overlay, str2bool, save_torch_img
from model.data.datasets import build_dataset

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


def test(args, cfg):
    train_model_file = os.path.basename(args.trained_model)
    head, ext = os.path.splitext(train_model_file)
    output_dir = os.path.join(args.output_dirname,head)

    device = torch.device(cfg.MODEL.DEVICE)

    if cfg.MODEL.BICUBIC_RESIDUAL:
        model = PBPN_Re(cfg).to(device)
    else:
        model = PBPN(cfg).to(device)

    if args.center_crop:
        vis_transform = VisualizeAugmentation(cfg)
    else:
        vis_transform = VisualizeAugmentationFull(cfg)

    if args.dataset == 'mini':
        vis_dataset = build_dataset(dataset_list=cfg.DATASETS.VIS, transform=vis_transform)
    elif args.dataset == 'train':
        vis_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=vis_transform)

    vis_loader = DataLoader(vis_dataset, num_workers=args.num_workers)

    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc:storage)))
    print('trained model was loaded from {}'.format(args.trained_model))

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    print('Visualize Starts')
    save_dir = os.path.join('Results', output_dir)
    model.eval()
    tic = time.time()
    psnr_list = []

    for (lr_images, hr_images, filename) in tqdm(vis_loader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        print('load image')
        with torch.no_grad():
            sr_images = chop_forward(lr_images, model, cfg.MODEL.SCALE_FACTOR, device, nGPUs=args.num_gpus, ensemble=args.ensemble, ensemble_re=False)
            print('finish making sr images')
            psnr = PSNR(hr_images, sr_images)
            psnr_list.append(psnr[0].cpu().item())
            print('finish calculate PSNR')

        heatmap = torch.abs((hr_images - sr_images))
        heatmap = heatmap[:, 0, :, :] + heatmap[:, 1, :, :] + heatmap[:, 2, :, :]
        
        filename = os.path.basename(filename[0])
        head, ext = os.path.splitext(filename)

        if args.center_crop:
            save_path = os.path.join(save_dir, 'heatmap', '{}_{}{}'.format(head,psnr[0],ext))
            save_heatmap_overlay(sr_images, heatmap, save_path)
            save_path = os.path.join(save_dir, 'sr_image', '{}_{}{}'.format(head,psnr[0],ext))
            save_torch_img(sr_images, save_path)
        else:
            save_path = os.path.join(save_dir, 'heatmap_full', '{}_{}{}'.format(head,psnr[0],ext))
            save_heatmap_overlay(sr_images, heatmap, save_path)
            save_path = os.path.join(save_dir, 'sr_image_full', '{}_{}{}'.format(head,psnr[0],ext))
            save_torch_img(sr_images, save_path)

    psnr_mean = sum(psnr_list) / len(psnr_list)
    print('PSNR: {}'.format(psnr_mean))
    toc = time.time()

    print('Test Finished. Avarage inference time: {:.2f}s'.format((toc-tic)/len(vis_loader)))


def main():
    parser = argparse.ArgumentParser(description='Perceptual Extreme Super Resolution for NTIRE2020')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')
    parser.add_argument('--output_dirname', type=str, default='visualize')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--trained_model', type=str, default='weights/iteration_400000.pth')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--center_crop', type=str2bool, default=True)
    parser.add_argument('--ensemble', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='mini')

    args = parser.parse_args()
    
    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    if len(args.config_file) > 0:
        print('Loaded configration file {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    cfg.freeze()

    test(args, cfg)

if __name__ == '__main__':
    main()