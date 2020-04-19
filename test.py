import argparse
import os
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader

from model.modeling.pbpn import PBPN, PBPN_Re
from model.config import cfg
from model.data.transforms.data_preprocessing import EvaluateAugmentation, EvaluateAugmentationFull
from model.utils.misc import fix_model_state_dict, save_torch_img, str2bool
from model.data.datasets import build_dataset
from model.utils.misc import chop_forward_2 as chop_forward


def test(args, cfg):
    device = torch.device(cfg.MODEL.DEVICE)

    if cfg.MODEL.GEN.OPTOIN.FINAL_RESIDUAL == 'none':
        model = PBPN_Re(cfg).to(device)
    else:
        model = PBPN(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('the number of model parameters: {}'.format(total_params))

    if args.center_crop:
        test_transform = EvaluateAugmentation(cfg)
    else:
        test_transform = EvaluateAugmentationFull(cfg)

    test_dataset = build_dataset(dataset_list=cfg.DATASETS.TEST, transform=test_transform)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers)

    train_model_file = os.path.basename(args.trained_model)
    head, ext = os.path.splitext(train_model_file)
    output_dir = os.path.join(args.output_dirname,head)

    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc:storage)))
    print('trained model was loaded from {}'.format(args.trained_model))

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus)))

    print('Test Starts')
    if args.center_crop:
        save_dir = os.path.join('Results', output_dir)
    else:
        save_dir = os.path.join('Results', output_dir + '_FULL')

    model.eval()
    total_inference_time = 0
    for (lr_images, filename) in tqdm(test_loader):
        lr_images = lr_images.to(device)

        tic = time.time()
        with torch.no_grad():
            if args.small:
                sr_images = chop_forward(lr_images, model, cfg.MODEL.GEN.SCALE_FACTOR, device, shave=8, slice_size=16, nGPUs=args.num_gpus, ensemble=args.ensemble)
            else:
                sr_images = chop_forward(lr_images, model, cfg.MODEL.GEN.SCALE_FACTOR, device, nGPUs=args.num_gpus, ensemble=args.ensemble)
        toc = time.time()
        total_inference_time += toc - tic
        save_path = os.path.join(save_dir, filename[0])
        save_torch_img(sr_images, save_path, args.center_crop)
    

    print('Test Finished. Avarage inference time: {:.2f}s'.format((total_inference_time)/len(test_loader)))


def main():
    parser = argparse.ArgumentParser(description='Perceptual Extreme Super Resolution for NTIRE2020')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE')
    parser.add_argument('--output_dirname', type=str, default='test')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--trained_model', type=str, default='weights/iteration_400000.pth')
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--center_crop', type=str2bool, default=False)
    parser.add_argument('--ensemble', type=str2bool, default=False)
    parser.add_argument('--small', type=str2bool, default=False)

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