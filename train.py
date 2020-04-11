import argparse
import os
import time
import datetime
import sys
import resource
import shutil
import json
from statistics import mean

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import apex
import tensorboardX

from model.utils.misc import str2bool, check_mkdir, save_torch_img, fix_model_state_dict, chop_forward, debug_set, save_args
from model.config import cfg
from model.modeling.build_model import build_model
from model.data.transforms.data_preprocessing import TrainAugmentation, EvaluateAugmentation, MiniTrainAugmentation
from model.data.datasets import build_dataset
from model.data import samplers
from model.engine.loss_functions import *
from model.modeling.sync_batchnorm import convert_model
from model.engine.trainer import do_train, do_pretrain



def train(args, cfg):
    device = torch.device(cfg.MODEL.DEVICE)
    
    print('Building model...')
    model = build_model(cfg, args.pretrain).to(device)

    print('--------------------Generator architecture--------------------')
    print(model.sr_model)
    print('------------------Discriminator architecture------------------')
    print(model.discriminators)

    print('Loading datasets...')
    train_transform = TrainAugmentation(cfg)
    train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform, edge=cfg.MODEL.EDGE)
    sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=True)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    eval_transform =  EvaluateAugmentation(cfg)
    eval_dataset = build_dataset(dataset_list=cfg.DATASETS.EVAL, transform=eval_transform, edge=cfg.MODEL.EDGE)
    eval_loader = DataLoader(eval_dataset, num_workers=args.num_workers)

    minitrain_transform = MiniTrainAugmentation(cfg)
    minitrain_dataset = build_dataset(dataset_list=cfg.DATASETS.VIS, transform=minitrain_transform, edge=False)
    minitrain_loader = DataLoader(minitrain_dataset, num_workers=args.num_workers)

    if cfg.SOLVER.GENERATOR_OPTIM == 'adam':
        optimizer = torch.optim.Adam(model.sr_model.parameters(), lr=cfg.SOLVER.GENERATOR_LR, betas=(0.5, 0.9))
    elif cfg.SOLVER.GENERATOR_OPTIM == 'sgd':
        optimizer = torch.optim.SGD(model.sr_model.parameters(), lr=cfg.SOLVER.GENERATOR_LR, momentum=0.6)
    else:
        raise ValueError('optimizer error: choose adam or sgd')

    if not args.pretrain:
        if cfg.SOLVER.DISCRIMINATOR_OPTIM == 'adam':
            d_optimizer = torch.optim.Adam(model.discriminators.parameters(), lr=cfg.SOLVER.DISCRIMINATOR_LR, betas=(0.5, 0.9))
        elif cfg.SOLVER.DISCRIMINATOR_OPTIM == 'sgd':
            d_optimizer = torch.optim.SGD(model.discriminators.parameters(), lr=cfg.SOLVER.DISCRIMINATOR_LR, momentum=0.6)
        else:
            raise ValueError('d_optimizer error: choose adam or sgd')

    if args.sync_batchnorm:
        model = convert_model(model).to(device)
  
    if args.mixed_precision:
        model.sr_model, optimizer = apex.amp.initialize(model.sr_model, optimizer, opt_level='O1')
        if not args.pretrain:
            model.discriminators, d_optimizer = apex.amp.initialize(model.discriminators, d_optimizer, opt_level='O1')

    if args.tensorboard:
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None
    
    if cfg.SCHEDULER.GENERATOR:
        g_scheduler = StepLR(optimizer, step_size=cfg.SCHEDULER.GENERATOR_STEP * cfg.SOLVER.GEN_TRAIN_RATIO, gamma=cfg.SCHEDULER.GENERATOR_GAMMA) # step size = gen_step * gen_dis ratio
    else:
        g_scheduler = None
    
    if cfg.SCHEDULER.DISCRIMINATOR:
        d_scheduler = StepLR(d_optimizer, step_size=cfg.SCHEDULER.DISCRIMINATOR_STEP, gamma=cfg.SCHEDULER.DISCRIMINATOR_GAMMA)
    else:
        d_scheduler = None

    # print(model.sr_model)
    if not args.pretrain:
        print('pretrain model was loaded from {}'.format(cfg.PRETRAIN_MODEL))
        model.sr_model.load_state_dict(fix_model_state_dict(torch.load(cfg.PRETRAIN_MODEL, map_location=lambda storage, loc:storage)))
        if len(cfg.PRETRAIN_D_MODEL) > 0:
            print('pretrain discriminator model was loaded from {}'.format(cfg.PRETRAIN_D_MODEL))
            model.discriminators.load_state_dict(fix_model_state_dict(torch.load(cfg.PRETRAIN_D_MODEL, map_location=lambda storage, loc:storage)))

    ### Start Training
    if args.pretrain:
        do_pretrain(args, cfg, model, optimizer, train_loader, eval_loader, device, summary_writer, g_scheduler)
        sys.exit()
    
    do_train(args, cfg, model, optimizer, d_optimizer, train_loader, eval_loader, minitrain_loader, device, summary_writer, g_scheduler, d_scheduler)

def main():
    parser = argparse.ArgumentParser(description='Perceptual Extreme Super-Resolution for NTIRE2020')
    parser.add_argument('--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('--mixed_precision', type=str2bool, default=False, help='')
    parser.add_argument('--tensorboard', type=str2bool, default=True)
    parser.add_argument('--num_gpus', type=int, default=2, help='')
    parser.add_argument('--num_workers', type=int, default=24, help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--save_step', type=int, default=1000, help='')
    parser.add_argument('--eval_step', type=int, default=1000, help='')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--resume_iter', type=int, default=0, help='')
    parser.add_argument('--pretrain', type=str2bool, default=False, help='')
    parser.add_argument('--sync_batchnorm', type=str2bool, default=False, help='')

    args = parser.parse_args()

    # load configration file
    if len(args.config_file) > 0:
        print('Loaded configration file {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)

    # define output folder name for save log
    if len(args.output_dirname) == 0:
        dt_now = datetime.datetime.now()
        output_dirname = str(dt_now.date()) + '_' + str(dt_now.time())
    else:
        output_dirname = args.output_dirname
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_dirname)
    cfg.freeze()

    # save configration
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    if len(args.config_file) > 0:
        shutil.copy(args.config_file,cfg.OUTPUT_DIR)
    argsfile_path = os.path.join(cfg.OUTPUT_DIR, "args.txt")
    save_args(args,argsfile_path)

    # setting for cuda
    torch.manual_seed(cfg.SEED)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)

    print('Running with config:\n{}'.format(cfg))

    # train model
    train(args, cfg)
    

if __name__ == '__main__':
    main()
