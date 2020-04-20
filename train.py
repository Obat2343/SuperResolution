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
    # build model
    print('Building model...')
    device = torch.device(cfg.DEVICE)
    model = build_model(cfg, args.pretrain).to(device)

    print('--------------------Generator architecture--------------------')
    print(model.sr_model)
    print('------------------Discriminator architecture------------------')
    print(model.discriminators)

    if args.sync_batchnorm:
        model = convert_model(model).to(device)

    # build train dataset
    print('Loading datasets...')
    train_transform = TrainAugmentation(cfg)
    train_dataset = build_dataset(dataset_list=cfg.DATASETS.TRAIN, transform=train_transform, edge=cfg.DATASETS.EDGE)
    sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.BATCH_SIZE, drop_last=True)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=cfg.NUM_WORKERS, batch_sampler=batch_sampler, pin_memory=True)

    # build evaluation dataset
    eval_transform =  EvaluateAugmentation(cfg)
    eval_dataset = build_dataset(dataset_list=cfg.DATASETS.EVAL, transform=eval_transform, edge=cfg.DATASETS.EDGE)
    eval_loader = DataLoader(eval_dataset, num_workers=cfg.NUM_WORKERS)

    # build minitrain dataset
    minitrain_transform = MiniTrainAugmentation(cfg)
    minitrain_dataset = build_dataset(dataset_list=cfg.DATASETS.VIS, transform=minitrain_transform, edge=False)
    minitrain_loader = DataLoader(minitrain_dataset, num_workers=cfg.NUM_WORKERS)

    # build optimizer and scheduler for generator(sr_model)
    if cfg.GEN.SOLVER.METHOD == 'adam':
        optimizer = torch.optim.Adam(model.sr_model.parameters(), lr=cfg.GEN.SOLVER.LR, betas=(cfg.GEN.SOLVER.BETA1, cfg.GEN.SOLVER.BETA2))
    elif cfg.GEN.SOLVER.METHOD == 'sgd':
        optimizer = torch.optim.SGD(model.sr_model.parameters(), lr=cfg.GEN.SOLVER.LR, momentum=0.6)
    else:
        raise ValueError('optimizer error: choose adam or sgd')
    
    if cfg.GEN.SCHEDULER.STEP > 0:
        g_scheduler = StepLR(optimizer, step_size=cfg.GEN.SCHEDULER.STEP * cfg.GEN.SOLVER.TRAIN_RATIO, gamma=cfg.GEN.SCHEDULER.GAMMA) # step size = gen_step * gen_dis ratio
    else:
        g_scheduler = None

    # build optimizer and scheduler for discriminator
    if not args.pretrain:
        if cfg.DIS.SOLVER.METHOD == 'adam':
            d_optimizer = torch.optim.Adam(model.discriminators.parameters(), lr=cfg.DIS.SOLVER.LR, betas=(cfg.DIS.SOLVER.BETA1, cfg.DIS.SOLVER.BETA2))
        elif cfg.DIS.SOLVER.METHOD == 'sgd':
            d_optimizer = torch.optim.SGD(model.discriminators.parameters(), lr=cfg.DIS.SOLVER.LR, momentum=0.6)
        else:
            raise ValueError('d_optimizer error: choose adam or sgd')
    
        if cfg.DIS.SCHEDULER.STEP > 0:
            d_scheduler = StepLR(d_optimizer, step_size=cfg.DIS.SCHEDULER.STEP, gamma=cfg.DIS.SCHEDULER.GAMMA)
        else:
            d_scheduler = None

    # convert model and optimizer to mixed precision model
    if args.mixed_precision:
        model.sr_model, optimizer = apex.amp.initialize(model.sr_model, optimizer, opt_level='O1')
        if not args.pretrain:
            model.discriminators, d_optimizer = apex.amp.initialize(model.discriminators, d_optimizer, opt_level='O1')

    # set tensorboard
    if args.tensorboard:
        summary_writer = tensorboardX.SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        summary_writer = None

    ### Start Training ###
    if args.pretrain:
        do_pretrain(args, cfg, model, optimizer, train_loader, eval_loader, device, summary_writer, g_scheduler)
        sys.exit()
    
    do_train(args, cfg, model, optimizer, d_optimizer, train_loader, eval_loader, minitrain_loader, device, summary_writer, g_scheduler, d_scheduler)

def main():
    parser = argparse.ArgumentParser(description='Perceptual Extreme Super-Resolution for NTIRE2020')
    parser.add_argument('-c','--config_file', type=str, default='', metavar='FILE', help='path to config file')
    parser.add_argument('-m','--mixed_precision', type=str2bool, default=False, help='')
    parser.add_argument('-t','--tensorboard', type=str2bool, default=True)
    parser.add_argument('-nw','--num_workers', type=int, default=24, help='')
    parser.add_argument('-ls','--log_step', type=int, default=50, help='')
    parser.add_argument('-ss','--save_step', type=int, default=1000, help='')
    parser.add_argument('-es','--eval_step', type=int, default=1000, help='')
    parser.add_argument('-o','--output_dirname', type=str, default='', help='')
    parser.add_argument('-r','--resume_iter', type=int, default=0, help='')
    parser.add_argument('-p','--pretrain', type=str2bool, default=False, help='')
    parser.add_argument('-s','--sync_batchnorm', type=str2bool, default=False, help='')
    parser.add_argument('-l','--load_model_path', type=str, default='', help='')

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
