import itertools
import time
import datetime
import os

import torch
import torch.nn as nn

# import apex

from model.utils.misc import str2bool, check_mkdir, save_torch_img, fix_model_state_dict, chop_forward, debug_set, save_checkpoint, load_checkpoint
from .loss_functions import Generator_Loss, Discriminator_Loss


def do_train(args, cfg, model, optimizer, d_optimizer, train_loader, eval_loader, minitrain_loader, device, summary_writer, g_scheduler, d_scheduler):
    max_iter = len(train_loader)
    trained_time = 0
    discriminator_name_list = list(cfg.DIS.NAME)
    discriminator_name_list.sort()
    print(discriminator_name_list)

    logging_dict = {}
    logging_dict['/train_disc/D_loss'] = 0
    logging_dict['/train_gen/loss'] = 0
    logging_dict['/train_gen/recon_loss'] = 0
    logging_dict['/train_gen/vgg_loss'] = 0
    logging_dict['/train_gen/style_loss'] = 0
    logging_dict['/train_gen/variation_regralization'] = 0
    for discriminator_name in discriminator_name_list:
        logging_dict['{}/generate'.format(discriminator_name)] = 0
        logging_dict['{}/fake'.format(discriminator_name)] = 0
        logging_dict['{}/real'.format(discriminator_name)] = 0

    logging_dict['/train_gen/gan_loss'] = 0
    logging_dict['/train_disc/real_loss'] = 0
    logging_dict['/train_disc/fake_loss'] = 0

    tic = time.time()
    end = time.time()

    g_loss_fn = Generator_Loss(args, cfg).to(device)
    d_loss_fn = Discriminator_Loss(args, cfg).to(device)

    
    if args.resume_iter > 0:
        g_checkpoint_path = os.path.join(cfg.OUTPUT_DIR, 'generator', 'iteration_{}.pth'.format(args.resume_iter))
        d_checkpoint_path = os.path.join(cfg.OUTPUT_DIR, 'discriminator', 'iteration_{}.pth'.format(args.resume_iter))
        model.sr_model, optimizer, g_scheduler = load_checkpoint(model.sr_model, optimizer, g_scheduler, g_checkpoint_path)
        model.discriminators, d_optimizer, d_scheduler = load_checkpoint(model.sr_model, d_optimizer, d_scheduler, d_checkpoint_path)
    else:
        model.sr_model, _, _ = load_checkpoint(model.sr_model, optimizer, g_scheduler, args.load_model_path)

    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg.NUM_GPUS)))

    print('Training Starts')
    model.train()
    for iteration, data in enumerate(train_loader, args.resume_iter+1):
        if cfg.DATASETS.EDGE == True:
            lr_images, hr_images, filename, edge_lr_images, edge_hr_images = data
        else:
            lr_images, hr_images, filename = data
            edge_lr_images, edge_hr_images = None, None

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
      
        if type(d_scheduler) is not type(None):
            d_scheduler.step()
        if type(g_scheduler) is not type(None):
            g_scheduler.step()

        #### Train Discriminator ####
        for _ in range(cfg.DIS.SOLVER.TRAIN_RATIO):
            d_optimizer.zero_grad()
            
            real_losses, fake_losses, gradient_penalties = [], [], []
            sr_images, sr_prediction, hr_prediction = model(lr_images, hr_images, edge_hr_images)
            # print('prediction sr max:{} min:{}'.format(torch.min(sr_prediction),torch.max(sr_prediction)))

            for i in range(len(sr_prediction)):
                real_loss, fake_loss = d_loss_fn(sr_prediction[i], hr_prediction[i])
                real_losses.append(real_loss)
                fake_losses.append(fake_loss)

                if cfg.LOSS.GRADIENT_PENALTY_WEIGHT != 0:
                    gradient_penalty = model(sr_images, hr_images, grad_penalty=True).mean()
                    gradient_penalties.append(gradient_penalty)

            if cfg.LOSS.GRADIENT_PENALTY_WEIGHT != 0:
                mean_real_loss, mean_fake_loss, mean_gradient_penalty = sum(real_losses) / len(real_losses), sum(fake_losses) / len(fake_losses), sum(gradient_penalties) / len(gradient_penalties)
                d_loss = mean_real_loss + mean_fake_loss + cfg.DIS.SOLVER.GRADIENT_PENALTY_WEIGHT * mean_gradient_penalty
            else:
                mean_real_loss, mean_fake_loss = sum(real_losses) / len(real_losses), sum(fake_losses) / len(fake_losses)
                d_loss = mean_real_loss + mean_fake_loss

            # update discriminator parametor
            if iteration >= cfg.GEN.TRAIN_START:
                pass
            else:              
                if args.mixed_precision:
                    with apex.amp.scale_loss(d_loss, d_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    d_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                d_optimizer.step()

        # for tensorboard
        logging_dict['/train_disc/D_loss'] += d_loss.item()
        logging_dict['/train_disc/real_loss'] += mean_real_loss.item()
        logging_dict['/train_disc/fake_loss'] += mean_fake_loss.item()
        for i, discriminator_name in enumerate(discriminator_name_list):
            logging_dict['{}/fake'.format(discriminator_name)] += real_losses[i]
            logging_dict['{}/real'.format(discriminator_name)] += fake_losses[i]
            

        #### Train Generator ####
        for _ in range(cfg.GEN.SOLVER.TRAIN_RATIO):

            # skip until train generator step
            if iteration <= cfg.GEN.TRAIN_START:
                break
            
            optimizer.zero_grad()
            
            sr_images, sr_predictions_g, hr_predictions_g = model(lr_images, hr_images, edge_hr_images)
            loss, loss_dict = g_loss_fn(sr_images, hr_images, sr_predictions_g, hr_predictions_g)
            
            # update parametor
            if args.mixed_precision:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # for tensorboard
        if iteration > cfg.GEN.TRAIN_START:
            logging_dict['/train_gen/loss'] += loss.item()
            
            if 'recon_loss' in loss_dict.keys():
                logging_dict['/train_gen/recon_loss'] += loss_dict['recon_loss']
            if 'style_loss' in loss_dict.keys():
                logging_dict['/train_gen/style_loss'] += loss_dict['style_loss']
            if 'vgg_loss' in loss_dict.keys():
                logging_dict['/train_gen/vgg_loss'] += loss_dict['vgg_loss']
            if 'variation_reg' in loss_dict.keys():
                logging_dict['/train_gen/variation_regralization'] += loss_dict['variation_reg']
            logging_dict['/train_gen/gan_loss'] += sum(loss_dict['gan_loss']) / len(loss_dict['gan_loss'])
                
            for i, discriminator_name in enumerate(discriminator_name_list):
                logging_dict['{}/generate'.format(discriminator_name)] = loss_dict['gan_loss'][i]

        #### save and print log ####
        trained_time += time.time() - end
        end = time.time() 
        if iteration % args.log_step == 0:
            for key in logging_dict.keys():
                logging_dict[key] /= args.log_step
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}, D_Loss: {:.6f}'.format(iteration, 
                optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)),
                logging_dict['/train_gen/loss'], logging_dict['/train_disc/D_loss']))

            if args.tensorboard:
                for key in logging_dict.keys():
                    summary_writer.add_scalar(key, logging_dict[key], global_step=iteration)
                    logging_dict[key] = 0
                summary_writer.add_scalar('/train_setting/LR', optimizer.param_groups[0]['lr'], global_step=iteration)
            tic = time.time()

        if iteration % args.eval_step == 0:
            print('=====> Evaluating...')
            model.eval()

            eval_save_dir = os.path.join(cfg.OUTPUT_DIR,'eval',str(iteration))

            for (lr_images, filename) in eval_loader:
                lr_images = lr_images.to(device)
                with torch.no_grad():
                    sr_images = chop_forward(lr_images, model, cfg.SCALE_FACTOR, device, nGPUs=cfg.NUM_GPUS)
                eval_save_path = os.path.join(eval_save_dir, filename[0])
                save_torch_img(sr_images, eval_save_path, center_crop=True)

            print('=====> Evaluation Complete: Iter: {:06d}'.format(iteration))
            print('=====> Save SR Images to {}'.format(eval_save_dir))

            minitrain_save_dir = os.path.join(cfg.OUTPUT_DIR,'minitrain',str(iteration))
            
            for data in minitrain_loader:
                lr_images, hr_images, filename = data
                lr_images = lr_images.to(device)
                with torch.no_grad():
                    sr_images = chop_forward(lr_images, model, cfg.SCALE_FACTOR, device, nGPUs=cfg.NUM_GPUS)
                
                minitrain_save_path = os.path.join(minitrain_save_dir, filename[0])
                save_torch_img(sr_images, minitrain_save_path, center_crop=True)

            print('=====> Evaluation Complete: Iter: {:06d}'.format(iteration))
            print('=====> Save SR Images to {}'.format(minitrain_save_dir))

            model.train()

        # checkpoint
        if iteration % args.save_step == 0:
            generator_path = os.path.join(cfg.OUTPUT_DIR, 'generator', 'iteration_{}.pth'.format(iteration))
            discriminator_path = os.path.join(cfg.OUTPUT_DIR, 'discriminator', 'iteration_{}.pth'.format(iteration))
            
            check_mkdir(generator_path)
            check_mkdir(discriminator_path)

            if cfg.NUM_GPUS > 1:
                save_checkpoint(model.module.sr_model, optimizer, g_scheduler)
                save_checkpoint(model.module.discriminators, d_optimizer, d_scheduler)
            else:
                save_checkpoint(model.sr_model, optimizer, g_scheduler)
                save_checkpoint(model.discriminators, d_optimizer, d_scheduler)

            print('=====> Save checkpoint to {}'.format(generator_path))


def do_pretrain(args, cfg, model, optimizer, train_loader, eval_loader, device, summary_writer, g_scheduler):
    max_iter = len(train_loader)
    trained_time = 0
    logging_loss = 0
    logging_last_loss = 0
    tic = time.time()
    end = time.time()

    # load model from iter x
    if args.resume_iter > 0:
        if len(args.load_model_path) > 0:
            checkpoint_path =  args.load_model_path
        else:
            checkpoint_path =  os.path.join(cfg.OUTPUT_DIR, 'generator', 'iteration_{}.pth'.format(args.resume_iter))
        
        model.sr_model, optimizer, g_scheduler = load_checkpoint(model.sr_model, optimizer, g_scheduler, checkpoint_path)
    
    # convert model to multi gpu model
    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg.NUM_GPUS)))

    # define loss function
    loss_fn = nn.L1Loss().to(device)

    # start training
    print('Pre-Training Starts')
    for iteration, data in enumerate(train_loader, args.resume_iter+1):
        # get images
        if cfg.DATASETS.EDGE == True:
            lr_images, hr_images, filename, edge_lr_images, edge_hr_images = data
        else:
            lr_images, hr_images, filename = data
            edge_lr_images, edge_hr_images = None, None

        # convert images from cpu to device
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        # reset gradient
        optimizer.zero_grad()
        
        # compute loss
        loss_list = []
        sr_images_list = model(lr_images)
        for sr_images in sr_images_list:
            loss = loss_fn(sr_images, hr_images)
            loss_list.append(loss)
        loss = sum(loss_list) / len(loss_list)

        # compute gradient
        if args.mixed_precision:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # update model
        optimizer.step()
        
        # update lr
        if type(g_scheduler) is not type(None):
            g_scheduler.step()

        # logging
        trained_time += time.time() - end
        end = time.time() 
        logging_loss += loss.item()
        logging_last_loss += loss_list[-1].item()

        # print and save log
        if iteration % args.log_step == 0:
            logging_loss /= args.log_step
            logging_last_loss /= args.log_step
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_loss))

            # write loss to tensorboard
            if args.tensorboard:
                summary_writer.add_scalar('/pre_train/loss', logging_loss, global_step=iteration)
                summary_writer.add_scalar('/pre_train/loss_last_output', logging_last_loss, global_step=iteration)
                summary_writer.add_scalar('/pre_train/LR', optimizer.param_groups[0]['lr'], global_step=iteration)
            
            # reset logging loss
            logging_loss = 0
            logging_last_loss = 0
            tic = time.time()

        # save sr image
        if iteration % args.eval_step == 0:
            print('=====> Evaluating...')
            model.eval()
            eval_save_dir = os.path.join(cfg.OUTPUT_DIR,'eval',str(iteration))

            for (lr_images, filename) in eval_loader:
                lr_images = lr_images.to(device)

                with torch.no_grad():
                    sr_images_list = model(lr_images)
                
                # save sr image
                for i, sr_images in enumerate(sr_images_list):
                    save_path = os.path.join(eval_save_dir, str(i+1), filename[0])
                    save_torch_img(sr_images, save_path)

            print('=====> Evaluation Complete: Iter: {:06d}'.format(iteration))
            print('=====> Save SR Images to {}'.format(eval_save_dir))

            model.train()

        # checkpoint (save model)
        if iteration % args.save_step == 0:
            generator_path = os.path.join(cfg.OUTPUT_DIR, 'generator', 'iteration_{}.pth'.format(iteration))
            g_optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'g_optimizer', 'iteration_{}.pth'.format(iteration))
            
            check_mkdir(generator_path)
            check_mkdir(g_optimizer_path)

            if cfg.NUM_GPUS > 1:
                save_checkpoint(model.module.sr_model, optimizer, g_scheduler, generator_path)
            else:
                save_checkpoint(model.sr_model, optimizer, g_scheduler, generator_path)
            
            print('=====> Save checkpoint to {}'.format(generator_path))
