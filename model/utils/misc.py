import numpy as np
from PIL import Image
import os
from collections import OrderedDict
from functools import reduce
import torch
import cv2
import json
import collections


def str2bool(s):
    return s.lower() in ('true', '1')


def check_mkdir(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def save_args(args,file_path="args_data.json"):
    with open(file_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def save_torch_img(image, save_path, center_crop=False, heatmap=None):
    check_mkdir(save_path)
    image = image.cpu().squeeze(0).numpy().transpose((1, 2, 0))*255
    image = image.clip(0, 255).astype(np.uint8)

    if heatmap is not None:
        image = overlay_heatmap(image, heatmap)

    image = Image.fromarray(image)

    if center_crop:
        image = crop_center(image, 1000, 1000)

    image.save(save_path)


def save_heatmap_overlay(image, heatmap, save_path):
    check_mkdir(save_path)

    image = image.cpu().squeeze(0).numpy().transpose((1, 2, 0))*255
    image = image.clip(0, 255).astype(np.uint8)

    heatmap = heatmap.cpu().squeeze(0).numpy()*255
    heatmap = heatmap.clip(0, 255).astype(np.uint8)
    heatmap  = 255 - heatmap

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


    overlayed_image = cv2.addWeighted(heatmap, 0.6, image, 0.4, 0)
    overlayed_image = Image.fromarray(overlayed_image)
    
    overlayed_image.save(save_path)


def overlay_heatmap(image, heatmap):
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, 0.6, image, 0.4, 0)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.sr_model.'):
            name = name[16:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def chop_forward(x, model, scale, device, shave=16, min_size=50000, nGPUs=1, ensemble=False, ensemble_re=False):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if ensemble:
                output_batch = ensemble_forward(input_batch, model)
            elif ensemble_re:
                output_batch = ensemble_reconst_forward(input_batch, model)
            else:
                output_batch = model(input_batch, only_last=True)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, device, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = torch.empty((b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output.to(device)

def ensemble_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        # with torch.no_grad():
        #     ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug, 2, True) for aug in inputlist]

    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def ensemble_reconst_forward(img, model, precision='single'):
    ### TODO
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        # with torch.no_grad():
        #     ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i,outputs in enumerate(outputlist):
        mean_output = sum(outputs) / len(outputs)
        outputlist[i] = mean_output

    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

class debug_set(object):
    def __init__(self, output_dir, dprint=False, dwrite=False):
        self.output_file = os.path.join(output_dir,'debug.txt')
        self.dprint = dprint
        self.dwrite = dwrite
        if self.dwrite == True:
            with open(self.output_file, mode='w') as f:
                f.write('debug file\n')

    def do_all(self,sentence):
        self.print(sentence)
        self.write(sentence)

    def print(self, sentence):
        if self.dprint == True:
            print(sentence)
    
    def write(self, sentence):
        if self.dwrite == True:
            with open(self.output_file, mode='a') as f:
                f.write('{}\n'.format(sentence))


def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def chop_forward_2(x, model, scale, device, shave=24, slice_size=48, nGPUs=1, ensemble=False, ensemble_re=False, debug=False):
    b, c, h, w = x.size()
    cut_size = slice_size + (2 * shave)
    if debug:
        print('size b:{} c:{} h:{} w:{}'.format(b,c,h,w))
        print('slice_size:{}'.format(slice_size))
        print('cut size:{}'.format(cut_size))
    if cut_size % 16 != 0:
        raise print('invalid cut size. cutsize = slice_size + (shave * 2). cutsize should be 16x')    

    max_i = h // slice_size
    max_j = w // slice_size

    remain_i = h % slice_size
    remain_j = w % slice_size

    if remain_i != 0:
        max_i += 1
    if remain_j != 0:
        max_j += 1

    if debug:
        print('max i: {}'.format(max_i))
        print('max j: {}'.format(max_j))

    input_list = []
    position_list = []
    crop_pos_list = []
    for i in range(int(max_i)):
        for j in range(int(max_j)):
            start_x = i * slice_size - shave
            start_y = j * slice_size - shave
            sr_start_x = (start_x + shave) * scale
            sr_start_y = (start_y + shave) * scale
            crop_start_x = shave * scale
            crop_start_y = shave * scale

            if start_x < 0:
                start_x = 0
                sr_start_x = 0
                crop_start_x = 0
            if start_y < 0:
                start_y = 0
                sr_start_y = 0
                crop_start_y = 0
            
            end_x = start_x + cut_size
            end_y = start_y + cut_size
            sr_end_x = (end_x - shave) * scale
            sr_end_y = (end_y - shave) * scale
            crop_end_x = scale * (cut_size - shave) 
            crop_end_y = scale * (cut_size - shave)

            if end_x > h:
                end_x = h
                start_x = h - cut_size
                sr_start_x = slice_size * i * scale
                sr_end_x = scale * h
                crop_start_x = (scale * cut_size) - (sr_end_x - sr_start_x)
                crop_end_x = scale * cut_size
            if end_y > w:
                end_y = w
                start_y = w - cut_size
                sr_start_y = position_list[-1][3]
                sr_end_y = scale * w
                crop_start_y = (scale * cut_size) - (sr_end_y - sr_start_y)
                crop_end_y = scale * cut_size

            cut_x = x[:,:,start_x:end_x, start_y:end_y]

            if debug:
                print('x start:{} end:{}   y stat:{} end:{}'.format(start_x,end_x,start_y,end_y))
                print('sr x start:{} end:{}   y start:{} end:{}'.format(sr_start_x,sr_end_x,sr_start_y,sr_end_y))
                # print('cut x size:{}'.format(cut_x.size()))
                print("")
            input_list.append(cut_x)
            position_list.append([sr_start_x,sr_end_x,sr_start_y,sr_end_y])
            crop_pos_list.append([crop_start_x,crop_end_x,crop_start_y,crop_end_y])
    
    outputlist = []
    num_cut_x = len(input_list)
    if debug:
        print('num cat x:{}'.format(num_cut_x))
    
    for i in range(0, num_cut_x, nGPUs):
        if debug:
            print('cat x id: {}'.format(i))
        
        if i + nGPUs > num_cut_x:
            # print('break')
            break

        input_batch = torch.cat(input_list[i:(i + nGPUs)], dim=0)
        if ensemble:
            output_batch = ensemble_forward(input_batch, model)
        elif ensemble_re:
            output_batch = ensemble_reconst_forward(input_batch, model)
        else:
            with torch.no_grad():
                output_batch = model(input_batch, only_last=True)
            
        output_batch = output_batch.to('cpu')

        # output_batch_wwwwww = tuple(output_batch_wwwwww)
        outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    
    if num_cut_x % nGPUs != 0:
        remained_data_num = num_cut_x % nGPUs
        input_batch = torch.cat(input_list[-nGPUs:], dim=0)
        if ensemble:
            output_batch = ensemble_forward(input_batch, model)
        elif ensemble_re:
            output_batch = ensemble_reconst_forward(input_batch, model)
        else:
            output_batch = model(input_batch, only_last=True)
        
        output_batch = output_batch.to('cpu')

        # output_batch_wwwwww = tuple(output_batch_wwwwww)
        outputlist.extend(output_batch[-remained_data_num:].chunk(remained_data_num, dim=0))

    if debug:
        print('num output: {}'.format(len(outputlist)))

    output = torch.empty((b, c, scale*h, scale*w))
    for (start_x,end_x,start_y,end_y),(crop_start_x,crop_end_x,crop_start_y,crop_end_y), data in zip(position_list, crop_pos_list, outputlist):
        if debug:
            print('paste position i:{} j:{}'.format(start_x,start_y))
            print('x len:{}   y len:{}'.format(end_x - start_x, end_y - start_y))
            print('crop x len:{}   crop y len:{}'.format(crop_end_x - crop_start_x, crop_end_y - crop_start_y))
        output[:,:,start_x:end_x,start_y:end_y] = data[:,:,crop_start_x:crop_end_x,crop_start_y:crop_end_y]
    
    b, c, final_h, final_w = output.size()
    if debug:
        print('final size b:{} c:{} h:{} w:{}'.format(b,c,final_h,final_w))
        print('true size b:{} c:{} h:{} w:{}'.format(b,c,scale * h,scale * w))
    return output.to(device)


def _sigmoid(x):
	y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
	return y