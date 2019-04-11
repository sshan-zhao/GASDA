from __future__ import print_function
import torch
import time
import numpy as np
from PIL import Image
import os

# save image to the disk
def save_images(visuals, results_dir, ind):
    
    for label, im_data in visuals.items():
        
        img_path = os.path.join(results_dir, '%.3d_%s.png' % (ind, label))
        if 'depth' in label:
            pass
        else:
            image_numpy = tensor2im(im_data)
            save_image(image_numpy, img_path, 'RGB')

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1)
    image_numpy = image_numpy / (2.0 / 255.0)
    return image_numpy.astype(imtype)

def tensor2depth(input_depth, imtype=np.int32):
    if isinstance(input_depth, torch.Tensor):
        depth_tensor = input_depth.data
    else:
        return input_depth
    depth_numpy = depth_tensor[0].cpu().float().numpy() 
    depth_numpy += 1.0
    depth_numpy /= 2.0
    depth_numpy *= 65535.0
    depth_numpy = depth_numpy.reshape((depth_numpy.shape[1], depth_numpy.shape[2]))
    return depth_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, imtype):
    image_pil = Image.fromarray(image_numpy, imtype)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

class SaveResults:
    def __init__(self, opt):
       
        self.img_dir = os.path.join(opt.checkpoints_dir, opt.expr_name, 'image')
        mkdirs(self.img_dir) 
        self.log_name = os.path.join(opt.checkpoints_dir, opt.expr_name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def save_current_results(self, visuals, epoch):
            
        for label, image in visuals.items():
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            if image is None:
                continue
            if 'depth' in label:
                depth_numpy = tensor2depth(image)
                save_image(depth_numpy, img_path, 'I')
            else:
                image_numpy = tensor2im(image)
                save_image(image_numpy, img_path, 'RGB')
            

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, lr, losses, t, t_data):
          
        message = '(epoch: %d, iters: %d, lr: %e, time: %.3f, data: %.3f) ' % (epoch, i, lr, t, t_data)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_validation_errors(self, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, domain):
        message = '(%s abs_rel: %.3f, sq_rel: %.3f, rmse: %.3f, rmse_log: %.3f, a1: %.3f, a2: %.3f, a3: %.3f)' % (domain, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % message)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
