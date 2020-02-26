import time
import torch.nn
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
from utils import dataset_util
import numpy as np
import os
from PIL import Image
import cv2

if __name__ == '__main__':
    opt = TestOptions().parse()
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)   
    print('#test images = %d' % dataset_size)
    
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    save_dir = os.path.join('results', opt.model+'_'+opt.suffix+'_'+opt.which_epoch)
    os.makedirs(save_dir)
    num_samples = len(data_loader)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    MAX_DEPTH = 80 #50
    MIN_DEPTH = 1e-3
    
    for ind, data in enumerate(data_loader):

        model.set_input(data)        
        model.test()

        visuals = model.get_current_visuals()

        gt_depth = np.squeeze(data['depth'].data.numpy())
        pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
            
        w = gt_depth.shape[1]
        h = gt_depth.shape[0]
       
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * h, 0.99189189 * h,
                            0.03594771 * w,  0.96405229 * w]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        
        pred_depth = cv2.resize(pred_depth, (w, h), cv2.INTER_CUBIC)
        pred_depth += 1.0
        pred_depth /= 2.0
        pred_depth *= 655.35
        
        # evaluate
        pred_depth[pred_depth<1e-3] = 1e-3
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    
        abs_rel[ind], sq_rel[ind], rms[ind], log_rms[ind], a1[ind], a2[ind], a3[ind] = dataset_util.compute_errors(gt_depth[mask], pred_depth[mask])
        
        # save
        pred_img = Image.fromarray(pred_depth.astype(np.int32)*100, 'I')
        pred_img.save('%s/%05d_pred.png'%(save_dir, ind))
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))
