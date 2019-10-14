import time
import torch.nn
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
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
            
    for ind, data in enumerate(data_loader):

        model.set_input(data)        
        model.test()

        visuals = model.get_current_visuals()

        gt_depth = np.squeeze(data['depth'].data.numpy())
        pred_depth = np.squeeze(visuals['pred'].data.cpu().numpy())
            
        w = gt_depth.shape[1]
        h = gt_depth.shape[0]
        w0 = pred_depth.shape[1] 
        pred_depth = cv2.resize(pred_depth, (w, h), cv2.INTER_CUBIC)
        pred_depth += 1.0
        pred_depth /= 2.0
        pred_depth *= 65535

        pred_depth[pred_depth<1e-3] = 1e-3
       
        pred_img = Image.fromarray(pred_depth.astype(np.int32), 'I')
        pred_img.save('%s/%05d_pred.png'%(save_dir, ind))
       
