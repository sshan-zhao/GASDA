import time
import torch.nn
from options.train_options import TrainOptions
from data import create_dataloader
from models import create_model
from utils.util import SaveResults
from utils import dataset_util, util
import numpy as np
import cv2
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    train_data_loader = create_dataloader(opt)
    train_dataset_size = len(train_data_loader)   
    print('#training images = %d' % train_dataset_size)
    
    model = create_model(opt)
    model.setup(opt)
    save_results = SaveResults(opt)
    total_steps = 0

    lr = opt.lr_task

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # training
        print("training stage (epoch: %s) starting...................." % epoch)
        for ind, data in enumerate(train_data_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                save_results.print_current_losses(epoch, epoch_iter, lr, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                model.save_networks('latest')

            if total_steps % opt.save_result_freq == 0:
                save_results.save_current_results(model.get_current_visuals(), epoch)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        lr = model.update_learning_rate()
