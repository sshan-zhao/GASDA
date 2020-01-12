# GASDA
This is the PyTorch implementation for our CVPR'19 paper:

**S. Zhao, H. Fu, M. Gong and D. Tao. Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation. [PAPER](https://sshan-zhao.github.io/papers/gasda.pdf) [POSTER](https://sshan-zhao.github.io/papers/gasda_poster.pdf)**

![Framework](https://github.com/sshan-zhao/GASDA/blob/master/img/framework.png)

## Environment
1. Python 3.6
2. PyTorch 0.4.1
3. CUDA 9.0
4. Ubuntu 16.04

## Datasets
[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)

[vKITTI](https://europe.naverlabs.com/Research/Computer-Vision/Proxy-Virtual-Worlds/)

Prepare the two datasets according to the datalists (*.txt in [datasets](https://github.com/sshan-zhao/GASDA/tree/master/datasets))
```
datasets
  |----kitti 
         |----2011_09_26         
         |----2011_09_28        
         |----.........        
  |----vkitti 
         |----rgb        
               |----0006              
               |-----.......             
         |----depth       
               |----0006        
               |----.......
```

## Training (Tesla V100, 16GB)
- Train [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) using the official experimental settings, or download our [pretrained models](https://1drv.ms/f/s!Aq9eyj7afTjMcZorokRKW4ATgZ8).

- Train F_t
```
python train.py --model ft --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_tgt_premodel ./cyclegan/G_Tgt.pth
```

- Train F_s
```
python train.py --model fs --gpu_ids 0 --batchSize 8 --loadSize 256 1024 --g_src_premodel ./cyclegan/G_Src.pth
```

- Train GASDA using the pretrained F_s, F_t and CycleGAN.
```
python train.py --freeze_bn --freeze_in --model gasda --gpu_ids 0 --batchSize 3 --loadSize 192 640 --g_src_premodel ./cyclegan/G_Src.pth --g_tgt_premodel ./cyclegan/G_Tgt.pth --d_src_premodel ./cyclegan/D_Src.pth --d_tgt_premodel ./cyclegan/D_Tgt.pth --t_depth_premodel ./checkpoints/vkitti2kitti_ft_bn/**_net_G_Depth_T.pth --s_depth_premodel ./checkpoints/vkitti2kitti_fs_bn/**_net_G_Depth_S.pth 
```
Note: this training strategy is different from that in our paper.

## Test
[MODELS](https://drive.google.com/open?id=1CvuGUTObRhpZpSTYxy-BIRhft6ttMJOP).

Copy the provided models to GASDA/checkpoints/vkitti2kitti_gasda/, and rename the models 1_* (e.g., 1_net_D_Src.pth), and then
```
python test.py --test_datafile 'test.txt' --which_epoch 1 --model gasda --gpu_ids 0 --batchSize 1 --loadSize 192 640
```
## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{zhao2019geometry,
  title={Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation},
  author={Zhao, Shanshan and Fu, Huan and Gong, Mingming and Tao, Dacheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9788--9798},
  year={2019}
}
```
## Acknowledgments
Code is inspired by [T^2Net](https://github.com/lyndonzheng/Synthetic2Realistic) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Contact
Shanshan Zhao: szha4333@uni.sydney.edu.au or sshan.zhao00@gmail.com
