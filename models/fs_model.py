import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util

class FSModel(BaseModel):
    def name(self):
        return 'FSModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Depth', type=float, default=0.01, help='weight for smooth loss')
            
            parser.add_argument('--lambda_R_Img', type=float, default=1.0,help='weight for image reconstruction')
            
            parser.add_argument('--g_src_premodel', type=str, default=" ",help='pretrained G_Src model')
        
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        if self.isTrain:
            self.loss_names = ['R_Depth_Src', 'S_Depth_Tgt', 'R_Img_Tgt']
          
        if self.isTrain:
            self.visual_names = ['src_img', 'fake_tgt', 'src_real_depth', 'src_gen_depth', 'tgt_left_img', 'tgt_gen_depth', 'warp_tgt_img', 'tgt_right_img']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            self.model_names = ['G_Depth_S']

        else:
            self.model_names = ['G_Depth_S']

        self.netG_Depth_S = networks.init_net(networks.UNetGenerator(norm='batch'), init_type='normal', gpu_ids=opt.gpu_ids)

        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            self.netG_Src.eval()
         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconLoss()

            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netG_Depth_S.parameters()),
                                                lr=opt.lr_task, betas=(0.9, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)

    def set_input(self, input):

        if self.isTrain:
            self.src_real_depth = input['src']['depth'].to(self.device)
            self.src_img = input['src']['img'].to(self.device)
            self.tgt_left_img = input['tgt']['left_img'].to(self.device)
            self.tgt_right_img = input['tgt']['right_img'].to(self.device)
            self.tgt_fb = input['tgt']['fb']

            self.num = self.src_img.shape[0]
        else:
            self.img = input['left_img'].to(self.device)

    def forward(self):

        if self.isTrain:

            self.fake_tgt = self.netG_Src(self.src_img).detach()
            self.out = self.netG_Depth_S(torch.cat((self.fake_tgt, self.tgt_left_img), 0))
            self.src_gen_depth = self.out[-1].narrow(0, 0, self.num)
            self.tgt_gen_depth = self.out[-1].narrow(0, self.num, self.num)
            
        else:
            self.pred = self.netG_Depth_S(self.img)[-1]

    def backward_G(self):

        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_R_Img = self.opt.lambda_R_Img
        lambda_S_Depth = self.opt.lambda_S_Depth

        self.loss_R_Depth_Src = 0.0
        real_depths = dataset_util.scale_pyramid(self.src_real_depth, 4)
        for (gen_depth, real_depth) in zip(self.out, real_depths):
            self.loss_R_Depth_Src += self.criterionDepthReg(gen_depth[:self.num,:,:,:], real_depth) * lambda_R_Depth

        l_imgs = dataset_util.scale_pyramid(self.tgt_left_img, 4)
        r_imgs = dataset_util.scale_pyramid(self.tgt_right_img, 4)
        self.loss_R_Img_Tgt = 0.0
        i = 0
        for (l_img, r_img, gen_depth) in zip(l_imgs, r_imgs, self.out):
            loss, self.warp_tgt_img = self.criterionImgRecon(l_img, r_img, gen_depth[self.num:,:,:,:], self.tgt_fb / 2**(3-i))
            self.loss_R_Img_Tgt += loss * lambda_R_Img
            i += 1

        i = 0
        self.loss_S_Depth_Tgt = 0.0
        for (gen_depth, img) in zip(self.out, l_imgs):
            self.loss_S_Depth_Tgt += self.criterionSmooth(gen_depth[self.num:,:,:,:], img) * self.opt.lambda_S_Depth / 2**i
            i += 1

        self.loss_G_Depth = self.loss_R_Img_Tgt + self.loss_S_Depth_Tgt + self.loss_R_Depth_Src
        self.loss_G_Depth.backward()

    def optimize_parameters(self):
        
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
