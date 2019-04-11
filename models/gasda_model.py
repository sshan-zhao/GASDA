import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
import torch.nn.functional as F
from utils import dataset_util

class GASDAModel(BaseModel):
    def name(self):
        return 'GASDAModelModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=50.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_C_Depth', type=float, default=50.0, help='weight for consistency')

            parser.add_argument('--lambda_S_Depth', type=float, default=0.01,
                                help='weight for smooth loss')
            
            parser.add_argument('--lambda_R_Img', type=float, default=50.0,help='weight for image reconstruction')
            # cyclegan
            parser.add_argument('--lambda_Src', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_Tgt', type=float, default=1.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=30.0,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

            parser.add_argument('--s_depth_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')
            parser.add_argument('--t_depth_premodel', type=str, default=" ",
                                help='pretrained depth estimation model')

            parser.add_argument('--g_src_premodel', type=str, default=" ",
                                help='pretrained G_Src model')
            parser.add_argument('--g_tgt_premodel', type=str, default=" ",
                                help='pretrained G_Tgt model')
            parser.add_argument('--d_src_premodel', type=str, default=" ",
                                help='pretrained D_Src model')
            parser.add_argument('--d_tgt_premodel', type=str, default=" ",
                                help='pretrained D_Tgt model')

            parser.add_argument('--train_mde', action='store_true', help='only trian G_Depth_T and G_Depth_S')
            parser.add_argument('--train_all', action='store_true', help='train the whole network')
            parser.add_argument('--freeze_bn', action='store_true', help='freeze the bn in mde')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
         
        if self.isTrain:
            assert not (opt.train_all and opt.train_mde) and (opt.train_all or opt.train_mde)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.isTrain:
            self.loss_names = ['R_Depth_Src_S', 'S_Depth_Tgt_S', 'R_Img_Tgt_S', 'C_Depth_Tgt']
            self.loss_names += ['R_Depth_Src_T', 'S_Depth_Tgt_T', 'R_Img_Tgt_T']
            # cyclegan
            self.loss_names += ['D_Src', 'G_Src', 'cycle_Src', 'idt_Src', 'D_Tgt', 'G_Tgt', 'cycle_Tgt', 'idt_Tgt']

         # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            visual_names_src = ['src_img', 'fake_tgt', 'rec_src', 'src_real_depth', 'src_gen_depth', 'src_gen_depth_t', 'src_gen_depth_s']
            visual_names_tgt = ['tgt_left_img', 'fake_src_left', 'rec_tgt_left', 'tgt_gen_depth', 'warp_tgt_img_s', 'warp_tgt_img_t', 'tgt_gen_depth_s', 'tgt_gen_depth_t', 'tgt_right_img']
            if self.opt.lambda_identity > 0.0:
                visual_names_src.append('idt_src_left')
                visual_names_tgt.append('idt_tgt')
            self.visual_names = visual_names_src + visual_names_tgt
        else:
            self.visual_names = ['pred', 'img', 'img_trans']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_Depth_S', 'G_Depth_T']

            # cyclegan
            self.model_names += ['G_Src', 'G_Tgt', 'D_Src', 'D_Tgt']
        else:
            self.model_names = ['G_Depth_S', 'G_Depth_T', 'G_Tgt']

        if len(opt.gpu_ids) > 1:
            norm = 'synbatch'
        else:
            norm = 'batch'
        self.netG_Depth_S = networks.init_net(networks.UNetGenerator(norm=norm), init_type='normal', gpu_ids=opt.gpu_ids)
        self.netG_Depth_T = networks.init_net(networks.UNetGenerator(norm=norm), init_type='normal', gpu_ids=opt.gpu_ids)

        # cyclegan
        self.netG_Src = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
        self.netG_Tgt = networks.init_net(networks.ResGenerator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.netD_Src = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)
            self.netD_Tgt = networks.init_net(networks.Discriminator(norm='instance'), init_type='kaiming', gpu_ids=opt.gpu_ids)

            self.init_with_pretrained_model('G_Depth_S', self.opt.s_depth_premodel)
            self.init_with_pretrained_model('G_Depth_T', self.opt.t_depth_premodel)
            # cyclegan
            if opt.train_mde:
                assert self.opt.g_src_premodel != " " and self.opt.g_tgt_premodel != " " and self.opt.d_src_premodel != " " and self.opt.d_tgt_premodel != " "
            self.init_with_pretrained_model('G_Src', self.opt.g_src_premodel)
            self.init_with_pretrained_model('G_Tgt', self.opt.g_tgt_premodel)
            self.init_with_pretrained_model('D_Src', self.opt.d_src_premodel)
            self.init_with_pretrained_model('D_Tgt', self.opt.d_tgt_premodel)

         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = torch.nn.L1Loss()
            self.criterionDepthCons = torch.nn.L1Loss()
            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconLoss()
            self.criterionLR = torch.nn.L1Loss()
            # cyclegan
            self.fake_src_pool = ImagePool(opt.pool_size)
            self.fake_tgt_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netG_Depth_S.parameters(),
                                                                    self.netG_Depth_T.parameters()),
                                                                    lr=opt.lr_task, betas=(0.9, 0.999))
            self.optimizer_G_trans = torch.optim.Adam(itertools.chain(self.netG_Src.parameters(), 
                                                                    self.netG_Tgt.parameters()),
                                                                    lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_Src.parameters(), 
                                                                    self.netD_Tgt.parameters()),
                                                                    lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)
            self.optimizers.append(self.optimizer_G_trans)
            self.optimizers.append(self.optimizer_D)

            if opt.train_mde:
                self.netG_Src.eval()
                self.netG_Tgt.eval()
                self.netD_Src.eval()
                self.netD_Tgt.eval()
            if opt.freeze_bn:
                self.netG_Depth_S.apply(networks.freeze_bn)
                self.netG_Depth_T.apply(networks.freeze_bn)
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

    def forward(self, phase='train'):

        if self.isTrain:
            if phase == 'val':
                if self.opt.freeze_bn:
                    self.netG_Depth_S.apply(networks.freeze_bn)
                    self.netG_Depth_T.apply(networks.freeze_bn)
                else:
                    self.netG_Depth_S.eval()
                    self.netG_Depth_T.eval()
                self.netG_Src.eval()
                self.netG_Tgt.eval()

            if phase == 'train':
                # translation
                if self.opt.train_all:
                    self.gen1 = self.netG_Src(torch.cat((self.src_img, self.tgt_left_img), 0))
                    self.fake_tgt = torch.narrow(self.gen1, 0, 0, self.num)
                    self.idt_src_left = torch.narrow(self.gen1, 0, self.num, self.num) #self.netG_Src(self.tgt_left_img)
                    self.rec_src = self.netG_Tgt(self.fake_tgt)
                    self.gen2 = self.netG_Tgt(torch.cat((self.tgt_left_img, self.src_img), 0))
                    self.fake_src_left = torch.narrow(self.gen2, 0, 0, self.num) #self.netG_Tgt(self.tgt_left_img)
                    self.idt_tgt = torch.narrow(self.gen2, 0, self.num, self.num) #self.netG_Tgt(self.src_img_ind)
                    self.rec_tgt_left = self.netG_Src(self.fake_src_left)

                # task
                if self.opt.train_mde:
                    self.fake_tgt = self.netG_Src(self.src_img).detach()
                    self.idt_src_left = None
                    self.rec_src = None
                    self.fake_src_left = self.netG_Tgt(self.tgt_left_img).detach()
                    self.idt_tgt = None
                    self.rec_tgt_left = None

                self.out_s = self.netG_Depth_S(torch.cat((self.fake_tgt,self.tgt_left_img),0))
                self.out_t = self.netG_Depth_T(torch.cat((self.src_img, self.fake_src_left), 0))

                self.src_gen_depth_t = torch.narrow(self.out_t[-1], 0, 0, self.num) #[:self.num,:,:,:]
                self.tgt_gen_depth_t = torch.narrow(self.out_t[-1], 0, self.num, self.num) #[self.num:,:,:,:]
                self.src_gen_depth_s = torch.narrow(self.out_s[-1], 0, 0, self.num) #[:self.num,:,:,:]
                self.tgt_gen_depth_s = torch.narrow(self.out_s[-1], 0, self.num, self.num) #[self.num:,:,:,:]

                self.tgt_gen_depth = (self.tgt_gen_depth_t + self.tgt_gen_depth_s) / 2.0
                self.src_gen_depth = (self.src_gen_depth_t + self.src_gen_depth_s) / 2.0

            elif phase == 'val':
                self.pred_s = self.netG_Depth_S(self.tgt_left_img)[-1]
                self.img_trans = self.netG_Tgt(self.tgt_left_img)
                self.pred_t = self.netG_Depth_T(self.img_trans)[-1]
                self.tgt_gen_depth = 0.5 * (self.pred_s + self.pred_t)
                #self.tgt_gen_depth = self.pred_s 
                self.src_gen_depth = None
                self.src_gen_depth_s = None
                self.src_gen_depth_t = None
                self.tgt_gen_depth_s = None
                self.tgt_gen_depth_t = None
                self.fake_tgt = None
                self.idt_src_left = None
                self.fake_src_left = None
                self.idt_tgt = None
                self.warp_tgt_img_t = None
                self.warp_tgt_img_s = None
                self.rec_src = None
                self.rec_tgt_left = None

            if phase == 'val':
                if not self.opt.freeze_bn:
                    self.netG_Depth_S.train()
                    self.netG_Depth_T.train()
                self.netG_Src.train()
                self.netG_Tgt.train()
    
        else:
            self.pred_s = self.netG_Depth_S(self.img)[-1]
            self.img_trans = self.netG_Tgt(self.img)
            self.pred_t = self.netG_Depth_T(self.img_trans)[-1]
            self.pred = 0.5 * (self.pred_s + self.pred_t)
            #self.pred = self.pred_s 

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_Src(self):
        fake_tgt = self.fake_tgt_pool.query(self.fake_tgt)
        self.loss_D_Src = self.backward_D_basic(self.netD_Src, self.tgt_left_img, fake_tgt)

    def backward_D_Tgt(self):
        fake_src_left = self.fake_src_pool.query(self.fake_src_left)
        self.loss_D_Tgt = self.backward_D_basic(self.netD_Tgt, self.src_img, fake_src_left)

    def backward_G(self):

        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_R_Img = self.opt.lambda_R_Img
        lambda_S_Depth = self.opt.lambda_S_Depth
        lambda_C_Depth = self.opt.lambda_C_Depth

        # =========================== translation ========================
        lambda_idt = self.opt.lambda_identity
        lambda_Src = self.opt.lambda_Src
        lambda_Tgt = self.opt.lambda_Tgt
       
        if self.opt.train_all:
            self.loss_G_Src = self.criterionGAN(self.netD_Src(self.fake_tgt), True)
            self.loss_G_Tgt = self.criterionGAN(self.netD_Tgt(self.fake_src_left), True)
            self.loss_cycle_Src = self.criterionCycle(self.rec_src, self.src_img) * lambda_Src
            self.loss_cycle_Tgt = self.criterionCycle(self.rec_tgt_left, self.tgt_left_img) * lambda_Tgt
            self.loss_idt_Src = self.criterionIdt(self.idt_src_left, self.tgt_left_img) * lambda_Tgt * lambda_idt
            self.loss_idt_Tgt = self.criterionIdt(self.idt_tgt, self.src_img) * lambda_Src * lambda_idt
            
        elif self.opt.train_mde:
            self.loss_G_Src = 0
            self.loss_G_Tgt = 0
            self.loss_cycle_Src = 0 
            self.loss_cycle_Tgt = 0 
            self.loss_idt_Tgt = 0
            self.loss_idt_Src = 0
        self.loss_G_GAN = self.loss_G_Src + self.loss_G_Tgt + self.loss_cycle_Src + self.loss_cycle_Tgt + self.loss_idt_Src + self.loss_idt_Tgt
        
        # ============================= task =============================
        # --------------------------- synthetic --------------------------
        self.loss_R_Depth_Src_S = 0.0
        real_depths = dataset_util.scale_pyramid(self.src_real_depth, 4)
        for (gen_depth, real_depth) in zip(self.out_s, real_depths):
            self.loss_R_Depth_Src_S += self.criterionDepthReg(gen_depth[:self.num,:,:,:], real_depth) * lambda_R_Depth
        self.loss_R_Depth_Src_T = 0.0
        for (gen_depth, real_depth) in zip(self.out_t, real_depths):
            self.loss_R_Depth_Src_T += self.criterionDepthReg(gen_depth[:self.num,:,:,:], real_depth) * lambda_R_Depth

        # ---------------------------- real ------------------------------
        # geometry consistency
        l_imgs = dataset_util.scale_pyramid(self.tgt_left_img, 4)
        r_imgs = dataset_util.scale_pyramid(self.tgt_right_img, 4)
        self.loss_R_Img_Tgt_S = 0.0
        i = 0
        for (l_img, r_img, gen_depth) in zip(l_imgs, r_imgs, self.out_s):
            loss, self.warp_tgt_img_s = self.criterionImgRecon(l_img, r_img, gen_depth[self.num:,:,:,:], self.tgt_fb / 2**(3-i))
            self.loss_R_Img_Tgt_S += loss * lambda_R_Img
            i += 1
        self.loss_R_Img_Tgt_T = 0.0
        i = 0
        for (l_img, r_img, gen_depth) in zip(l_imgs, r_imgs, self.out_t):
            loss, self.warp_tgt_img_t = self.criterionImgRecon(l_img, r_img, gen_depth[self.num:,:,:,:], self.tgt_fb / 2**(3-i))
            self.loss_R_Img_Tgt_T += loss * lambda_R_Img
            i += 1
        # smoothness
        i = 0
        self.loss_S_Depth_Tgt_S = 0.0
        for (gen_depth, img) in zip(self.out_s, l_imgs):
            self.loss_S_Depth_Tgt_S += self.criterionSmooth(gen_depth[self.num:,:,:,:], img) * self.opt.lambda_S_Depth / 2**i
            i += 1
        i = 0
        self.loss_S_Depth_Tgt_T = 0.0
        for (gen_depth, img) in zip(self.out_t, l_imgs):
            self.loss_S_Depth_Tgt_T += self.criterionSmooth(gen_depth[self.num:,:,:,:], img) * self.opt.lambda_S_Depth / 2**i
            i += 1

        # depth consistency
        self.loss_C_Depth_Tgt = 0.0
        for (gen_depth1, gen_depth2) in zip(self.out_s, self.out_t):
            self.loss_C_Depth_Tgt += self.criterionDepthCons(gen_depth1[self.num:,:,:,:], gen_depth2[self.num:,:,:,:]) * lambda_C_Depth

        self.loss_G = self.loss_R_Depth_Src_S + self.loss_R_Depth_Src_T + self.loss_G_GAN + self.loss_R_Img_Tgt_T + self.loss_R_Img_Tgt_S + self.loss_S_Depth_Tgt_T + self.loss_S_Depth_Tgt_S + self.loss_C_Depth_Tgt
        # self.loss_G = self.loss_G_GAN #+ self.loss_R_Img_Tgt_T + self.loss_S_Depth_Tgt_T + self.loss_C_Depth_Tgt

        self.loss_G.backward()
  

    def optimize_parameters(self, epoch=1, phase='train'):
        # forward
        
        if phase == 'train':
            self.forward()
            #self.set_requires_grad([self.netG_Src, self.netG_Tgt, self.netG_Depth_S, self.netG_Depth_T], True)
            if self.opt.train_all:
                self.set_requires_grad([self.netD_Src, self.netD_Tgt], False)
                self.optimizer_G_trans.zero_grad()
            self.optimizer_G_task.zero_grad()
            self.backward_G()
            if self.opt.train_all:
                self.optimizer_G_trans.step()
            self.optimizer_G_task.step()
            self.loss_D_Src = 0
            self.loss_D_Tgt = 0

            #self.set_requires_grad([self.netG_Src, self.netG_Tgt, self.netG_Depth_S, self.netG_Depth_T], False)
            if self.opt.train_all:
                self.set_requires_grad([self.netD_Src, self.netD_Tgt], True)
                self.optimizer_D.zero_grad()
                self.backward_D_Src()
                self.backward_D_Tgt()
                self.optimizer_D.step()

        else:
            # G_Depth
            self.forward('val')
            #self.backward_G(bw=False)
