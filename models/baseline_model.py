import torch
import itertools
from .base_model import BaseModel
from . import networks


class BaselineModel(BaseModel):
    def name(self):
        return 'BaselineModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if is_train:
            parser.add_argument('--lambda_R_Depth', type=float, default=1.0, help='weight for reconstruction loss')
            parser.add_argument('--lambda_S_Depth', type=float, default=0.1,
                                help='weight for smooth loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.isTrain:
		
            self.loss_names = ['R_Depth_Src', 'S_Depth_Src']
	    
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['src_real_depth', 'src_gen_depth', 'src_img', 'tgt_gen_depth', 'tgt_left_img']
            
        else:
            self.visual_names = ['pred', 'img']

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
                self.model_names = ['G_Depth', 'G_Feat']
        else:  
            self.model_names = ['G_Depth', 'G_Feat']
	
        self.netG_Feat = networks.define_G(which_model_netG=opt.which_model_netG, gpu_ids=self.gpu_ids, stage='feat')
        self.netG_Depth = networks.define_G(which_model_netG=opt.which_model_netG, use_dropout=self.isTrain, up_size=self.opt.loadSize, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids, stage='depth')            
         
        if self.isTrain:
            # define loss functions
            self.criterionDepthReg = networks.BerHuLoss()
            self.criterionSmooth = networks.SmoothLoss()
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_Feat.parameters(),
								self.netG_Depth.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):

        if self.isTrain:
            self.src_real_depth = input['src']['depth'].to(self.device)
            self.src_img = input['src']['img'].to(self.device)
            self.tgt_left_img = input['tgt']['left_img'].to(self.device)
        else:
            self.img = input['img'].to(self.device)

    def forward(self, phase='train'):
        if self.isTrain:

	    num = self.src_img.shape[0]
            self.feat, _ = self.netG_Feat_Src(torch.cat((self.src_img, self.tgt_left_img), 0))
	    self.feat = torch.tensor(self.feat, requires_grad=True)
	    #self.src_feat = [torch.tensor(feat, requires_grad=True) for feat in self.src_feat]
	
	    self.depth = self.netG_Depth(self.feat)

	    self.src_gen_depth = self.depth[:num,:,:,:]
	    self.tgt_gen_depth = self.depth[num:,:,:,:]
		
        else:
	 
            self.pred = self.netG_Depth(self.netG_Feat(self.img)[0])

    def backward_G(self, bw=True):

        lambda_R_Depth = self.opt.lambda_R_Depth
        lambda_S_Depth = self.opt.lambda_S_Depth
       
        # regression loss, BerHu Loss
        self.loss_R_Depth_Src = self.criterionDepthReg(self.src_gen_depth, self.src_real_depth) * lambda_R_Depth
        # smoothness loss
        self.loss_S_Depth_Src = self.criterionSmooth(self.src_gen_depth, self.src_img) * lambda_S_Depth
	self.loss_G_Depth += self.loss_S_Depth_Src + self.loss_R_Depth_Src
        
        self.loss_G_Depth.backward()
  

    def optimize_parameters(self, train_iter=1, phase='train'):
        # forward
        
        if phase == 'train':
            # G_Depth
	    self.forward()
	    self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        else:
            # G_Depth
            self.forward(phase)
            #self.backward_G(bw=False)
