from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--src_train_datafile', type=str, default='train.txt', help='stores data list, in src_root')
        parser.add_argument('--tgt_train_datafile', type=str, default='train.txt', help='stores data list, in tgt_root')
        parser.add_argument('--print_freq', type=int, default=32, help='frequency of showing training results on console')
        parser.add_argument('--save_result_freq', type=int, default=3200, help='frequency of saving the latest prediction results')
        parser.add_argument('--save_latest_freq', type=int, default=3200, help='frequency of saving the latest trained model')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr_task', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_trans', type=float, default=5e-5, help='initial learning rate for adam')
        parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        parser.add_argument('--scale_pred', action='store_true', help='scale prediction according the ratio of median value')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--no_val', action='store_true', help='validation')

        self.isTrain = True
        return parser
