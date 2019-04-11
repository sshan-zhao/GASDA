from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--root', type=str, default='datasets/kitti', help='data root')
        parser.add_argument('--test_datafile', type=str, default='test.txt', help='stores data list, in root')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--save', action='store_true', help='save results')
        parser.add_argument('--test_dataset', type=str, default='kitti', help='kitti|stereo|make3d')
        
        self.isTrain = False
        return parser
