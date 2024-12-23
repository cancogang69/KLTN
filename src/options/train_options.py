from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save-latest-freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save-epoch-freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save-by-iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue-train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch-count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n-epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n-epochs-decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan-mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool-size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr-policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr-decay-iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # validation parameters
        parser.add_argument("--val-freq", type=int, default=2, help="th number of epochs until validation")
        # dataset parameters
        parser.add_argument("--train-anno-path", type=str, required=True, help="path to the train json annotation file")
        parser.add_argument("--val-anno-path", type=str, required=True, help="path to the validation json annotation file")

        self.isTrain = True
        return parser
