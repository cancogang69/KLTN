import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # network saving and loading parameters
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input-nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output-nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='unet_256', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')  
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
    parser.add_argument('--gan-mode', type=str, default='vanilla', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument('--pool-size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--lr-policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr-decay-iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument("--save-path", type=str, default="checkpoints")
    parser.add_argument('--loss-type', type=float, default="l1", help='loss function [l1, cross_entropy]')
    parser.add_argument('--lambda-L1', type=float, default=100.0, help='weight for L1 loss')
    # validation parameters
    parser.add_argument("--val-freq", type=int, default=2, help="th number of epochs until validation")
    parser.add_argument("--plot-save-path", type=str, default="val_output")
    # dataset parameters
    parser.add_argument("--train-anno-path", type=str, required=True, help="path to the train json annotation file")
    parser.add_argument("--val-anno-path", type=str, required=True, help="path to the validation json annotation file")
    parser.add_argument('--image-root', type=str)
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no-flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--save-latest-freq', type=int, default=5000, help='frequency of saving the latest results')
    args = parser.parse_args()
    return args