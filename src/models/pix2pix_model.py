import torch
from .base_model import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        if opt.use_extra_info:
            opt.input_nc = opt.input_nc + 2

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout)
        if self.opt.model_generator_path is not None:
            self.load_network(self.opt.model_generator_path, "G")
        else:
            networks.init_weights(self.netG, opt.init_type, opt.init_gain)

        if self.opt.is_ddp:
            networks.to_ddp(self.netG, self.rank)
        else:
            self.netG = self.netG.to(self.device)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm)
            if self.opt.model_discriminator_path is not None:
                self.load_network(self.opt.model_discriminator_path, "D")
            else:
                networks.init_weights(self.netD, opt.init_type, opt.init_gain)

            if self.opt.is_ddp:
                networks.to_ddp(self.netD, self.rank)
            else:
                self.netD = self.netD.to(self.device)

            self.scaler = torch.cuda.amp.GradScaler()

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            if self.opt.loss_type == "l1":
                self.criterionPixel = torch.nn.L1Loss()
            elif self.opt.loss_type == "cross_entropy":
                self.criterionPixel = torch.nn.CrossEntropyLoss()
            elif self.opt.loss_type == "none":
                self.criterionPixel = None
            else:
                raise Exception(f"The {self.opt.loss_type} loss function is not supported")
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.opt.optimizer_type == "adam":
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            elif self.opt.optimizer_type == "sgd":
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9)
            else:
                raise Exception(f"Does not implement {self.opt.optimizer_type} optimizer")

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input.to(self.device)
        self.real_B = target.to(self.device)

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A).to(self.device) # G(A)


    def forward_only(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            predict = self.netG(input)

        return predict.detach().cpu()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(pred_fake.to(self.device), False)
            # Real
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.scaler.scale(self.loss_D).backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_pixel = self.criterionPixel(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN.to(self.device) + self.loss_G_pixel.to(self.device)
            
        self.scaler.scale(self.loss_G).backward()

    def optimize_parameters(self, is_discriminator_backprop=True):
        self.forward()          

        # update D
        if is_discriminator_backprop:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()     
            self.backward_D()               
            self.scaler.step(self.optimizer_D)
        else:
            self.loss_D = None 

        # update G
        self.set_requires_grad(self.netD, False)  
        self.optimizer_G.zero_grad()       
        self.backward_G()                   
        self.scaler.step(self.optimizer_G)

        if self.loss_D is not None:
            return self.loss_G.item(), self.loss_D.item()
        else:
            return self.loss_G.item(), 0    
