import csv
import os
import time

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models.inception import inception_v3
import torch.autograd as autograd
from MNIST_score import get_inception_score as MNIST_IS

from models import dc_D, dc_G, dc_d, dc_g, GoodDiscriminator, GoodGenerator, GoodDiscriminatord, MLP, DMLP, DCGAN_Generator, DCGAN_Discriminator, DCGAN_weights_init, MLP_relu
from optims import BCGD, ACGD, OCGD, FR
# from optims.cgd import BCGD
# from optims.acgd import ACGD
# from optims.ocgd import OCGD
from optims.cgd_utils import zero_grad
from utils import *
from Synthetic_Dataset import Synthetic_Dataset
from Synthetic_Dataset_1D import Synthetic_Dataset_1D
from deepviz import DeepVisuals_2D, data_folder
from deepviz_1D import DeepVisuals_1D
import deepviz_real
from ComputationalTools import *

import argparse
from datetime import datetime
import random

real_data_folder = "Datasets"
log_folder = "Logs"

seed = torch.randint(0, 1000000, (1,))


# bad seeds: 850527
# good seeds: 952132, 64843


def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


def detransform(x):
    return (x + 1.0) / 2.0


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def Get_models(args, output_shape=None, gpu_num=1):
    """
    Default:
    G = GoodGenerator()
    D = GoodDiscriminator()
    """
    # TODO: Add architecture selection
    if args.arch == "default":
        if args.data in ["cifar"]:
            G = dc_g(z_dim=args.z_dim, ngf=args.G_base_filter_num)
            D = dc_d(ndf=args.D_base_filter_num)
        elif args.data in ["mnist"]:
            G = DCGAN_Generator(nz=args.z_dim, ngf=args.G_base_filter_num, freeze_w=args.freeze_w, alpha_mobility=args.alpha_mobility)
            D = DCGAN_Discriminator(ndf=args.D_base_filter_num, alpha_mobility=args.alpha_mobility_D)
        else:
            raise NotImplementedError
    elif args.arch == "wgan":
        if args.data in ["cifar"]:
            G = GoodGenerator(dim=args.G_base_filter_num)
            D = GoodDiscriminator(dim=args.D_base_filter_num)
        else:
            raise NotImplementedError
    elif args.arch == "mlp":
        if output_shape is None:
            x_dim = args.x_dim
        else:
            x_dim = np.prod(output_shape)

        if not hasattr(args, "use_spectral_norm"):
            args.use_spectral_norm = False

        G_output_dim = x_dim
        if hasattr(args, "brenier"):
            if args.brenier:
                G_output_dim = 1

        if hasattr(args, "lazy"):
            lazy = args.lazy
        else:
            lazy = False


        if hasattr(args, "init_scheme"):
            init_scheme = args.init_scheme
        else:
            init_scheme = "relu"


        mono = False
        if hasattr(args, "mono"):
            if args.mono:
                mono = True

        if mono:
            G = MLP_relu(mono_feature=1, non_mono_feature=0, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num=args.g_hidden, non_mono_hidden_num=5)
        else:
            G = MLP(args.g_layers, args.g_hidden, args.z_dim, G_output_dim, output_shape=output_shape, type="G", alt_act_prop=args.alt_act_prop, alt_act_factor=args.alt_act_factor, alt_act_type=args.alt_act_type, freeze_w=args.freeze_w, freeze_b=args.freeze_b, freeze_v=args.freeze_v, alpha_mobility=args.alpha_mobility, use_spectral_norm=args.use_spectral_norm, lazy=lazy, init_scheme=init_scheme)



        D = MLP(args.d_layers, args.d_hidden, x_dim, 1, output_shape=output_shape, type="D", bias_scale=args.d_bias_scale, alpha_mobility=args.alpha_mobility_D, use_spectral_norm=args.use_spectral_norm, lazy=lazy, init_scheme=init_scheme)
        # print("G.hidden_layer.weight", G.hidden_layer.weight)
        # print("G.hidden_layer.bias", G.hidden_layer.bias)
        # print("G.output_layer.weight", G.output_layer.weight)
    elif args.arch == "dmlp_g":
        if output_shape is None:
            x_dim = args.x_dim
        else:
            x_dim = np.prod(output_shape)

        if not hasattr(args, "use_spectral_norm"):
            args.use_spectral_norm = False

        G = DMLP(args.g_layers, args.g_hidden, args.z_dim, x_dim, output_shape=output_shape, type="G")
        D = MLP(args.d_layers, args.d_hidden, x_dim, 1, output_shape=output_shape, type="D", bias_scale=args.d_bias_scale, alpha_mobility=args.alpha_mobility_D, use_spectral_norm=args.use_spectral_norm)
        # print("G.hidden_layer.weight", G.hidden_layer.weight)
        # print("G.hidden_layer.bias", G.hidden_layer.bias)
        # print("G.output_layer.weight", G.output_layer.weight)
    else:
        raise NotImplementedError

    return G, D


class GAN():
    def __init__(self, args=None, G=None, D=None, video_title=None, dataset_name=None, loss_type=None, method=None, z_dim=8, batch_size=256, lr_x=0.1, lr_y=0.1, show_iter=100, weight_decay=0.0, penalty_x=0.0, penalty_y=0.0, test_data_num=256, gp_weight=10, gpu_num=1, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), save_iter=10000, opt_type="sgd", seed=2020, plot_lim=7, cuda_deterministic=False, use_tensorboard=True):

        self.device = device
        print(f"{'=' * 40}\n[Using device: {self.device}]\n{'=' * 40}")
        self.current_time = Now()
        self.use_tensorboard = use_tensorboard

        if args is None:
            self.seed = seed
            np.random.seed(seed)
            torch.manual_seed(seed=seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            self.video_title = video_title
            self.loss_type = loss_type
            self.method = method
            self.opt_type = opt_type

            self.lr_x = lr_x
            self.lr_y = lr_y
            self.batch_size = batch_size
            self.show_iter = show_iter
            self.save_iter = save_iter
            self.z_dim = z_dim
            self.weight_decay = weight_decay
            self.penalty_x = penalty_x
            self.penalty_y = penalty_y
            self.gp_weight = gp_weight

            self.dataset_name = dataset_name
            self.test_data_num = test_data_num

        else:
            self.video_title, args, self.verboseprint = Get_experiment_name_from_args(args)
            self.args = args
            self.seed = args.seed
            np.random.seed(args.seed)
            torch.manual_seed(seed=args.seed)
            random.seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            random.seed(args.seed)
            if self.args.cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            print('random seed: %d' % args.seed)

            self.loss_type = args.divergence
            self.method = args.method
            self.opt_type = args.opt_type

            self.lr_x = args.g_lr
            self.lr_y = args.d_lr
            self.batch_size = args.batch_size
            self.show_iter = show_iter
            self.plot_iter = args.plot_iter
            self.save_iter = args.save_iter
            self.eval_iter = args.eval_iter
            self.z_dim = args.z_dim
            self.z_std = args.z_std

            self.weight_decay = args.weight_decay
            self.penalty_x = args.g_penalty
            self.penalty_y = args.d_penalty
            self.gp_weight = args.gp

            self.dataset_name = args.data
            self.test_data_num = args.test_data_num
            self.use_tensorboard = args.use_tensorboard

            if self.dataset_name in ["cifar"]:
                output_shape = [3, 32, 32]
            elif self.dataset_name in ["mnist"]:
                output_shape = [1, 28, 28]
            else:
                output_shape = None

            G, D = Get_models(args, output_shape, gpu_num=gpu_num)

        if self.args.opt_type == "sgd":
            self.args.gamma = 0

        print('lr_x: %f \n'
              'lr_y: %f \n'
              'weight decay: %.5f\n'
              'l2 penalty on discriminator: %.5f\n'
              'l2 penalty on generator: %.5f\n'
              'gradient penalty weight: %.2f'
              % (self.lr_x, self.lr_y, self.weight_decay, self.penalty_y, self.penalty_x, self.gp_weight))
        self.log_path = os.path.join(log_folder, f"{self.video_title}_{self.current_time}")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.iteration = 0
        self.epoch = 0
        if self.args.spiky_init:
            self.performing_spiky_init = True
        else:
            self.performing_spiky_init = False

        self.data_fileroot = os.path.join(real_data_folder, self.dataset_name)
        if self.dataset_name == "cifar":
            self.dataset = CIFAR10(root=self.data_fileroot, train=True, download=True, transform=transform)
        elif self.dataset_name == "mnist":
            self.dataset = MNIST(root=self.data_fileroot, train=True, transform=transform, download=True)
            digit_list = [int(ele) for ele in args.mnist_digits]
            print("digit_list", digit_list)
            print("self.dataset.targets\n", self.dataset.targets)
            selected_data_idx_list = [self.dataset.targets == digit for digit in digit_list]
            print("selected_data_idx_list\n", selected_data_idx_list)

            selected_data_idx_final = torch.zeros_like(selected_data_idx_list[0], dtype=torch.int)
            print("selected_data_idx_final\n", selected_data_idx_final)

            for digit, selected_data_idx in zip(digit_list, selected_data_idx_list):
                selected_data_idx_final = selected_data_idx_final + selected_data_idx.type(torch.int32)
                print("selected_data_idx_final", selected_data_idx_final)
                print("selected_data_idx", torch.sum(selected_data_idx))
                print("selected_data_idx_final", torch.sum(selected_data_idx_final))

            # print(self.dataset.targets == 0)
            # print(self.dataset.targets == 1)
            self.dataset.data = self.dataset.data[selected_data_idx_final > 0]
            self.dataset.targets = self.dataset.targets[selected_data_idx_final > 0]
            print(len(self.dataset))
            # print(self.dataset.data[8000].numpy())
            # exit(0)
        elif self.dataset_name == "mog1d":
            self.dataset = Synthetic_Dataset_1D("mog", num_samples=5000, args=self.args)
        else:
            self.dataset = Synthetic_Dataset(self.dataset_name, std=self.args.mog_std, scale=self.args.mog_scale, sample_per_mode=1000)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

        self.G = G.to(self.device)
        self.D = D.to(self.device)



        if gpu_num > 1:
            self.G = nn.DataParallel(self.G, list(range(gpu_num)))
            self.D = nn.DataParallel(self.D, list(range(gpu_num)))

        if self.dataset_name in ["cifar"]:
            self.G.apply(weights_init_g)
            self.D.apply(weights_init_d)
        elif self.dataset_name in ["mnist"]:
            self.G.apply(DCGAN_weights_init)
            self.D.apply(DCGAN_weights_init)
        else:
            pass


        if self.args.lazy and self.args.data in ["mnist", "cifar"]:
            self.G_0 = copy.deepcopy(self.G)
            self.D_0 = copy.deepcopy(self.D)
        else:
            self.G_0 = None
            self.D_0 = None

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_regression = nn.MSELoss()

        noise_shape = (self.test_data_num, self.z_dim)
        self.fixed_noise = torch.randn(noise_shape, device=device) * self.args.z_std
        self.exp_start_time = time.time()



        """ Real data """
        if self.dataset_name in ["mnist", "cifar"]:
            x_real_np = np.zeros((self.batch_size * 10, 2))
        elif self.dataset_name == "mog1d":
            x_real_np = self.dataset.all_samples
            x_real_label = self.dataset.all_labels
        else:
            x_real_np = self.dataset.sample_points
            x_real_label = self.dataset.ith_center

        if self.dataset_name in ["mnist", "cifar"]:
            self.deepVisuals = deepviz_real.DeepVisuals_real(args=self.args, name=self.video_title, z_test=self.fixed_noise.cpu().detach().numpy())
            self.deepVisuals.attr["dataset"] = self.dataset
            print("\n\nusing deepviz_real\n")
        elif self.dataset_name == "mog1d":
            self.z_linspace = np.linspace(- 4 * self.args.z_std, 4 * self.args.z_std, 201)[:, np.newaxis]
            self.x_linspace = np.linspace(self.dataset.MG.Get_bounding_box()[0], self.dataset.MG.Get_bounding_box()[1], 201)[:, np.newaxis]
            self.deepVisuals = DeepVisuals_1D(args=self.args, name=self.video_title, z_test=self.fixed_noise.cpu().detach().numpy(), dataset=self.dataset, x_real=x_real_np.ravel(), x_real_label=x_real_label, z_linspace=self.z_linspace.ravel(), x_linspace=self.x_linspace.ravel())
            self.spiky_examples_linspace = np.concatenate([(self.dataset.MG.Get_bounding_box()[1] - self.dataset.MG.Get_bounding_box()[0]) / (-3 * self.args.z_std) * self.z_linspace[self.z_linspace < 0] + self.dataset.MG.Get_bounding_box()[0], (self.dataset.MG.Get_bounding_box()[1] - self.dataset.MG.Get_bounding_box()[0]) / (3 * self.args.z_std) * self.z_linspace[self.z_linspace >= 0] + self.dataset.MG.Get_bounding_box()[0]])[:, np.newaxis]
        else:
            """ Initialize deep viz """
            self.bbox_x = np.array([-self.args.plot_lim_x, self.args.plot_lim_x, -self.args.plot_lim_x, self.args.plot_lim_x])
            self.bbox_z = np.array([-self.args.plot_lim_z, self.args.plot_lim_z, -self.args.plot_lim_z, self.args.plot_lim_z])
            self.xx_z, self.yy_z = np.mgrid[self.bbox_z[0]:self.bbox_z[1]:50j, self.bbox_z[2]:self.bbox_z[3]:50j]
            self.z_mesh_vec = np.asarray(np.vstack([self.xx_z.ravel(), self.yy_z.ravel()]).T, np.float32)
            """ D grid """
            self.xx_D, self.yy_D = np.mgrid[self.bbox_x[0]:self.bbox_x[1]:50j, self.bbox_x[2]:self.bbox_x[3]:50j]
            self.positions_D = np.vstack([self.xx_D.ravel(), self.yy_D.ravel()]).T
            print(f"positions_D.shape: {self.positions_D.shape}")
            self.deepVisuals = DeepVisuals_2D(args=self.args, z_mesh=self.z_mesh_vec, x_real=x_real_np, xx_D=self.xx_D, yy_D=self.yy_D, xx_z=self.xx_z, yy_z=self.yy_z, bbox_x=self.bbox_x, bbox_z=self.bbox_z, name=self.video_title, z_test=self.fixed_noise.cpu().detach().numpy(), dataset=self.dataset)


        self.alpha_list = np.arange(0, 1.55, 0.05)
        self.var_shape_list_G = [param.shape for param in self.G.parameters()]
        self.var_shape_list_D = [param.shape for param in self.D.parameters()]

        # self.deepVisuals.Init_figure()

        self.handle_dict = {}
        self.short_title = f"{self.dataset_name}_{self.loss_type}_{self.opt_type}_{self.method}"
        self.handle_dict["show_info_txt"] = open(os.path.join(self.log_path, f"{self.short_title}.txt"), "w")
        self.handle_dict["show_info_csv"] = open(os.path.join(self.log_path, f"{self.short_title}.csv"), "w")


    def load(self, path):
        checkpoints = torch.load(path)
        self.D.load_state_dict(checkpoints['D'])
        self.G.load_state_dict(checkpoints['G'])

    def generate_data(self, z=None):
        if z is None:
            z = torch.randn((self.batch_size, self.z_dim), device=self.device) * self.args.z_std

        # print("G_output.shape", G_output.shape)

        if self.args.brenier:
            z.requires_grad = True
            G_output = self.G(z.float())

            # print(autograd.grad(G_output[0, :], z, create_graph=True, retain_graph=True, allow_unused=True))
            # print(autograd.grad(G_output[0, :], z, create_graph=True, retain_graph=True, allow_unused=True)[0])
            # print(autograd.grad(G_output[0, :], z, create_graph=True, retain_graph=True, allow_unused=True)[0][0, :])
            # print(autograd.grad(G_output[0, :], z, create_graph=True, retain_graph=True, allow_unused=True)[0][0, :].unsqueeze(0))

            dU_dz = torch.cat([autograd.grad(G_output[i, :], z, create_graph=True, retain_graph=True, allow_unused=True)[0][i, :].unsqueeze(0) for i in range(G_output.shape[0])], dim=0)
            fake_data = dU_dz
            # print("fake_data.shape", fake_data.shape)
        else:
            G_output = self.G(z.float())
            fake_data = G_output

        if self.args.lazy and self.args.data in ["mnist", "cifar"]:
            if self.G_0 is not None:
                # print("np.sum(G(z)) before", np.sum(fake_data.detach().cpu().numpy().ravel()))
                fake_data = fake_data - self.G_0(z.float())
                # print("np.sum(G(z)) after", np.sum(fake_data.detach().cpu().numpy().ravel()))

        return fake_data

    def discriminate_data(self, x):
        # print(x.shape)
        score = self.D(x)
        if self.args.lazy and self.args.data in ["mnist", "cifar"]:
            if self.D_0 is not None:
                # print("np.sum(D(x)) before", np.sum(score.detach().cpu().numpy().ravel()))
                score = score - self.D_0(x)
                # print("np.sum(D(x)) after", np.sum(score.detach().cpu().numpy().ravel()))

        return score


    def gradient_penalty(self, real_x, fake_x):
        alpha = torch.randn((self.batch_size, 1, 1, 1), device=self.device)
        alpha = alpha.expand_as(real_x)
        interploted = alpha * real_x.data + (1.0 - alpha) * fake_x.data
        interploted.requires_grad = True
        interploted_d = self.discriminate_data(interploted)
        gradients = \
        torch.autograd.grad(outputs=interploted_d, inputs=interploted, grad_outputs=torch.ones(interploted_d.size(), device=self.device), create_graph=True,
                            retain_graph=True)[0]
        gradients = gradients.view(self.batch_size, -1)
        if self.use_tensorboard:
            self.writer.add_scalars('Gradients', {'D gradient L2norm': gradients.norm(p=2, dim=1).mean().item()}, self.iteration)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1.0) ** 2).mean()


    def get_inception_score(self, batch_num, splits_num=10):
        net = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        resize_module = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(
            self.device)
        preds = np.zeros((self.batch_size * batch_num, 1000))
        for e in range(batch_num):
            imgs = resize_module(self.generate_data())
            with torch.no_grad():
                pred = F.softmax(net(imgs), dim=1).data.cpu().numpy()
            preds[e * self.batch_size: e * self.batch_size + self.batch_size] = pred
        split_score = []
        chunk_size = preds.shape[0] // splits_num
        for k in range(splits_num):
            pred_chunk = preds[k * chunk_size: k * chunk_size + chunk_size, :]
            kl_score = pred_chunk * (
                    np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_score = np.mean(np.sum(kl_score, 1))
            split_score.append(np.exp(kl_score))
        return np.mean(split_score), np.std(split_score)

    def l2penalty(self):
        p_d = 0
        p_g = 0
        if self.penalty_y != 0:
            for p in self.D.parameters():
                p_d += torch.dot(p.view(-1), p.view(-1))
        if self.penalty_x != 0:
            for p in self.G.parameters():
                p_g += torch.dot(p.view(-1), p.view(-1))
        return self.penalty_y * p_d - self.penalty_x * p_g

    def load_checkpoint(self, chkpt_path, count, load_d=False, load_g=False):
        self.iteration = count
        checkpoint = torch.load(chkpt_path)
        if load_d:
            self.D.load_state_dict(checkpoint['D'])
            print('load Discriminator from %s' % chkpt_path)
        if load_g:
            self.G.load_state_dict(checkpoint['G'])
            print('load Generator from %s' % chkpt_path)
        # print('load models from %s' % chkpt_path)

    def save_checkpoint(self, pathname=None):
        if pathname is None:
            pathname = self.video_title

        # chk_name = './checkpoints/%.5f%s-%.4f-%.4f/' % (self.penalty_y, dataset, self.lr_x, self.lr_y)
        chk_pathname = os.path.join("Checkpoints", f"{pathname}_{self.current_time}")
        if not os.path.exists(chk_pathname):
            os.makedirs(chk_pathname)

        chk_filename = os.path.join(chk_pathname, f"epoch-{self.epoch}_iteration-{self.iteration}.pth")
        torch.save({
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            "args": self.args
        }, chk_filename)
        print('save models at %s' % chk_filename)

    def writer_init(self):
        self.writer = SummaryWriter(logdir=self.log_path)
        fieldnames = ['iter', 'is_mean', 'is_std', 'time', 'gradient calls']
        f = open(os.path.join(self.log_path, 'inception_score.csv'), 'w')
        self.iswriter = csv.DictWriter(f, fieldnames)
        self.show_info_writer = csv.DictWriter(self.handle_dict["show_info_csv"], ["epoch", "iter", "batch", "total_time", "training_time", "grad_corr_time", "plot_time", "eig_time", "is_time", "is_mean", "is_std", "loss_G_tot", "loss_D_tot", "loss_G", "loss_D"])

    def show_info(self, timer, D_loss=None, G_loss=None):
        if self.dataset_name in ["mnist", "cifar"]:
            z = self.fixed_noise[:(self.test_data_num // 4), ...]
            fake_data = self.generate_data(z).detach()
            fake_data = detransform(fake_data)

            if self.use_tensorboard:
                self.writer.add_images('Generated images', fake_data, global_step=self.iteration, dataformats='NCHW')
        # print(type(e))
        # print('Fail to plot')

    def print_info(self, timer, D_loss=None, G_loss=None):
        if G_loss is not None:
            print('Iter :%d , D_loss: %.5f, G_loss: %.5f, time: %.3fs' % (
                self.iteration, D_loss.item(), G_loss.item(), timer))
        else:
            print('Iter : %d, Loss: %.5f, time: %.3fs' % (self.iteration, D_loss.item(), timer))
        fake_data = self.G(self.fixed_noise).detach()
        fake_data = detransform(fake_data)
        vutils.save_image(fake_data, os.path.join(self.log_path, "Figs", f"iter-{self.iteration}.png"))

    def plot_d(self, d_real, d_fake):
        if self.use_tensorboard:
            self.writer.add_scalars('Discriminator output',
                                {'real': d_real.mean().item(), 'fake': d_fake.mean().item()},
                                self.iteration)

    def plot_grad(self, update_raw_norm_x, update_raw_norm_y, grad_corr_norm_x=None, grad_corr_norm_y=None, update_norm_x=None, update_norm_y=None):

        # print("update_raw_norm_y", update_raw_norm_y, "update_raw_norm_x", update_raw_norm_x)
        if update_raw_norm_y is not None and update_raw_norm_x is not None:
            self.writer.add_scalars('Delta', {'update_raw_norm_y': update_raw_norm_y, 'update_raw_norm_x': update_raw_norm_x}, self.iteration)
        if grad_corr_norm_y is not None and grad_corr_norm_x is not None:
            self.writer.add_scalars('Delta', {'grad_corr_norm_y': grad_corr_norm_y, 'grad_corr_norm_x': grad_corr_norm_x}, self.iteration)
        if update_norm_y is not None and update_norm_x is not None:
            self.writer.add_scalars('Delta', {'update_norm_y': update_norm_y, 'update_norm_x': update_norm_x}, self.iteration)

    def plot_param(self, loss_G, loss_D, loss_G_tot, loss_D_tot):
        self.writer.add_scalars('Loss', {'loss_G': loss_G.item()}, self.iteration)
        self.writer.add_scalars('Loss', {'loss_D': loss_D.item()}, self.iteration)
        self.writer.add_scalars('Loss', {'loss_G_tot': loss_G_tot.item()}, self.iteration)
        self.writer.add_scalars('Loss', {'loss_D_tot': loss_D_tot.item()}, self.iteration)

        wg = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]), p=2).item()
        wd = torch.norm(torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]), p=2).item()

        self.writer.add_scalars('Weights', {'params_norm_x': wg, 'params_norm_y': wd}, self.iteration)

    def Get_optimizer(self, collect_info=False):
        if self.method == 'BCGD':
            return BCGD(max_params=self.G.parameters(), min_params=self.D.parameters(), lr_max=self.lr_x, lr_min=self.lr_y, device=self.device, solve_x=False,
                        collect_info=collect_info)
        elif self.method == 'ACGD':
            return ACGD(max_params=self.G.parameters(), min_params=self.D.parameters(), lr_max=self.lr_x, lr_min=self.lr_y, device=self.device, solve_x=False, collect_info=collect_info)
        elif self.method in ["simgd", "fr", "fr2", "fr3", "frm", "gdn", "nfr", "cgd"]:
            follow_x = False
            follow_y = False
            maximin = False
            cgd = False
            if self.method in ["cgd"]:
                cgd = True

            if self.method in ["fr", "fr3"]:
                follow_y = True
            if self.method in ["fr2", "fr3"]:
                follow_x = True
            if self.method in ["frm"]:
                maximin = True
            if self.opt_type in ["gdn", "nfr"]:
                newton_y = True
            else:
                newton_y = False

            if self.opt_type in ["rmsprop", "adam"]:
                adapt_x = True
                adapt_y = True

                if self.args.adapt_x:
                    adapt_y = False
                if self.args.adapt_y:
                    adapt_x = False

            else:
                adapt_x = False
                adapt_y = False


            if self.opt_type in ["adam"]:
                use_momentum = True
            else:
                use_momentum = False

            return FR(params_x=list(params for params in self.G.parameters() if params.requires_grad), params_y=list(params for params in self.D.parameters() if params.requires_grad), lr_x=self.lr_x, lr_y=self.lr_y, device=self.device, collect_info=collect_info, follow_x=follow_x, follow_y=follow_y, maximin=maximin, adapt_x=adapt_x, adapt_y=adapt_y, hessian_reg_x=self.args.g_hessian_reg, hessian_reg_y=self.args.d_hessian_reg, calculate_eig_vals=self.args.eigs, use_momentum=use_momentum, gamma_x=self.args.g_lr_decay, gamma_y=self.args.d_lr_decay, momentum=self.args.momentum, beta=self.args.gamma, newton_y=newton_y, cgd=cgd, zeta=self.args.zeta, rmsprop_init_1=self.args.rmsprop_init_1)

        else:
            raise NotImplementedError

    def Get_loss(self, d_real, d_fake, real_x, fake_x):
        if self.performing_spiky_init:
            loss_G = self.criterion_regression(fake_x, torch.from_numpy(self.spiky_examples_linspace).to(self.device).float())
            loss_D = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
        else:
            if self.loss_type == 'JS':
                if self.args.data in ["mnist", "cifar"]:
                    alpha_G = self.args.alpha_mobility
                    alpha_D = self.args.alpha_mobility_D
                else:
                    alpha_G = alpha_D = 1

                loss_D = (self.criterion(alpha_D * d_real, torch.ones(d_real.shape, device=self.device)) + self.criterion(alpha_D * d_fake, torch.zeros(d_fake.shape, device=self.device))) / alpha_D ** 2

                if alpha_G != alpha_D:
                    loss_G = -self.criterion(alpha_G * d_fake, torch.zeros(d_fake.shape, device=self.device)) / alpha_G ** 2
                else:
                    loss_G = -loss_D

            elif self.loss_type == "wgan":
                loss_D = d_fake.mean() - d_real.mean()
                loss_G = -loss_D
                if self.gp_weight != 0:
                    loss_D = loss_D + self.gradient_penalty(real_x=real_x, fake_x=fake_x)
            elif self.loss_type == 'NS':
                loss_D = self.criterion(d_real, torch.ones(d_real.shape, device=self.device)) + self.criterion(d_fake, torch.zeros(d_fake.shape, device=self.device))
                loss_G = self.criterion(d_fake, torch.ones(d_fake.shape, device=self.device))
            else:
                raise NotImplementedError

        if hasattr(self.args, "mono"):
            if self.args.mono:
                loss_G += self.mono_loss_weight * self.G.generate_regularizer()

        return loss_G, loss_D, loss_G - self.l2penalty(), loss_D + self.l2penalty()

    def train(self, epoch_num=-1, iteration_num=-1, collect_info=True, dataname=None, logname=None):

        if dataname is None:
            dataname = self.dataset_name
        if logname is None:
            logname = self.dataset_name
        if epoch_num <= 0:
            epoch_num = int(1e8)
        else:
            iteration_num = int(1e12)

        timer = time.time()
        start = time.time()
        is_lsit = []
        if collect_info:
            self.writer_init()
            self.iswriter.writeheader()
            self.show_info_writer.writeheader()

        self.optimizer = self.Get_optimizer(collect_info=collect_info)

        iter_num = -1
        total_time = 0.
        is_mean = 0.
        is_std = 0.
        is_time = 0

        iteration_time_list = []
        forwardpass_time_list = []
        backprop_time_list = []
        update_time_list = []
        tot_is_time_list = []
        tot_tensorboard_time_list = []
        tot_eig_time_list = []
        tot_plot_time_list = []

        if self.performing_spiky_init:
            spiky_init_timer = Timer()
            spiky_optimizer = torch.optim.Adam(params=self.G.parameters())

            for i in range(999999):
                z = torch.from_numpy(self.z_linspace).to(self.device)
                fake_x = self.generate_data(z.float())
                loss_G_reg = self.criterion_regression(fake_x, torch.from_numpy(self.spiky_examples_linspace).to(self.device).float())
                if i % 1000 == 0:
                    print(f"[Spiky init iter {i}], loss {loss_G_reg:.4f}, time {spiky_init_timer.Print(print_time=False):.3f}")
                if loss_G_reg.item() < self.args.spiky_init_loss_thres:
                    self.performing_spiky_init = False
                    print(f"[Spiky init complete at iter {i}], loss {loss_G_reg:.4f}, time {spiky_init_timer.Print(print_time=False):.3f}")
                    break

                spiky_optimizer.zero_grad()
                loss_G_reg.backward()
                spiky_optimizer.step()

            spiky_init_timer.Print(msg="[Spiky init]")

        print(f"epoch_num {epoch_num}, iteration_num {iteration_num}")
        for e in range(epoch_num):
            for batch_id, real_x in enumerate(self.dataloader):
                if self.iteration == 0 and self.args.show_init_params:
                    print("G params", torch.cat([_.flatten() for _ in self.G.parameters()]).view(-1, 1))
                    print("D params", torch.cat([_.flatten() for _ in self.D.parameters()]).view(-1, 1))
                    iteration_num = 0

                # if self.iteration % self.save_iter == 0:
                #     self.save_checkpoint()
                iteration_timer = Timer()
                forwardpass_timer = Timer()

                real_x = real_x[0].to(self.device).float()
                d_real = self.discriminate_data(real_x)

                if self.performing_spiky_init:
                    z = torch.from_numpy(self.z_linspace).to(self.device)
                    lr_x = 1e-5
                    lr_y = 0
                else:
                    z = torch.randn((self.batch_size, self.z_dim), device=self.device) * self.args.z_std
                    lr_x = None
                    lr_y = None
                fake_x = self.generate_data(z.float())
                # print("fake_x/shape", fake_x.shape)
                d_fake = self.discriminate_data(fake_x)

                loss_G, loss_D, loss_G_tot, loss_D_tot = self.Get_loss(d_real, d_fake, real_x, fake_x)
                forwardpass_time_list.append(forwardpass_timer.Print(print_time=False))
                if loss_G.item() < self.args.spiky_init_loss_thres:
                    self.performing_spiky_init = False

                backprop_timer = Timer()
                self.optimizer.zero_grad()
                self.optimizer.backward(loss_G=loss_G_tot, loss_D=loss_D_tot, lr_x=lr_x, lr_y=lr_y)
                backprop_time_list.append(backprop_timer.Print(print_time=False))

                print_locator = False
                if self.iteration % self.eval_iter == 0 and self.dataset_name in ["mnist", "cifar"]:
                    is_timer = Timer()
                    if print_locator: print("----- IS -----")
                    if self.dataset_name in ["cifar"]:
                        is_mean, is_std = self.get_inception_score(batch_num=500)
                    elif self.dataset_name in ["mnist"]:
                        fake_x_np = fake_x.cpu().detach().numpy()
                        # print(f"fake_x_np shape: {fake_x_np.shape}")
                        is_mean, is_std = MNIST_IS(self.args, fake_x_np, device=self.device)
                    else:
                        raise NotImplementedError
                    is_lsit.append(is_mean)
                    is_time = is_timer.Print(print_time=False)
                    tot_is_time_list.append(is_time)
                    # print(f"[Epoch {self.epoch}, Iter {self.iteration}, Batch {batch_id}, Time {self.optimizer.culmulative_training_time:.3f}, Grad_corr {self.optimizer.culmulative_grad_corr_time:.3f}] Inception score: mean: {is_mean:.4f}, std: {is_std:.4f}")
                    # print(f'{self.optimizer.info}')
                    if print_locator: print("----- IS completed -----")

                    if collect_info:
                        self.iswriter.writerow({'iter': self.iteration, 'is_mean': is_mean, 'is_std': is_std, 'time': time.time() - start, 'gradient calls': 2 * iter_num + 4})
                        self.writer.add_scalars("Evaluation", {'is_mean': is_mean, 'is_std': is_std}, self.iteration)

                if collect_info:
                    tensorboard_timer = Timer()

                    if print_locator: print("----- info -----")

                    grad_raw_norm_x, grad_raw_norm_y, grad_corr_norm_x, grad_corr_norm_y, \
                    update_tot_norm_x, update_tot_norm_y, training_time, gc_time, \
                    iter_num, lr_x_actual, lr_y_actual = self.optimizer.get_info()
                    if self.use_tensorboard:
                        self.plot_param(loss_G, loss_D, loss_G_tot, loss_D_tot)
                    # print(f"------ Iteration={self.iteration}, grad_raw_norm_x={grad_raw_norm_x:.3e}, grad_raw_norm_y={grad_raw_norm_y:.3e}, grad_corr_norm_x={grad_corr_norm_x:.3e}, grad_corr_norm_y={grad_corr_norm_y:.3e}, update_tot_norm_x={update_tot_norm_x:.3e}, update_tot_norm_y={update_tot_norm_y:.3e}, training_time={training_time:.3e}, gc_time={gc_time:.3e}, lr_x_actual={lr_x_actual:.3e}, lr_y_actual={lr_y_actual:.3e}")
                        self.writer.add_scalars('Observables', {'grad_raw_norm_x': grad_raw_norm_x, 'grad_raw_norm_y': grad_raw_norm_y, 'grad_corr_norm_x': grad_corr_norm_x, 'grad_corr_norm_y': grad_corr_norm_y, 'update_tot_norm_x': update_tot_norm_x, 'update_tot_norm_y': update_tot_norm_y}, self.iteration)
                        self.writer.add_scalars('Time cost(s)', {'gc_time': gc_time, 'training_time': training_time}, self.iteration)

                    if print_locator: print("----- info completed -----")

                    tot_tensorboard_time_list.append(tensorboard_timer.Print(print_time=False))

                    if self.iteration % self.plot_iter == 0:
                        eig_timer = Timer()
                        if print_locator: print("----- eigs -----")

                        if self.save_iter > 0:
                            if self.iteration % self.save_iter == 0:
                                eigs = self.args.eigs
                            else:
                                eigs = False
                        else:
                            eigs = False

                        eig_vals_Hxx_f, eig_vals_Hyy_g, eig_vals_Hxx_f_reg, eig_vals_Hyy_g_reg, eig_vals_J, eig_vals_Hxx_f_Schur, eig_vals_Hyy_g_Schur, eig_calculation_time = self.optimizer.Calculate_eig_vals(loss_G_tot, loss_D_tot, eigs=eigs)
                        if self.use_tensorboard:
                            tensorboard_timer = Timer()
                            self.writer.add_scalars('Time cost(s)', {'eig_calculation_time': eig_calculation_time}, self.iteration)
                            tot_tensorboard_time_list.append(tensorboard_timer.Print(print_time=False))

                        if print_locator: print("----- eigs completed -----")
                        if print_locator: print("----- plot -----")
                        tot_eig_time_list.append(eig_timer.Print(print_time=False))


                        plot_timer = Timer()
                        if collect_info:
                            self.show_info(D_loss=loss_D_tot, timer=total_time)
                            self.plot_d(d_real, d_fake)
                        else:
                            self.print_info(D_loss=loss_D_tot, timer=time.time() - timer)

                        timer = time.time()
                        total_time = time.time() - self.exp_start_time
                        if self.use_tensorboard:
                            tensorboard_timer = Timer()
                            self.writer.add_scalars('Time cost(s)', {'gc_time': gc_time, 'training_time': training_time, 'total_time': total_time}, self.iteration)
                            tot_tensorboard_time_list.append(tensorboard_timer.Print(print_time=False))

                        fake_x_test = self.generate_data(self.fixed_noise)

                        state_dict_G = None
                        state_dict_D = None

                        if self.args.save_param:
                            if self.iteration % self.plot_iter == 0:
                                state_dict_G = self.G.state_dict()
                                state_dict_D = self.D.state_dict()

                        if self.iteration == iteration_num - 1 or self.iteration == 0:
                            state_dict_G = self.G.state_dict()
                            state_dict_D = self.D.state_dict()
                            print("model saved")

                        idd = {"iter": self.iteration, "loss_G": loss_G, "loss_D": loss_D, "loss_G_tot": loss_G_tot, "loss_D_tot": loss_D_tot, "grad_raw_norm_x": grad_raw_norm_x, "grad_raw_norm_y": grad_raw_norm_y, "jacobians_mat": None, "eig_vals_Hxx_f": eig_vals_Hxx_f, "eig_vals_Hyy_g": eig_vals_Hyy_g, "cumul_training_time": self.optimizer.culmulative_training_time, "eig_vals_Hxx_f_Schur": eig_vals_Hxx_f_Schur, "eig_vals_Hyy_g_Schur": eig_vals_Hyy_g_Schur, "update_tot_norm_x": update_tot_norm_x, "update_tot_norm_y": update_tot_norm_y, "grad_corr_norm_x": grad_corr_norm_x, "grad_corr_norm_y": grad_corr_norm_y," use_x_corr": None, "corr_norm_x_ma": None," corr_rel_norm_x": None, "corr_rel_norm_y": None, "corr_rel_norm_x_ma": None, "corr_norm_y_ma": None, "corr_rel_norm_y_ma": None, "total_time": total_time, "state_dict_G": state_dict_G, "state_dict_D": state_dict_D, "G": self.G, "D": self.D}

                        if self.dataset_name in ["mnist", "cifar"]:
                            if self.args.show_init_params:
                                x_out = None
                            else:
                                x_out = detransform(fake_x_test.detach()).cpu().numpy()
                            idd.update({"x_out": x_out, "inception_score": (is_mean, is_std)})
                        elif self.dataset_name == "mog1d":
                            G_z_linspace = self.generate_data(torch.from_numpy(self.z_linspace).to(self.device).float()).cpu().detach().numpy().ravel()
                            # G_z_linspace = self.G(torch.from_numpy(self.z_linspace).to(self.device).float()).cpu().detach().numpy().ravel()
                            D_output_linspace = self.discriminate_data(torch.from_numpy(self.x_linspace).to(self.device).float()).cpu().detach().numpy().ravel()
                            idd.update({"x_out": fake_x_test.cpu().detach().numpy().ravel(), "G_z_linspace": G_z_linspace, "D_output_linspace": D_output_linspace, "performing_spiky_init": self.performing_spiky_init})
                        else:
                            x_fake_mesh_vec_out = self.generate_data(torch.from_numpy(self.z_mesh_vec).to(self.device).float()).cpu().detach().numpy()
                            # x_fake_mesh_vec_out = self.G(torch.from_numpy(self.z_mesh_vec).to(self.device).float()).cpu().detach().numpy()
                            D_output_linspace = self.discriminate_data(torch.from_numpy(self.positions_D).to(self.device).float()).cpu().detach().numpy()
                            D_output_grid = np.reshape(D_output_linspace.T, self.xx_D.shape)
                            x_fake_1_grid = np.reshape(x_fake_mesh_vec_out[:, 0].T, self.xx_z.shape)
                            x_fake_2_grid = np.reshape(x_fake_mesh_vec_out[:, 1].T, self.xx_z.shape)
                            idd.update({"x_out": fake_x_test.cpu().detach().numpy(), "D_output_grid": D_output_grid, "z_mesh_vec": self.z_mesh_vec, "x_fake_mesh_vec_out": x_fake_mesh_vec_out, "x_fake_1_grid": x_fake_1_grid, "x_fake_2_grid": x_fake_2_grid})

                        self.deepVisuals.Plot_step(idd)
                        # if self.args.save_param:
                        #     np.savetxt(os.path.join(self.log_path, "params_x.csv"), torch.cat([p.contiguous().view(-1) for p in self.G.parameters()]).detach().cpu().numpy(), delimiter=',')
                        #     np.savetxt(os.path.join(self.log_path, "params_y.csv"), torch.cat([p.contiguous().view(-1) for p in self.D.parameters()]).detach().cpu().numpy(), delimiter=',')
                        plot_time = plot_timer.Print(print_time=False)
                        tot_plot_time_list.append(plot_time)

                        show_info = f"[Epoch {self.epoch}, Iter {self.iteration}, Batch {batch_id}] Total time {total_time:.3f}, Training time {self.optimizer.culmulative_training_time:.3f}, Grad_corr time {self.optimizer.culmulative_grad_corr_time:.3f}, Plot time {plot_time:.3f}, Eig time: {eig_calculation_time:.3f}, IS time: {is_time:.3f} [loss_G_tot: {loss_G_tot.item():.3f}, loss_D_tot: {loss_D_tot.item():.3f}, loss_G: {loss_G.item():.3f}, loss_D: {loss_D.item():.3f}, is_mean: {is_mean:.4f}, is_std: {is_std:.4f}], spiky_init: {self.performing_spiky_init}"
                        print(show_info)
                        if self.iteration > 0:
                            print(f"[Timer tot] iter: {np.sum(iteration_time_list):.3f}, forward: {np.sum(forwardpass_time_list):.3f}, backward: {np.sum(backprop_time_list):.3f}, update: {np.sum(update_time_list):.3f}, IS: {np.sum(tot_is_time_list):.3f}, tensorboard: {np.sum(tot_tensorboard_time_list):.3f}, eig: {np.sum(tot_eig_time_list):.3f}, plot: {np.sum(tot_plot_time_list):.3f}")
                            print(f"[Timer avg] iter: {np.mean(iteration_time_list):.3f}, forward: {np.mean(forwardpass_time_list):.3f}, backward: {np.mean(backprop_time_list):.3f}, update: {np.mean(update_time_list):.3f}, IS: {np.mean(tot_is_time_list):.3f}, tensorboard: {np.mean(tot_tensorboard_time_list):.3f}, eig: {np.mean(tot_eig_time_list):.3f}, plot: {np.mean(tot_plot_time_list):.3f}")

                        with open(os.path.join(self.log_path, f"{self.short_title}.txt"), "a") as f:
                            f.write(f"{show_info}\n")

                        if self.use_tensorboard:
                            tensorboard_timer = Timer()
                            with open(os.path.join(self.log_path, f"{self.short_title}.csv"), "a"):
                                self.show_info_writer.writerow({"epoch": self.epoch, "iter": self.iteration, "batch": batch_id, "total_time": total_time, "training_time": self.optimizer.culmulative_training_time, "grad_corr_time": self.optimizer.culmulative_grad_corr_time, "plot_time": plot_time, "eig_time": eig_calculation_time, "is_time": is_time, "is_mean": is_mean, "is_std": is_std, "loss_G_tot": loss_G_tot.item(), "loss_D_tot": loss_D_tot.item(), "loss_G": loss_G.item(), "loss_D": loss_D.item()})
                            tot_tensorboard_time_list.append(tensorboard_timer.Print(print_time=False))

                        if print_locator: print("----- plot completed -----")


                self.iteration += 1
                update_timer = Timer()
                self.optimizer.step()
                update_time_list.append(update_timer.Print(print_time=False))

                iteration_time_list.append(iteration_timer.Print(print_time=False))

                if iteration_num >= 0 and self.iteration > iteration_num:
                    break
            else:
                """ Only excuted if the inner loop didn't break """
                self.epoch += 1
                continue
            break


        # self.save_checkpoint()
        len_list = len(is_lsit)
        plt.plot(np.arange(len_list) * 5, is_lsit)
        plt.ylabel('Inception score')
        plt.xlabel('iteration(k)')
        plt.savefig(os.path.join(self.log_path, "IS_plot.png"), dpi=400)
        # plt.show()
        self.deepVisuals.handle.close()
        self.handle_dict["show_info_txt"].close()
        self.writer.close()

def Train_GAN(args):
    gan = GAN(args=args)
    gan.train(epoch_num=args.epoch, iteration_num=args.iteration + 1)
    print(gan.deepVisuals.attr['name'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=-1, help="number of iterations of training")
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--test_data_num", type=int, default=256, help="number of test data")
    parser.add_argument("--g_lr", type=float, default=0.002, help="generator learning rate")
    parser.add_argument("--g_lr_warmup", type=float, default=0.00000001, help="generator warmup learning rate")
    parser.add_argument("--g_lr_decay", type=float, default=1., help="generator learning rate decay")
    parser.add_argument("--d_lr", type=float, default=0.0005, help="discriminator learning rate")
    parser.add_argument("--d_lr_warmup", type=float, default=0.00000001, help="discriminator warmup learning rate")
    parser.add_argument("--d_lr_decay", type=float, default=1., help="discriminator learning rate decay")
    parser.add_argument("--warmup_iter", type=int, default=1, help="number of warmup iterations")
    parser.add_argument("--lr", type=float, default=0.001, help="discriminator learning rate")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0, help="weight decay factor")
    parser.add_argument("--gp", type=float, default=0, help="gradient penalty")
    parser.add_argument("--z_dim", type=int, default=-1, help="dimension of latent noise")
    parser.add_argument("--z_std", type=float, default=1, help="std of latent noise")
    parser.add_argument("--mog_std", type=float, default=0.1, help="std of mog")
    parser.add_argument("--mog_scale", type=float, default=1, help="scale of mog")
    parser.add_argument("--g_hidden", type=int, default=32, help="dimension of hidden units")
    parser.add_argument("--d_hidden", type=int, default=32, help="dimension of hidden units")
    parser.add_argument("--g_layers", type=int, default=2, help="num of hidden layer")
    parser.add_argument("--d_layers", type=int, default=2, help="num of hidden layer")
    parser.add_argument("--d_bias_scale", type=float, default=1, help="d_bias_scale")
    parser.add_argument("--x_dim", type=int, default=2, help="data dimension")
    parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
    parser.add_argument("--arch", type=str, default="default", help="which gan architecture")
    parser.add_argument("--data", type=str, default="cifar", help="which dataset") # mog1d, grid5, ...
    parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
    parser.add_argument("--g_act", type=str, default="relu", help="which activation function for gen")  # elu, relu, tanh
    parser.add_argument("--d_act", type=str, default="relu", help="which activation function for disc")  # elu, relu, tanh
    parser.add_argument("--divergence", type=str, default="JS", help="which activation function for disc")  # NS, JS, indicator, wgan
    parser.add_argument("--method", type=str, default="fr", help="which optimization regularization method to use")
    parser.add_argument("--opt_type", type=str, default="sgd", help="which optimization method to use")
    parser.add_argument("--reg_param", type=float, default=5.0, help="reg param for JARE")
    parser.add_argument("--momentum", type=float, default=0.5, help="momentum coefficient for the whole system")
    parser.add_argument("--gamma", type=float, default=0.999, help="gamma for adaptive learning rate")
    parser.add_argument("--zeta", type=float, default=1, help="learning rate adaptivity")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--vanilla_gradient", "--vg", action='store_true', help="whether to remove preconditioning")
    parser.add_argument("--pre", action='store_true', help="whether to use preconditioning")
    parser.add_argument("--follow_ridge", action='store_false', help="whether to use  follow-the-ridge")
    parser.add_argument("--inner_iter", type=int, default=5, help="conjugate gradient or gradient descent steps")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--damping", type=float, default=1.0, help="damping term for CG")
    parser.add_argument("--adapt_damping", action='store_false', help="whether to adapt damping")
    parser.add_argument("--real_time_video", action='store_false', help="whether to play video in real time")
    parser.add_argument("--plot_iter", type=int, default=150, help="number of iter to plot")
    parser.add_argument("--save_iter", type=int, default=-1, help="number of iter to save")
    parser.add_argument("--eval_iter", type=int, default=-1, help="number of iter to eval")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--spanning_init", action='store_true', help="whether to spanning_init")
    parser.add_argument("--si_lr", type=float, default=0.01, help="spanning_init learning rate")
    parser.add_argument("--degen_z", action='store_true', help="whether to reduce dim_z")
    parser.add_argument("--eigs", action='store_true', help="whether to calculate eig vals")
    parser.add_argument("--conj_grad", action='store_true', help="whether to conj_grad")
    parser.add_argument("--fair", action='store_true', help="whether to use a fair loss")
    parser.add_argument("--conj_grad2", action='store_true', help="whether to conj_grad2")
    parser.add_argument("--adapt", action='store_true', help="whether to adaptively turn off x grad correction")
    parser.add_argument("--adapt2", action='store_true', help="whether to adaptively turn off x grad correction 2")
    parser.add_argument("--thres", type=float, default=1e-2, help="threshold for turning off x grad correction")
    parser.add_argument("--relthres", type=float, default=1e-4, help="threshold for turning off x grad correction")
    parser.add_argument("--g_penalty", type=float, default=0, help="l2 norm penalty of G params")
    parser.add_argument("--g_hessian_reg", type=float, default=0, help="reg of G hessian")
    parser.add_argument("--d_penalty", type=float, default=0, help="l2 norm penalty of D params")
    parser.add_argument("--d_hessian_reg", type=float, default=0, help="reg of D hessian")
    parser.add_argument("--plot_lim_z", type=float, default=7, help="plot_lim_z")
    parser.add_argument("--plot_lim_x", type=float, default=7, help="plot_lim_x")
    parser.add_argument("--verbose", action='store_true', help="whether to print debug info")
    parser.add_argument("--verbose_update", action='store_true', help="whether to print update info")
    parser.add_argument("--noext", action='store_true', help="whether not to use extra gradient")
    parser.add_argument("--noext2", action='store_true', help="whether not to use extra gradient")
    parser.add_argument("--est_eig", action='store_true', help="whether not to use extra gradient")
    parser.add_argument("--eig_num", type=int, default=6, help="number of eigenvalues to be estiamted")
    parser.add_argument("--eig_maxiter", type=int, default=1000, help="maximum number of iterations for estimating eigenvalues")
    parser.add_argument("--save_param", action='store_true', help="whether not to save params")
    parser.add_argument("--G_base_filter_num", type=int, default=64, help="number of base filter number for G")
    parser.add_argument("--D_base_filter_num", type=int, default=64, help="number of base filter number for D")
    parser.add_argument("--simple_viz", action='store_true', help="whether to simplify viz")

    parser.add_argument("--alt_act_prop", type=float, default=0, help="alt_act_prop")
    parser.add_argument("--alt_act_factor", type=float, default=1, help="alt_act_factor")
    parser.add_argument('--alt_act_type', default="relu", type=str, help='alt_act_type')
    parser.add_argument('--num_splits', default=10, help='for MNIST IS')
    parser.add_argument('--model_dir', default="Saved_models/mnist_model_10.ckpt")
    parser.add_argument('--freeze_w', action='store_true', help="freeze_w")
    parser.add_argument('--freeze_b', action='store_true', help="freeze_b")
    parser.add_argument('--freeze_v', action='store_true', help="freeze_v")
    parser.add_argument('--cuda_deterministic', action='store_true', help="cuda_deterministic")
    parser.add_argument("--alpha_mobility", type=float, default=1, help="alpha for BP mobility")
    parser.add_argument("--alpha_mobility_D", type=float, default=1, help="alpha for BP mobility")

    parser.add_argument("--data_folder", type=str, default=data_folder, help="data_folder")
    parser.add_argument("--target_model", type=str, default=data_folder, help="target_model")
    parser.add_argument("--mnist_digits", type=str, default="0123456789", help="which digits to use")
    parser.add_argument("--use_spectral_norm", action='store_true', help="whether to use spectral_norm")
    parser.add_argument("--rmsprop_init_1", action='store_true', help="whether to init sq_avg in RMSProp as 1 (0 otherwise)")
    parser.add_argument("--spiky_init", action='store_true', help="whether to use spiky_init")
    parser.add_argument("--spiky_init_loss_thres", type=float, default=0.01, help="spiky_init_loss_thres")

    parser.add_argument("--Pi", type=str, default="", help="Pi")
    parser.add_argument("--Mu", type=str, default="", help="Mu")
    parser.add_argument("--Sigma2", type=str, default="", help="Sigma2")

    parser.add_argument("--use_tensorboard", action='store_true', help="whether to use tensorboard")
    parser.add_argument("--adapt_x", action='store_true', help="whether to use adaptiv lr ONLY for x")
    parser.add_argument("--adapt_y", action='store_true', help="whether to use adaptiv lr ONLY for y")

    parser.add_argument("--save_path", type=str, default="Data", help="exp data save path")
    parser.add_argument("--n_grad_norm_sample", type=int, default=0, help="n_grad_norm_sample")
    parser.add_argument("--n_interpolate", type=int, default=0, help="n_interpolate")
    # parser.add_argument("--n_grad_norm_sample", type=int, default=5, help="n_grad_norm_sample")
    # parser.add_argument("--n_interpolate", type=int, default=26, help="n_interpolate")


    parser.add_argument("--brenier", action='store_true', help="whether to use Brenier potential")
    parser.add_argument("--lazy", action='store_true', help="whether to refer to lazy training (https://papers.nips.cc/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf) (Section 3.1 for 2 layer fully connected; Section 3.2 for deep CNN)")

    parser.add_argument('--init_scheme', default="relu", type=str, help='init_scheme for dense layer weights')
    parser.add_argument("--show_init_params", action='store_true', help="whether to show_init_params")
    parser.add_argument("--notes", type=str, default="", help="notes")
    parser.add_argument("--mono", action='store_true', help="whether to use monotonic networks")
    parser.add_argument("--mono_loss_weights", type=float, default=0, help="weights of monotonicity regularization")



    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True

    st = time.time()
    Train_GAN(args)
    print(f"Experiment completed. Time: {time.time() - st:.3f}")
