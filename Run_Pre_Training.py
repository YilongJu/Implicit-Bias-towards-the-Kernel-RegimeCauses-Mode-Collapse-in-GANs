
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models.inception import inception_v3

import argparse
from Run_GAN_Training import *
from ComputationalTools import *
from models import MLP
from Synthetic_Dataset_1D import Synthetic_Dataset_1D
from deepviz_pre import DeepVisuals_Pre


pre_model_folder = "Pre_Models"
if not os.path.exists(pre_model_folder):
    os.makedirs(pre_model_folder)

def Train_Pre(args):
    training_timer = Timer()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{'=' * 40}\n[Using device: {device}]\n{'=' * 40}")
    exp_name = f"{args.data}-xl{args.x_l}-xu{args.x_u}-ys{args.scale_y}-xoff{args.offset_x}-ns{args.noise_std}-np{args.num_periods}-ns{args.num_samples}_{args.opt_type}_MLP-{args.layers}-{args.hidden}_it{args.iteration}_aa{args.alt_act_prop}-{args.alt_act_factor}-{args.alt_act_type}_sd{args.seed}"
    model_save_folder = os.path.join(pre_model_folder, exp_name)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    rng = np.random.RandomState(seed=args.seed)
    dataset = Synthetic_Dataset_1D(args.data, rng, num_samples=args.num_samples, range_x=[args.x_l, args.x_u], scale_y=args.scale_y, noise_std=args.noise_std, num_periods=args.num_periods, offset_x=args.offset_x)

    net = MLP(n_hidden_layers=args.layers, n_hidden_neurons=args.hidden, input_dim=1, output_dim=1, type="D", use_bias=False, alt_act_prop=args.alt_act_prop, alt_act_factor=args.alt_act_factor, alt_act_type=args.alt_act_type).to(device)
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    if args.opt_type == "sgd":
        optimizer = torch.optim.SGD(params=net.parameters(), lr=args.lr)
    elif args.opt_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params=net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not implemented.")


    deepVisuals_Pre = DeepVisuals_Pre(args, dataset, exp_name)
    deepVisuals_Pre.Init_figure()

    """ Training """
    for i in range(args.iteration + 1):
        input = torch.from_numpy(dataset.x).to(device).float().unsqueeze(1)
        target = torch.from_numpy(dataset.y).to(device).float().unsqueeze(1)
        preds = net(input)
        loss = criterion(preds, target)

        if i % args.plot_iter == 0:

            print(f"Iter {i} / {args.iteration}, Loss {loss.item():.4f}, Time: {training_timer.Print(print_time=False)}")
            preds_linspace = net(torch.from_numpy(dataset.x_linspace).to(device).float().unsqueeze(1)).cpu().detach().numpy().ravel()
            idd = {"iter": i, "preds": preds, "preds_linspace": preds_linspace, "loss": loss, "state_dict": net.state_dict()}
            deepVisuals_Pre.Plot_step(idd)

        if i % args.save_iter == 0:
            torch.save({"args": args, "state": net.state_dict()}, os.path.join(model_save_folder, f"iter_{i}-{args.iteration}.pth"))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    deepVisuals_Pre.handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=-1, help="number of iterations of training")
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--test_data_num", type=int, default=256, help="number of test data")
    parser.add_argument("--lr", type=float, default=0.002, help="generator learning rate")
    parser.add_argument("--lr_warmup", type=float, default=0.00000001, help="generator warmup learning rate")
    parser.add_argument("--lr_decay", type=float, default=0, help="generator learning rate decay")
    parser.add_argument("--warmup_iter", type=int, default=1, help="number of warmup iterations")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0, help="weight decay factor")
    parser.add_argument("--gp", type=float, default=0, help="gradient penalty")
    parser.add_argument("--z_dim", type=int, default=-1, help="dimension of latent noise")
    parser.add_argument("--z_std", type=float, default=1, help="std of latent noise")
    parser.add_argument("--mog_std", type=float, default=-1, help="std of mog")
    parser.add_argument("--mog_scale", type=float, default=-1, help="scale of mog")
    parser.add_argument("--hidden", type=int, default=32, help="dimension of hidden units")
    parser.add_argument("--layers", type=int, default=2, help="num of hidden layer")
    parser.add_argument("--x_dim", type=int, default=2, help="data dimension")
    parser.add_argument("--x_l", type=float, default=-1, help="x_l")
    parser.add_argument("--x_u", type=float, default=1, help="x_u")
    parser.add_argument("--scale_y", type=float, default=1, help="scale_y")
    parser.add_argument("--noise_std", type=float, default=0, help="noise_std")
    parser.add_argument("--num_periods", type=float, default=1, help="num_periods")
    parser.add_argument("--num_samples", type=int, default=20, help="num_samples")
    parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
    parser.add_argument("--arch", type=str, default="default", help="which gan architecture")
    parser.add_argument("--data", type=str, default="cifar", help="which dataset")
    parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
    parser.add_argument("--act", type=str, default="relu", help="which activation function for gen")  # elu, relu, tanh
    parser.add_argument("--divergence", type=str, default="JS", help="which activation function for disc")  # NS, JS, indicator, wgan
    parser.add_argument("--method", type=str, default="fr", help="which optimization regularization method to use")
    parser.add_argument("--opt_type", type=str, default="sgd", help="which optimization method to use")
    parser.add_argument("--reg_param", type=float, default=5.0, help="reg param for JARE")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum coefficient for the whole system")
    parser.add_argument("--gamma", type=float, default=0.999, help="gamma for adaptive learning rate")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--inner_iter", type=int, default=5, help="conjugate gradient or gradient descent steps")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--damping", type=float, default=1.0, help="damping term for CG")
    parser.add_argument("--adapt_damping", action='store_false', help="whether to adapt damping")
    parser.add_argument("--real_time_video", action='store_false', help="whether to play video in real time")
    parser.add_argument("--plot_iter", type=int, default=150, help="number of iter to plot")
    parser.add_argument("--save_iter", type=int, default=10000, help="number of iter to save")
    parser.add_argument("--eval_iter", type=int, default=-1, help="number of iter to eval")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--spanning_init", action='store_true', help="whether to spanning_init")
    parser.add_argument("--si_lr", type=float, default=0.01, help="spanning_init learning rate")
    parser.add_argument("--degen_z", action='store_true', help="whether to reduce dim_z")
    parser.add_argument("--penalty", type=float, default=0, help="l2 norm penalty of G params")
    parser.add_argument("--hessian_reg", type=float, default=0, help="reg of G hessian")
    parser.add_argument("--plot_lim_z", type=float, default=7, help="plot_lim_z")
    parser.add_argument("--plot_lim_x", type=float, default=7, help="plot_lim_x")
    parser.add_argument("--verbose", action='store_true', help="whether to print debug info")
    parser.add_argument("--base_filter_num", type=int, default=64, help="number of base filter number for G")

    parser.add_argument("--alt_act_prop", type=float, default=0, help="alt_act_prop")
    parser.add_argument("--alt_act_factor", type=float, default=1, help="alt_act_factor")
    parser.add_argument('--alt_act_type', default="relu", type=str, help='alt_act_type')

    parser.add_argument("--offset_x", type=float, default=0, help="offset_x")



    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(seed=args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    Train_Pre(args)
