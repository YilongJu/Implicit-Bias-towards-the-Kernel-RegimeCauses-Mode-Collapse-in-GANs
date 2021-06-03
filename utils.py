from argparse import ArgumentParser
import sys

from ComputationalTools import *
from Synthetic_Dataset_1D import Synthetic_Dataset_1D

from models import dc_D, dc_G, dc_d, dc_g, GoodDiscriminator, GoodGenerator, GoodDiscriminatord, MLP, DMLP, DCGAN_Generator, DCGAN_Discriminator, DCGAN_weights_init






def Get_models(args, output_shape=None, gpu_num=1):
    """
    Default:
    G = GoodGenerator()
    D = GoodDiscriminator()
    """
    # TODO: Add architecture selection
    if args.arch == "default":
        if args.data in ["cifar"]:
            G = GoodGenerator(dim=args.G_base_filter_num)
            D = GoodDiscriminator(dim=args.D_base_filter_num)
        elif args.data in ["mnist"]:
            G = DCGAN_Generator(nz=args.z_dim, ngf=args.G_base_filter_num, freeze_w=args.freeze_w)
            D = DCGAN_Discriminator(ndf=args.D_base_filter_num)
        else:
            raise NotImplementedError
    elif args.arch == "mlp":
        if output_shape is None:
            x_dim = args.x_dim
        else:
            x_dim = np.prod(output_shape)

        if not hasattr(args, "use_spectral_norm"):
            args.use_spectral_norm = False

        if args.brenier:
            G_output_dim = 1
        else:
            G_output_dim = x_dim

        G = MLP(args.g_layers, args.g_hidden, args.z_dim, G_output_dim, output_shape=output_shape, type="G", alt_act_prop=args.alt_act_prop, alt_act_factor=args.alt_act_factor, alt_act_type=args.alt_act_type, freeze_w=args.freeze_w, freeze_b=args.freeze_b, freeze_v=args.freeze_v, alpha_mobility=args.alpha_mobility, use_spectral_norm=args.use_spectral_norm)
        D = MLP(args.d_layers, args.d_hidden, x_dim, 1, output_shape=output_shape, type="D", bias_scale=args.d_bias_scale, alpha_mobility=args.alpha_mobility_D, use_spectral_norm=args.use_spectral_norm)
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


def train_seq_parser():
    usage = 'Parser for sequential training'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--datapath', type=str, default='./datas')
    parser.add_argument('--model', type=str, default='DCGAN')
    parser.add_argument('--model_weight', type=str, default='random')
    parser.add_argument('--z_dim', type=int, default=96)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--optimizer', type=int, default='Adam')
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='WGAN')
    parser.add_argument('--d_penalty', type=float, default=0.0)
    parser.add_argument('--gp_weight', type=float, default=10)
    parser.add_argument('--d_iter', type=int, default=5,
                        help='-1: only update generator, 0: only update discriminator')
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--show_iter', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='test')
    parser.add_argument('--gpu_num', type=int, default=1)

    return parser


def prepare_parser():
    usage = 'Parser for training'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10')
    parser.add_argument(
        '--datapath', type=str, default='cifar10'
    )
    parser.add_argument(
        '--model', type=str, default='DCGAN')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--startPoint', type=int, default=0)
    parser.add_argument(
        '--dropout', action='store_true', default=False)
    parser.add_argument(
        '--z_dim', type=int, default=128)
    parser.add_argument(
        '--batchsize', type=int, default=64)

    parser.add_argument(
        '--optimizer', type=str, default='Adam')
    parser.add_argument(
        '--lr_d', type=float, default=2e-4)
    parser.add_argument(
        '--lr_g', type=float, default=2e-4)
    parser.add_argument(
        '--momentum', type=float, default=0.9)
    parser.add_argument(
        '--loss_type', type=str, default='WGAN')
    parser.add_argument(
        '--g_penalty', type=float, default=0.0)
    parser.add_argument(
        '--d_penalty', type=float, default=0.0)
    # parser.add_argument('--use_gp', action='store_true', default=False)
    parser.add_argument(
        '--gp_weight', type=float, default=0.0)
    parser.add_argument(
        '--d_iter', type=int, default=1)

    parser.add_argument(
        '--epoch_num', type=int, default=600)
    parser.add_argument(
        '--show_iter', type=int, default=500)
    parser.add_argument(
        '--eval_iter', type=int, default=5000)

    parser.add_argument(
        '--eval_is', action='store_true', default=False)
    parser.add_argument(
        '--eval_fid', action='store_true', default=False)
    parser.add_argument('--logdir', type=str, default='DC-WGAN-GP')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--collect_info', action='store_true', default=False)
    return parser


def eval_parser():
    usage = 'Parser for eval'
    parser = ArgumentParser(description=usage)
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--begin', type=int, default=4000)
    parser.add_argument('--end', type=int, default=400000)
    parser.add_argument('--step', type=int, default=4000)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument(
        '--eval_is', action='store_true', default=False)
    parser.add_argument(
        '--eval_fid', action='store_true', default=False)
    return parser


def Get_experiment_name_from_args(args):
    verboseprint = print if args.verbose else lambda *a, **k: None

    if args.eval_iter < 0:
        args.eval_iter = args.plot_iter

    args.adapt_damping = True  # True, False
    args.vanilla_gradient = True
    if args.pre:
        args.vanilla_gradient = False
    # args.divergence = "JS" # standard, JS, indicator, wganV7FH4
    # args.opt_type = "sgd" # sgd, rmsprop, adam
    args.real_time_video = False  # True, False

    print(args)
    # automatically setup the name
    reg_text = ""
    if args.weight_decay > 0:
        reg_text += f"_wd{args.weight_decay}"

    if args.g_penalty > 0:
        reg_text += f"_ge{args.g_penalty}"

    if args.d_penalty > 0:
        reg_text += f"_de{args.d_penalty}"

    if args.g_hessian_reg < 0:
        args.g_hessian_reg = 2 * args.g_penalty

    if args.d_hessian_reg < 0:
        args.d_hessian_reg = 2 * args.d_penalty

    if args.method in ["fr", "fr2", "fr3", "fr4", "fr5", "frm"]:
        if args.g_hessian_reg > 0:
            reg_text += f"_gh{args.g_hessian_reg}"

        if args.d_hessian_reg > 0:
            reg_text += f"_dh{args.d_hessian_reg}"

    if args.gp > 0:
        reg_text += f"_gp{args.gp}"

    if args.fair:
        reg_text += f"_fa"

    g_lr_decay_text = f"-{args.g_lr_decay}" if args.g_lr_decay != 1.0 else ""
    d_lr_decay_text = f"-{args.d_lr_decay}" if args.d_lr_decay != 1.0 else ""
    lr_text = f"lrG{args.g_lr}{g_lr_decay_text}_lrD{args.d_lr}{d_lr_decay_text}"
    if args.method == "cgd":
        lr_text = f"lr{args.lr}"
    elif args.method in ["fr", "fr2", "fr3", "frm"]:
        if args.adapt:
            reg_text += f"_ad"
        if args.adapt2:
            reg_text += f"_ad2"
        if args.adapt or args.adapt2:
            reg_text += f"_th{args.thres}_rth{args.relthres}"
    elif args.method == "jare":
        reg_text = f"_reg{args.reg_param}"

    if args.noext:
        reg_text += f"_nx"
    if args.noext2:
        reg_text += f"_nx2"

    pre_text = ""
    if args.pre:
        pre_text += "_pre"

    cj_text = ""
    if args.conj_grad:
        cj_text += "_cj"
    cj2_text = ""
    if args.conj_grad2:
        cj2_text += "_cj2"

    # if args.method in ["fr", "fr2", "fr3", "frm", "cgd"]:
    #     reg_text += f"_int{args.inner_iter}"

    arch_text = ""
    if args.arch in ["mlp", "dmlp_g"]:
        d_bias_scale_text = ""
        if args.d_bias_scale != 1:
            d_bias_scale_text = f"-bs{args.d_bias_scale}"

        g_arch_text = ""
        if args.arch in ["dmlp_g"]:
            g_arch_text = "-sep"

        if hasattr(args, "brenier"):
            if args.brenier:
                g_arch_text += "-bre"

        arch_text += f"_G{g_arch_text}{args.g_layers}-{args.g_hidden}_D{args.d_layers}-{args.d_hidden}{d_bias_scale_text}"
    else:
        arch_text += f"_{args.arch}-G{args.G_base_filter_num}-D{args.D_base_filter_num}"

    mog_text = ""
    if args.data in ["mnist", "cifar"]:
        pass
    elif args.data == "mog1d":
        dataset = Synthetic_Dataset_1D("mog", num_samples=5000, args=args)
        mog_text += f"-{dataset.MG.Get_full_Str(check_length=True)}"
        args.x_dim = 1
    else:
        mog_text += f"-{args.mog_scale}-{args.mog_std}"

    opt_text = ""
    invidual_adapt_text = ""
    if args.adapt_x and args.adapt_y:
        raise ValueError("args.adapt_x and args.adapt_y cannot both be true.")
    if args.adapt_x:
        invidual_adapt_text = "-x"
    if args.adapt_y:
        invidual_adapt_text = "-y"

    opt_text += invidual_adapt_text
    if args.opt_type in ["rmsprop", "adam"]:
        opt_text += f"-gm{args.gamma}"
        if args.rmsprop_init_1:
            init_text = "-1"
        else:
            init_text = "-0"
        opt_text += init_text

    if args.opt_type in ["adam"]:
        opt_text += f"-mo{args.momentum}"

    if args.z_dim == -1:
        if args.data in ["mnist", "cifar"]:
            args.z_dim = 128
        else:
            args.z_dim = 2

    ie_text = ""
    if args.iteration > 0:
        ie_text += f"_it{args.iteration}"
    if args.epoch > 0:
        ie_text += f"_ep{args.epoch}"

    ie_text += f"-pi{args.plot_iter}"
    if (args.iteration < 0 and args.epoch < 0) or (args.iteration > 0 and args.epoch > 0):
        raise ValueError

    video_title = f"{args.data}{mog_text}_{args.method}_{args.divergence}_{args.opt_type}{opt_text}{pre_text}{ie_text}{arch_text}_{lr_text}{reg_text}{cj_text}{cj2_text}_zd{args.z_dim}_zs{args.z_std}_bs{args.batch_size}_sd{args.seed}"

    if args.data not in ["mnist", "cifar", "mog1d"]:
        video_title += f"_ds{args.data_std}"
    if args.data in ["mnist"]:
        video_title += f"_dg{args.mnist_digits}"

    if args.save_param:
        video_title += "_sp"
    if args.spanning_init:
        video_title += f"_silr{args.si_lr}"
    if args.degen_z:
        video_title += f"_dz"
    if args.simple_viz:
        video_title += f"_sv"

    if args.alt_act_prop > 0:
        video_title += f"_a{args.alt_act_prop}-{args.alt_act_type}"
        if args.alt_act_factor != 1:
            video_title += f"-{args.alt_act_factor}"

    if args.eigs:
        video_title += "_e"

    if args.freeze_w:
        video_title += "_fw"
    if args.freeze_b:
        video_title += "_fb"
    if args.freeze_v:
        video_title += "_fv"

    if args.use_spectral_norm:
        video_title += "_sn"

    if args.spiky_init:
        video_title += "_si"

    lazy_text = ""
    if hasattr(args, "lazy"):
        if args.lazy:
            lazy_text += "lz-"
    video_title += f"_{lazy_text}aG{args.alpha_mobility}-D{args.alpha_mobility_D}"

    if hasattr(args, "init_scheme"):
        init_scheme_text = f"_insc-{args.init_scheme}" if args.init_scheme != "relu" else ""
        video_title += init_scheme_text

    if hasattr(args, "show_init_params"):
        if args.show_init_params:
            video_title += f"_init"


    if hasattr(args, "mono") and hasattr(args, "mono_loss_weights"):
        if args.mono:
            video_title += f"_mono-{args.mono_loss_weights}"

    if hasattr(args, "notes"):
        notes = f"_{args.notes}" if args.notes != "" else ""
        video_title += notes



    return video_title, args, verboseprint


def log(**datum):
    data = []
    sys.stdout.write('\r')
    sys.stdout.write(str(datum))
    data.append(datum)

    return list(data[0].values())