import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import math
import torch
import numpy as np
from BPs import *
import matplotlib.pyplot as plt
import os

DIM = 64

def Init_layer(layer, weight_scale=1., bias_scale=1., lazy=False, is_second_layer=False, init_scheme="relu"):
    init.kaiming_uniform_(layer.weight, a=0, nonlinearity=init_scheme) # Original = math.sqrt(5), no effect to ReLU
    layer.weight = nn.Parameter(layer.weight * weight_scale)
    # print("layer.weight.shape", layer.weight.shape)

    if lazy:
        # print("[Original] layer.weight", layer.weight)
        with torch.no_grad():
            if is_second_layer:
                H_half = layer.weight.shape[1] // 2
                layer.weight[:, H_half:] = nn.Parameter(-layer.weight[:, :H_half])
            else:
                H_half = layer.weight.shape[0] // 2
                layer.weight[H_half:] = nn.Parameter(layer.weight[:H_half])
        # print("[Symmetrized] layer.weight", layer.weight)

    if layer.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(layer.bias, -bound, bound)
        layer.bias = nn.Parameter(layer.bias * bias_scale)

        if lazy:
            # print("[Original] layer.bias", layer.bias)
            H_half = len(layer.bias) // 2
            with torch.no_grad():
                layer.bias[H_half:] = nn.Parameter(layer.bias[:H_half])
            # print("[Symmetrized] layer.bias", layer.bias)


def Freeze_convnet_weights(seq_module_list):
    last_conv_i = -1
    for i in range(len(seq_module_list)):
        layer_name = seq_module_list[i].__class__.__name__
        if layer_name.find('Conv') != -1:
            seq_module_list[i].weight.requires_grad = False
            print(f"Layer {i} {layer_name} weight requires_grad {seq_module_list[i].weight.requires_grad}.")
            last_conv_i = i

    if last_conv_i > 0:
        layer_name = seq_module_list[last_conv_i].__class__.__name__
        seq_module_list[last_conv_i].weight.requires_grad = True
        print(f"Layer {last_conv_i} {layer_name} weight requires_grad {seq_module_list[last_conv_i].weight.requires_grad}.")


class MLP_relu(nn.Module):
    def __init__(self, mono_feature, non_mono_feature, mono_sub_num=1, non_mono_sub_num=1, mono_hidden_num=5, non_mono_hidden_num=5, compress_non_mono=False,
                 normalize_regression=False):
        super(MLP_relu, self).__init__()
        self.normalize_regression = normalize_regression
        self.compress_non_mono = compress_non_mono
        if compress_non_mono:
            self.non_mono_feature_extractor = nn.Linear(non_mono_feature, 10, bias=True)
            self.mono_fc_in = nn.Linear(mono_feature + 10, mono_hidden_num, bias=True)
        else:
            self.mono_fc_in = nn.Linear(mono_feature + non_mono_feature, mono_hidden_num, bias=True)

        bottleneck = 10
        self.non_mono_fc_in = nn.Linear(non_mono_feature, non_mono_hidden_num, bias=True)
        self.mono_submods_out = nn.ModuleList([nn.Linear(mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)])
        self.mono_submods_in = nn.ModuleList([nn.Linear(2 * bottleneck, mono_hidden_num, bias=True) for i in range(mono_sub_num)])
        self.non_mono_submods_out = nn.ModuleList([nn.Linear(non_mono_hidden_num, bottleneck, bias=True) for i in range(mono_sub_num)])
        self.non_mono_submods_in = nn.ModuleList([nn.Linear(bottleneck, non_mono_hidden_num, bias=True) for i in range(mono_sub_num)])

        self.mono_fc_last = nn.Linear(mono_hidden_num, 1, bias=True)
        self.non_mono_fc_last = nn.Linear(non_mono_hidden_num, 1, bias=True)

        self.in_list = None
        self.out_list = None

    def forward(self, mono_feature, non_mono_feature=None):
        y = self.non_mono_fc_in(non_mono_feature)
        y = F.relu(y)

        if self.compress_non_mono:
            non_mono_feature = self.non_mono_feature_extractor(non_mono_feature)
            non_mono_feature = F.hardtanh(non_mono_feature, min_val=0.0, max_val=1.0)

        x = self.mono_fc_in(torch.cat([mono_feature, non_mono_feature], dim=1))
        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            x = F.hardtanh(x, min_val=0.0, max_val=1.0)

            y = self.non_mono_submods_out[i](y)
            y = F.hardtanh(y, min_val=0.0, max_val=1.0)

            x = self.mono_submods_in[i](torch.cat([x, y], dim=1))
            x = F.relu(x)

            y = self.non_mono_submods_in[i](y)
            y = F.relu(y)

        x = self.mono_fc_last(x)

        y = self.non_mono_fc_last(y)

        out = x + y
        if self.normalize_regression:
            out = F.sigmoid(out)
        return out

    def generate_regularizer(self, in_list=None, out_list=None):
        if in_list is None:
            in_list = self.in_list
        if out_list is None:
            out_list = self.out_list

        length = len(in_list)
        reg_loss = 0.
        min_derivative = 0.0
        for i in range(length):
            xx = in_list[i]
            yy = out_list[i]
            for j in range(yy.shape[1]):
                grad_input = torch.autograd.grad(torch.sum(yy[:, j]), xx, create_graph=True, allow_unused=True)[0]
                grad_input_neg = -grad_input
                grad_input_neg += .2
                grad_input_neg[grad_input_neg < 0.] = 0.
                if min_derivative < torch.max(grad_input_neg ** 2):
                    min_derivative = torch.max(grad_input_neg ** 2)
        reg_loss = min_derivative
        return reg_loss

    def reg_forward(self, feature_num, mono_num, bottleneck=10, num=512):
        in_list = []
        out_list = []
        if self.compress_non_mono:
            input_feature = torch.rand(num, mono_num + 10).cuda()
        else:
            input_feature = torch.rand(num, feature_num).cuda()
        input_mono = input_feature[:, :mono_num]
        input_non_mono = input_feature[:, mono_num:]
        input_mono.requires_grad = True

        x = self.mono_fc_in(torch.cat([input_mono, input_non_mono], dim=1))
        in_list.append(input_mono)

        x = F.relu(x)
        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)
            out_list.append(x)

            input_feature = torch.rand(num, 2 * bottleneck).cuda()
            input_mono = input_feature[:, :bottleneck]
            input_non_mono = input_feature[:, bottleneck:]
            in_list.append(input_mono)
            in_list[-1].requires_grad = True

            x = self.mono_submods_in[i](torch.cat([input_mono, input_non_mono], dim=1))
            x = F.relu(x)

        x = self.mono_fc_last(x)
        out_list.append(x)

        self.in_list = in_list
        self.out_list = out_list

        return in_list, out_list



class MLP(nn.Module):
    def __init__(self, n_hidden_layers, n_hidden_neurons, input_dim, output_dim, output_shape=None, type=None, use_bias=False, alt_act_prop=0., alt_act_factor=1., alt_act_type="relu", verbose=False, weight_scale=1., bias_scale=1., vweight_scale=1.0, freeze_w=False, freeze_b=False, freeze_v=False, alpha_mobility=-1, use_spectral_norm=False, lazy=False, init_scheme="relu"):
        """
        Create an MLP
        :param n_hidden_layers:
        :param n_hidden_neurons:
        :param input_dim:
        :param output_dim:
        :param output_shape:
        :param type: Default: None, ["G, "D"]
        :param use_bias:
        :param alt_act_prop:
        :param alt_act_factor:
        :param alt_act_type:
        :param verbose:
        """
        super(MLP, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_shape = output_shape
        self.type = type
        self.weight_scale = weight_scale
        self.use_spectral_norm = use_spectral_norm
        self.lazy = lazy
        self.init_scheme = init_scheme

        self.alpha_mobility = alpha_mobility
        if self.alpha_mobility > 0:
            self.bias_scale = self.alpha_mobility
            self.weight_scale = self.alpha_mobility
            if self.lazy:
                self.vweight_scale = self.alpha_mobility
            else:
                self.vweight_scale = 1. / self.alpha_mobility
        else:
            self.bias_scale = bias_scale
            self.weight_scale = weight_scale
            self.vweight_scale = vweight_scale

        if self.n_hidden_layers == 0:
            self.output_layer = nn.Linear(self.input_dim, self.output_dim, bias=use_bias)
        else:
            self.hidden_layer = nn.Linear(self.input_dim, self.n_hidden_neurons)
            Init_layer(self.hidden_layer, weight_scale=self.weight_scale, bias_scale=self.bias_scale, lazy=self.lazy, is_second_layer=False, init_scheme=self.init_scheme)
            if freeze_w:
                self.hidden_layer.weight.requires_grad = False
            if freeze_b:
                self.hidden_layer.bias.requires_grad = False
            self.deeper_hidden_layers = nn.ModuleList([nn.Linear(self.n_hidden_neurons, self.n_hidden_neurons) for _ in range(self.n_hidden_layers - 1)])
            self.output_layer = nn.Linear(self.n_hidden_neurons, self.output_dim, bias=use_bias)
            Init_layer(self.output_layer, weight_scale=self.vweight_scale, lazy=self.lazy, is_second_layer=True)
            if freeze_v:
                self.output_layer.weight.requires_grad = False

        if self.use_spectral_norm:
            if self.n_hidden_layers > 0:
                self.hidden_layer = spectral_norm(self.hidden_layer)
                self.deeper_hidden_layers = nn.ModuleList([spectral_norm(deeper_hidden_layer) for deeper_hidden_layer in self.deeper_hidden_layers])

            self.output_layer = spectral_norm(self.output_layer)


        self.act_func = nn.ReLU()
        self.alt_act_prop = alt_act_prop
        self.alt_act_num = np.floor(alt_act_prop * self.n_hidden_neurons).astype(int)
        self.alt_act_factor = alt_act_factor
        self.alt_act_type = alt_act_type
        if self.alt_act_type == "relu":
            self.alt_act_func = nn.ReLU()
        elif self.alt_act_type == "sigmoid":
            self.alt_act_func = nn.Sigmoid()
        else:
            raise NotImplementedError

        self.verbose = verbose
        self.verboseprint = print if verbose else lambda *a, **k: None

    def Alt_act(self, pre_act):
        if self.alt_act_num > 0:
            show_num = 6
            # self.verboseprint(f"\nPre relu: shape {pre_act.shape}, alt_act_num {self.alt_act_num}\n{pre_act}")
            pre_act_0 = pre_act[:, :self.alt_act_num]
            pre_act_1 = pre_act[:, self.alt_act_num:]

            # self.verboseprint(f"{pre_act_0[:, :show_num]}\n{pre_act_1[:, :show_num]}")
            act_tuple = (self.alt_act_factor * self.alt_act_func(pre_act_0), self.act_func(pre_act_1))
            # self.verboseprint(act_tuple[0].shape, act_tuple[1].shape)
            # self.verboseprint(f"{act_tuple[0][:, :show_num]}\n{act_tuple[1][:, :show_num]}")

            post_act = torch.cat(act_tuple, dim=1)
            # self.verboseprint(f"Post relu: shape {post_act.shape}, alt_act_num {self.alt_act_num}\n{post_act}")
            return post_act
        else:
            return self.act_func(pre_act)

    def forward(self, net):
        # print("net.shape", net.shape) # [128, 3, 32, 32]
        if self.type == "D": # Flatten input for D
            net = net.view(net.shape[0], -1)
            # print("net.shape", net.shape) # [128, 3072]
        if self.n_hidden_layers > 0:
            net = self.hidden_layer(net)
            net = self.Alt_act(net)
            for hidden_layer in self.deeper_hidden_layers:
                net = hidden_layer(net)
                net = self.Alt_act(net)

        net = self.output_layer(net)
        if (self.output_shape is not None) and (self.type == "G"): # Fold long vector to image for G
            net = net.view([-1] + self.output_shape)

        return net


class DMLP(nn.Module):
    def __init__(self, n_hidden_layers, n_hidden_neurons, input_dim, output_dim, output_shape=None, type=None, use_bias=False):
        """
        Create an MLP
        :param n_hidden_layers:
        :param n_hidden_neurons:
        :param input_dim:
        :param output_dim:
        :param output_shape:
        :param type: Default: None, ["G, "D"]
        :param use_bias:
        :param alt_act_prop:
        :param alt_act_factor:
        :param alt_act_type:
        :param verbose:
        """
        super(DMLP, self).__init__()
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_shape = output_shape
        self.type = type

        self.hidden_layer_list = nn.ModuleList([nn.Linear(self.input_dim, self.n_hidden_neurons) for _ in range(self.output_dim)])
        self.output_layer_list = nn.ModuleList([nn.Linear(self.n_hidden_neurons, 1, bias=use_bias) for _ in range(self.output_dim)])

        self.act_func = nn.ReLU()

    def forward(self, net):
        net = torch.cat([output_layer(self.act_func(hidden_layer(net)))
                         for hidden_layer, output_layer in zip(self.hidden_layer_list, self.output_layer_list)], dim=1)
        return net

class GoodGenerator(nn.Module):
    def __init__(self, dim=64):
        super(GoodGenerator, self).__init__()
        self.dim = dim
        self.preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * self.dim),
            nn.BatchNorm1d(4 * 4 * 4 * self.dim),
            nn.ReLU(True),
        )

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * self.dim),
            nn.ReLU(True),
            # nn.Softplus(),
            nn.ConvTranspose2d(2 * self.dim, self.dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            # nn.Softplus(),
            nn.ConvTranspose2d(self.dim, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.main_module(output)
        # print("output.shape", output.shape) # [batch_size, 3, 32, 32]
        return output.view(-1, 3, 32, 32)


class GoodDiscriminator(nn.Module):
    def __init__(self, dim=64):
        super(GoodDiscriminator, self).__init__()
        self.dim = dim
        self.main_module = nn.Sequential(
            nn.Conv2d(3, self.dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 16x16
            nn.Conv2d(self.dim, 2 * self.dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 8x8
            nn.Conv2d(2 * self.dim, 4 * self.dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            # nn.Softplus(),
            # nn.Dropout2d(),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * self.dim, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * self.dim)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output


class GoodDiscriminatord(nn.Module):
    def __init__(self, dropout=0.5):
        super(GoodDiscriminatord, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(3, DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 16x16
            nn.Conv2d(DIM, 2 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 8x8
            nn.Conv2d(2 * DIM, 4 * DIM, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            # 4 x 4
        )
        self.linear = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        output = self.main_module(input)
        output = output.view(-1, 4 * 4 * 4 * DIM)
        # print(output.shape)
        output = self.linear(output)
        # print(output.shape)
        return output


class dc_d(nn.Module):
    def __init__(self, ndf=32):
        super(dc_d, self).__init__()
        self.ndf = ndf
        self.conv = nn.Sequential(
            # 3 * 32x32
            nn.Conv2d(in_channels=3, out_channels=self.ndf, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            # 32 * 14x14
            nn.Conv2d(in_channels=self.ndf, out_channels=2 * self.ndf, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
            # 64 * 5x5
        )
        self.fc = nn.Sequential(
            nn.Linear(50 * self.ndf, 32 * self.ndf),
            nn.LeakyReLU(0.01),
            nn.Linear(32 * self.ndf, 1)
        )

    def forward(self, x):
        # print("input shape", x.shape)
        x = self.conv(x)
        # print("conv shape", x.shape)
        x = x.view(x.shape[0], -1)
        # print("flat shape", x.shape)
        return self.fc(x)


class dc_g(nn.Module):
    def __init__(self, z_dim=96, ngf=64):
        super(dc_g, self).__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 16 * self.ngf),
            nn.ReLU(),
            nn.BatchNorm1d(16 * self.ngf),
            nn.Linear(16 * self.ngf, 8 * 8 * 2 * self.ngf),
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 2 * self.ngf),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * self.ngf, out_channels=self.ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.ngf),
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # print("input shape", x.shape)
        x = self.fc(x)
        # print("fc shape", x.shape)
        x = x.view(x.shape[0], 2 * self.ngf, 8, 8)
        # print("flat shape", x.shape)
        return self.convt(x)


class DC_g(nn.Module):
    def __init__(self, z_dim=100, channel_num=3):
        super(DC_g, self).__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # 1024 * 4x4
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 * 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 * 16x16
            nn.ConvTranspose2d(256, channel_num, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # 3 * 32x32
        )

    def forward(self, input):
        return self.main_module(input)


class DC_d(nn.Module):
    def __init__(self, channel_num=3):
        super(DC_d, self).__init__()
        self.main_module = nn.Sequential(
            nn.Conv2d(channel_num, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # 1024 * 4x4
            nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, input):
        return self.main_module(input)


class DC_generator(nn.Module):
    def __init__(self, z_dim=100, channel_num=3, feature_num=64):
        super(DC_generator, self).__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_num * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.ReLU(inplace=True),
            # (feature_num * 8) * 4x4
            nn.ConvTranspose2d(feature_num * 8, feature_num * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.ReLU(inplace=True),
            # (feature_num * 4) * 8x8
            nn.ConvTranspose2d(feature_num * 4, feature_num * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.ReLU(inplace=True),
            # (feature_num * 2) * 16x16
            nn.ConvTranspose2d(feature_num * 2, feature_num, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(inplace=True),
            # (feature_num * 2) * 32x32
            nn.ConvTranspose2d(feature_num, channel_num, kernel_size=4, stride=2, padding=1,
                               bias=False),
            # channel_num * 64x64
            nn.Tanh()
        )

    def forward(self, input):
        return self.main_module(input)


class DC_discriminator(nn.Module):
    def __init__(self, channel_num=3, feature_num=64):
        super(DC_discriminator, self).__init__()
        self.main_module = nn.Sequential(
            # channel_num * 64x64
            nn.Conv2d(channel_num, feature_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num) * 32x32
            nn.Conv2d(feature_num, feature_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 2) * 16x16
            nn.Conv2d(feature_num * 2, feature_num * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 4) * 8x8
            nn.Conv2d(feature_num * 4, feature_num * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_num * 8) * 4x4
            nn.Conv2d(feature_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # feature_num * 16x16
        )

    def forward(self, input):
        return self.main_module(input)


class DC_discriminatord(nn.Module):
    def __init__(self, channel_num=3, feature_num=64):
        super(DC_discriminatord, self).__init__()
        self.main_module = nn.Sequential(
            # channel_num * 64x64
            nn.Conv2d(channel_num, feature_num, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num) * 32x32
            nn.Conv2d(feature_num, feature_num * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 2) * 16x16
            nn.Conv2d(feature_num * 2, feature_num * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 4) * 8x8
            nn.Conv2d(feature_num * 4, feature_num * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(),
            # (feature_num * 8) * 4x4
            nn.Conv2d(feature_num * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # feature_num * 16x16
        )

    def forward(self, input):
        return self.main_module(input)


class dc_D(nn.Module):
    def __init__(self, ngf=32):
        super(dc_D, self).__init__()
        self.ngf = ngf
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.ngf, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=self.ngf, out_channels=2 * self.ngf, kernel_size=5, stride=1),
            nn.LeakyReLU(0.01),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


class dc_G(nn.Module):
    def __init__(self, z_dim=96, ngf=64):
        super(dc_G, self).__init__()
        self.ngf = ngf
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 2 * self.ngf),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 2 * self.ngf),
        )
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * self.ngf, out_channels=self.ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.ngf),
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 2 * self.ngf, 7, 7)
        return self.convt(x)


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, output_embedding=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if not output_embedding:
            out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


class DCGAN_Generator(nn.Module): # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    def __init__(self, nc=1, nz=100, ngf=64, freeze_w=False, alpha_mobility=1):
        super(DCGAN_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh())
        self.nz = nz
        self.alpha_mobility = alpha_mobility

        self.freeze_w = freeze_w
        if self.freeze_w:
            Freeze_convnet_weights(self.main)



    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        output = self.main(input)
        return output


class DCGAN_Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64, alpha_mobility=1):
        super(DCGAN_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())
        self.alpha_mobility = alpha_mobility


    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


def DCGAN_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    for i in range(10):
        torch.manual_seed(i)
        # G = MLP(n_hidden_layers=1, n_hidden_neurons=32, input_dim=1, output_dim=1, output_shape=None, type="D", verbose=True, freeze_w=True)
        G = MLP(n_hidden_layers=1, n_hidden_neurons=32, input_dim=2, output_dim=1, output_shape=None, type="G", verbose=True)
        with torch.no_grad():
            print(G)
            print(G.parameters())
            print("\n0")
            print(list(G.parameters()))
            print("\n1")

            for params in G.parameters():
                print(params)
                print(params.requires_grad)

            print(list(params for params in G.parameters() if params.requires_grad))
            # print(G.hidden_layer)
            # print(G.hidden_layer.weight)
            # print(G.hidden_layer.bias)
            # print(G.deeper_hidden_layers)
            # print(G.output_layer)
            # print(G.output_layer.weight)
            # print(G.output_layer.bias)

            # print("weight norm", torch.norm(G.hidden_layer.weight, p=2, dim=1))

            BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(G.hidden_layer.weight, G.hidden_layer.bias, G.output_layer.weight)
            print("BP_directions_G\n", BP_directions_G)
            print("BP_signed_distances_G\n", BP_signed_distances_G)
            print("BP_delta_slopes_G\n", BP_delta_slopes_G)

            print(G.state_dict())
            G_hidden_layer_weights_np = G.state_dict()["hidden_layer.weight"].cpu().numpy()
            # print(G_hidden_layer_weights_np)
            # # print([G_hidden_layer_weight for G_hidden_layer_weight in G_hidden_layer_weights_np])
            # a = torch.from_numpy(np.array([[1., 2.], [3., 1.]])).float()
            # print("G(a)", G(a))

            x_linspace = np.linspace(-10, 10, 101)
            y_linspace = G(torch.from_numpy(x_linspace[:, np.newaxis]).float()).cpu().detach().numpy().ravel()
            plt.plot(x_linspace, y_linspace)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("NN functions at Initialization")
    plt.axes().set_aspect(0.5)
    # plt.show()
    plt.savefig(os.path.join("Pre", f"init.png"), dpi=400)




