import torch
from torch import autograd
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader
from torch.nn import functional as F

import numpy as np
import pandas as pd
import scipy.sparse.linalg
import scipy
from scipy.io import savemat

import time
import datetime
import glob
import os
import platform
import argparse
import random


from utils import *
from deepviz import *
from deepviz_pre import DeepVisuals_Pre
import deepviz_real
import Run_Aux_Training
# from Run_Aux_Training import Get_classification_logit_prob_class, Load_aux_classifier
from Run_GAN_Training import Get_models, transform, real_data_folder
from Synthetic_Dataset import Synthetic_Dataset


# import matlab.engine
#
# def Estimate_Lipschitz(weight_path, alpha=0, beta=1, form="neuron", split=False, parallel=False, verbose=False, split_size=2, num_neurons=100, num_workers=0, num_decision_vars=10):
#     """ https://github.com/arobey1/LipSDP """
#
#     timer = Timer()
#     # print("Estimate_Lipschitz")
#     eng = matlab.engine.start_matlab()
#     eng.addpath(r'matlab_engine')
#     eng.addpath(r'matlab_engine/weight_utils')
#     eng.addpath(r'matlab_engine/error_messages')
#     eng.addpath(r'examples/saved_weights')
#
#     # print("eng initialized")
#     network = {
#         'alpha': matlab.double([alpha]),
#         'beta': matlab.double([beta]),
#         'weight_path': [weight_path],
#     }
#
#     lip_params = {
#         'formulation': form,
#         'split': matlab.logical([split]),
#         'parallel': matlab.logical([parallel]),
#         'verbose': matlab.logical([verbose]),
#         'split_size': matlab.double([split_size]),
#         'num_neurons': matlab.double([num_neurons]),
#         'num_workers': matlab.double([num_workers]),
#         'num_dec_vars': matlab.double([num_decision_vars])
#     }
#
#     L = eng.solve_LipSDP(network, lip_params, nargout=1)
#     timer.Print(msg=f"L = {L}")
#
#     return L


weight_mat_dir = "mat_dir"
total_row_num = 0


def Get_init_type(video_title, summary_dir, task_dir, exp_data_name, from_expname_list):
    init_type = None
    if from_expname_list:
        b4_expname_list = []
        with open("b4exp_names.txt") as f:
            b4_expname_list = [expname[:-1] for expname in f.readlines()]

        #         print(b4_expname_list, b4_expname_list)
        if exp_data_name in b4_expname_list:
            init_type = "b4"
        else:
            init_type = "b123"
    else:
        all_expdata_filename = os.listdir(os.path.join(summary_dir, task_dir))
        all_init_expdata_filename = [expdata_filename for expdata_filename in all_expdata_filename if "init" in expdata_filename]
        #         print(all_init_expdata_filename)
        #         print(video_title)

        config_matched_filename_list = [expdata_filename for expdata_filename in all_init_expdata_filename if video_title in expdata_filename]
        #         print(config_matched_filename_list)
        if len(config_matched_filename_list) > 0:
            if any(["_init_b4_" in expdata_filename for expdata_filename in config_matched_filename_list]):
                init_type = "b4"
            elif any(["_init_b123_" in expdata_filename for expdata_filename in config_matched_filename_list]):
                init_type = "b123"
            else:
                init_type = None
        else:
            init_type = None

    #         print(config_matched_filename_list, init_type)

    return init_type


def Get_params_init_from_file(video_title, summary_dir, task_dir, exp_data_name, task_type, from_expname_list=False):
    init_type = Get_init_type(video_title, summary_dir, task_dir, exp_data_name, from_expname_list)
    #     print("init_type", init_type)
    #     print("video_title", video_title)
    if "_insc-relu" not in video_title:
        video_title += "_insc-relu"
    #     print("video_title", video_title)
    filename_to_search_for_init = video_title + f"_init_{init_type}"
    #     print("filename_to_search_for_init", filename_to_search_for_init)
    all_expdata_filename = os.listdir(os.path.join(summary_dir, task_dir))
    init_matched_filename_list = [expdata_filename for expdata_filename in all_expdata_filename if filename_to_search_for_init in expdata_filename]
    #     print("init_matched_filename_list", init_matched_filename_list)
    if len(init_matched_filename_list) > 0:
        init_matched_filename = init_matched_filename_list[0]
        #         print("init_matched_filename", init_matched_filename)
        if task_type == "pre":
            deepviz = DeepVisuals_Pre()
            deepviz.Load_data(init_matched_filename, data_folder=os.path.join(summary_dir, task_dir))
        elif task_type == "2d":
            try:
                deepviz = DeepVisuals_2D(data_folder=os.path.join(summary_dir, task_dir))
                status = deepviz.Load_data(init_matched_filename)
                print("New deepviz")
            except:
                try:
                    deepviz = DeepVisuals_2D()
                    status = deepviz.Load_data(init_matched_filename, data_folder=os.path.join(summary_dir, task_dir))
                    print("Old deepviz")
                except:
                    print("Data broken")
                    return
        elif task_type == "real":
            deepviz = deepviz_real.DeepVisuals_real(data_folder=os.path.join(summary_dir, task_dir))
            status = deepviz.Load_data(init_matched_filename)
        else:
            raise NotImplementedError("Unknown type of deepviz")

        #         print(deepviz.data_container_dict)
        idd_init = next(deepviz.data_container_generator)
        #         print(idd_init)
        return idd_init["state_dict_G"], idd_init["state_dict_D"]

    else:
        return None, None



def Save_deepviz_data_as_csv(result_path, exp_data_name, task_dir, task_type, set_header=False, debug=False, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), save_weight_mat=False, calculate_NTK=False, existing_row_num=0, summary_dir="Summaries", add_new_columns=False, only_last_t=True, calculate_per_num=10, calculate_all_metrics=False, get_init_from_expname_list=False):
    status = -1

    task_type = task_type.lower()
    if task_type == "pre":
        deepviz = DeepVisuals_Pre()
        deepviz.Load_data(exp_data_name, data_folder=os.path.join(summary_dir, task_dir))
        status = 0
    elif task_type in "2d":
        try:
            deepviz = DeepVisuals_2D(data_folder=os.path.join(summary_dir, task_dir))
            status = deepviz.Load_data(exp_data_name)
            print("New deepviz")
        except:
            try:
                deepviz = DeepVisuals_2D()
                status = deepviz.Load_data(exp_data_name, data_folder=os.path.join(summary_dir, task_dir))
                print("Old deepviz")
            except:
                print("Data broken")
                return
    elif task_type == "real":
        deepviz = deepviz_real.DeepVisuals_real(data_folder=os.path.join(summary_dir, task_dir))
        status = deepviz.Load_data(exp_data_name)
    else:
        raise NotImplementedError("Unknown type of deepviz")

    if status == -1:
        return

    deepviz.Calculate_max_t()
    args = deepviz.attr["args"]
    if not hasattr(args, "alpha_mobility_D"):
        args.alpha_mobility_D = 1

    print("max_t =", deepviz.attr["max_t"])
    print(args)

    if task_type == "2d" and args.data not in ["mnist", "cifar"]:
        dataset_config = f"{args.data}-{args.mog_scale}-{args.mog_std}"
        aux_classifier_loaded, real_data_prob, real_mode_num = Run_Aux_Training.Load_aux_classifier(dataset_config)

    if task_type == "pre":
        hyperparameter_list = ["data", "scale_y", "opt_type", "noise_std", "iteration", "seed", "layers", "hidden", "lr", "num_samples", "alt_act_prop", "alt_act_factor", "loss"]
    elif task_type == "2d":
        hyperparameter_list = ["data", "mog_scale", "mog_std", "plot_lim_x", "method", "opt_type", "use_spectral_norm", "z_dim", "iteration", "batch_size", "seed", "alpha_mobility", "alpha_mobility_D", "g_layers", "d_layers", "g_hidden", "d_hidden", "g_lr", "d_lr", "gamma", "zeta"]
    elif task_type == "real":
        hyperparameter_list = ["data", "method", "opt_type", "use_spectral_norm", "z_dim", "iteration", "batch_size", "seed", "alpha_mobility", "alpha_mobility_D", "covered_mode_num", "G_base_filter_num", "D_base_filter_num", "g_lr", "d_lr", "gamma"]
    else:
        raise NotImplementedError("Unknown type of deepviz")

    results_dict = dict.fromkeys(hyperparameter_list)
    for key in results_dict:
        val = None
        if key == "use_spectral_norm":
            if not hasattr(args, "use_spectral_norm"):
                val = False
            else:
                val = getattr(args, "use_spectral_norm")
        elif key == "opt_type":
            opt_type = getattr(args, "opt_type")
            if opt_type in ["rmsprop", "adam"]:
                if hasattr(args, "adapt_x") and hasattr(args, "adapt_y"):
                    if args.adapt_x:
                        opt_type += f"-x"
                    if args.adapt_y:
                        opt_type += f"-y"
            val = opt_type
        elif key == "zeta":
            if args.opt_type == "sgd":
                val = 0.0
            if args.opt_type in ["rmsprop", "adam"]:
                if hasattr(args, key):
                    val = getattr(args, key)
                else:
                    val = 1.0
        else:
            val = getattr(args, key) if hasattr(args, key) else None

        results_dict[key] = val

    max_t = 5 if debug else deepviz.attr["max_t"]
    if max_t == 0:
        max_t = 1000
    print("max_t", max_t)
    """ Initialize nets for calculating NTK change """
    if args.data in ["cifar"]:
        output_shape = [3, 32, 32]
    elif args.data in ["mnist"]:
        output_shape = [1, 28, 28]
    else:
        output_shape = None

    video_title, args, verboseprint = Get_experiment_name_from_args(args)

    G, D = Get_models(args, output_shape)
    G = G.to(device)
    D = D.to(device)
    test_data_num = 256
    n_grad_norm_sample = 10
    n_interpolate = 21
    n_grad_approx = 20
    std_grad_approx = 1e-4
    grad_norm_amplitude = 18

    z_test = deepviz.attr["z_test"][:test_data_num, :]
    z_test_torch = torch.from_numpy(z_test).to(device)
    z_test_torch_no_grad = torch.from_numpy(z_test).to(device)
    z_test_torch_no_grad.requires_grad = False

    rng = np.random.RandomState(seed=args.seed)
    if args.data not in ["mnist", "cifar"]:
        dataset = Synthetic_Dataset(args.data, rng, std=args.mog_std, scale=args.mog_scale, sample_per_mode=1000)
        x_test, x_test_label = dataset.sample(test_data_num, with_neg_samples=True)
        # x_test_real_torch = torch.from_numpy(x_test).to(device).float()
    else:
        if args.data == "mnist":
            dataset = MNIST(root=os.path.join(real_data_folder, "mnist"), train=False, transform=transform, download=True)
            dataloader = DataLoader(dataset=dataset, batch_size=test_data_num, shuffle=True, num_workers=0, drop_last=True)
            x_test, x_test_label = next(iter(dataloader))
            # x_test_real_torch = x_test.to(device).float()
        elif args.data == "cifar":
            dataset = CIFAR10(root=os.path.join(real_data_folder, "cifar"), train=True, download=True, transform=transform)
            dataloader = DataLoader(dataset=dataset, batch_size=test_data_num, shuffle=True, num_workers=0, drop_last=True)
        else:
            raise  NotImplementedError("x_test_real of CIFAR10 is not implemented.")

    np.random.seed(0)

    n_params_G = torch.cat([_.flatten() for _ in G.parameters()]).view(-1).shape[0]
    n_params_D = torch.cat([_.flatten() for _ in D.parameters()]).view(-1).shape[0]

    ntk_G_1_init = None
    ntk_G_2_init = None
    ntk_D_init = None

    act_pattern_G_init = None
    act_pattern_D_init = None
    relu_output_vec_G_init = None
    relu_output_vec_D_init = None

    params_G_init = None
    params_G_prev = None

    params_D_init = None
    params_D_prev = None

    BP_directions_G_prev = None
    BP_signed_distances_G_prev = None
    BP_delta_slopes_G_prev = None
    BP_directions_D_prev = None
    BP_signed_distances_D_prev = None
    BP_delta_slopes_D_prev = None

    BP_directions_G_init = None
    BP_signed_distances_G_init = None
    BP_delta_slopes_G_init = None
    BP_directions_D_init = None
    BP_signed_distances_D_init = None
    BP_delta_slopes_D_init = None

    weights_G_prev = None
    biases_G_prev = None
    vweights_G_prev = None
    weights_D_prev = None
    biases_D_prev = None
    vweights_prev = None

    weights_G_init = None
    biases_G_init = None
    vweights_G_init = None
    weights_D_init = None
    biases_D_init = None
    vweights_init = None

    update_angle_BP_directions_G_tot_deg = 0
    update_BP_distances_G_tot_norm = 0
    update_BP_delta_slopes_G_tot_norm = 0
    update_angle_BP_directions_D_tot_deg = 0
    update_BP_distances_D_tot_norm = 0
    update_BP_delta_slopes_D_tot_norm = 0


    BP_Lp_norm_list = np.linspace(1, 2, 11)

    BP_distances_G_mean_init = np.nan
    BP_delta_slopes_G_Lp_norm_init_list = [np.nan for p in BP_Lp_norm_list]
    BP_distances_D_mean_init = np.nan
    BP_delta_slopes_D_Lp_norm_init_list = [np.nan for p in BP_Lp_norm_list]

    weights_G_norm_init = np.nan
    biases_G_norm_init = np.nan
    vweights_G_norm_init = np.nan
    weights_D_norm_init = np.nan
    biases_D_norm_init = np.nan
    vweights_D_norm_init = np.nan


    params_G_norm_init = np.nan
    params_D_norm_init = np.nan

    try:
        idd = next(deepviz.data_container_generator)
    except:
        print("no data found")
        return

    idd_next = None
    is_last_t = False
    state_dict_G = None
    state_dict_D = None
    arch = args.arch
    print("n_params_G", n_params_G, "n_params_D", n_params_D, "arch", arch)

    for t in range(max_t):
        """ Save data to csv for each iteration """

        if t % (max_t // 5) == 0:
            print(t, end=", ")
        if t == max_t - 1:
            print(t)
        try:
            idd_next = next(deepviz.data_container_generator)
        except:
            print(f"Read idd failed at iter {t}")
            is_last_t = True

        if (not (is_last_t or t == 0 or ((not only_last_t) and t % calculate_per_num == 0))) or idd is None:
            idd = idd_next
            continue

        global total_row_num
        total_row_num += 1

        if total_row_num <= existing_row_num and t > 0:
            idd = idd_next
            continue

        print("t", t, "is_last_t", is_last_t)
        BP_directions_G = None
        BP_signed_distances_G = None
        BP_delta_slopes_G = None
        BP_directions_D = None
        BP_signed_distances_D = None
        BP_delta_slopes_D = None

        state_dict_G_tmp = idd.get("state_dict_G", None)
        state_dict_D_tmp = idd.get("state_dict_D", None)
        if state_dict_G_tmp is not None:
            state_dict_G = state_dict_G_tmp
        if state_dict_D_tmp is not None:
            state_dict_D = state_dict_D_tmp

        if t == 0 and (state_dict_G is None or state_dict_D is None) and args.data in ["mnist", "cifar"]:
            state_dict_G, state_dict_D = Get_params_init_from_file(video_title, summary_dir, task_dir, exp_data_name, task_type, from_expname_list=get_init_from_expname_list)
            if state_dict_G is not None or state_dict_D is not None:
                print("\n==================== init found ====================\n")

        if state_dict_G is not None:
            G.load_state_dict(state_dict_G)
        if state_dict_D is not None:
            D.load_state_dict(state_dict_D)

        # x_test_torch = G(z_test_torch)
        x_test_torch = Get_output_by_batch(G, z_test_torch)
        x_test_torch_no_grad = x_test_torch.detach()


        results_dict.update({"iter": idd["iter"]})
        """ =========================================================================================================== """
        """ Product of delta-slopes in each output dimension """
        """ =========================================================================================================== """
        delta_slope_inner_prod = np.nan
        if ((not add_new_columns) or calculate_all_metrics) and (is_last_t or t == 0 or (not only_last_t)) and arch == "mlp":
        # if (add_new_columns or calculate_all_metrics) and (is_last_t or t == 0 or (not only_last_t)) and arch == "mlp":
            if BP_directions_G is None or BP_signed_distances_G is None or BP_delta_slopes_G is None:
                BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"], state_dict_G["hidden_layer.bias"], state_dict_G["output_layer.weight"])

            delta_slope_inner_prod = np.dot(np.abs(BP_delta_slopes_G[0, :]), np.abs(BP_delta_slopes_G[1, :])) / args.alpha_mobility ** 2

        delta_slope_inner_prod_dict = {"delta_slope_inner_prod": delta_slope_inner_prod}
        results_dict.update(delta_slope_inner_prod_dict)

        timer.Print(msg="Product of delta-slopes in each output dimension")

        """ =========================================================================================================== """
        """ BP density weighted by delta-slopes """
        """ =========================================================================================================== """
        BP_density_range = BP_density_range_dim_1 = BP_density_range_dim_2 = np.nan
        BP_G_entropy = BP_G_1_entropy = BP_G_2_entropy = BP_D_entropy = np.nan
        affine_BP_G_prop = np.nan
        if ((not add_new_columns) or calculate_all_metrics) and (is_last_t or t == 0 or (not only_last_t)) and arch == "mlp":
            """ Generator """
            if BP_directions_G is None or BP_signed_distances_G is None or BP_delta_slopes_G is None:
                BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"], state_dict_G["hidden_layer.bias"], state_dict_G["output_layer.weight"])

            """     BP metrics """
            affine_BP_G_prop = np.sum(np.abs(BP_signed_distances_G) > 3 * args.z_std) / len(BP_signed_distances_G)
            BP_loc_G = BP_directions_G * BP_signed_distances_G[:, np.newaxis]

            axis_range = [-2.1 * args.z_std, 2.1 * args.z_std]
            x_range = axis_range
            y_range = axis_range
            bandwidth = 0.05
            n_interpolate_KDE = 121j

            """ BP density - dim 0 """
            try:
                weight_list_pos = np.maximum(BP_delta_slopes_G[0, :], 0)
                weight_list_neg = np.minimum(BP_delta_slopes_G[0, :], 0)
                xx, yy, density_KDE_pos = KDE(x_range, y_range, BP_loc_G, weight_list_pos, bw_method=bandwidth, n=n_interpolate_KDE)
                xx, yy, density_KDE_neg = KDE(x_range, y_range, BP_loc_G, weight_list_neg, bw_method=bandwidth, n=n_interpolate_KDE)
                density_KDE = density_KDE_pos - density_KDE_neg
                density_KDE_normalized = np.abs(density_KDE) / np.linalg.norm(density_KDE.ravel(), ord=1)
                density_KDE_normalized_vec = density_KDE_normalized.ravel()

                BP_density_range_dim_1 = np.max(density_KDE.ravel()) - np.min(density_KDE.ravel())
                BP_G_1_entropy = Shannon_entropy(density_KDE_normalized_vec)
            except:
                pass

            """ BP density - dim 1 """
            try:
                weight_list_pos = np.maximum(BP_delta_slopes_G[1, :], 0)
                weight_list_neg = np.minimum(BP_delta_slopes_G[1, :], 0)
                xx, yy, density_KDE_pos = KDE(x_range, y_range, BP_loc_G, weight_list_pos, bw_method=bandwidth, n=n_interpolate_KDE)
                xx, yy, density_KDE_neg = KDE(x_range, y_range, BP_loc_G, weight_list_neg, bw_method=bandwidth, n=n_interpolate_KDE)
                density_KDE = density_KDE_pos - density_KDE_neg
                density_KDE_normalized = np.abs(density_KDE) / np.linalg.norm(density_KDE.ravel(), ord=1)
                density_KDE_normalized_vec = density_KDE_normalized.ravel()

                BP_density_range_dim_2 = np.max(density_KDE.ravel()) - np.min(density_KDE.ravel())
                BP_G_2_entropy = Shannon_entropy(density_KDE_normalized_vec)
            except:
                pass

            BP_density_range = np.mean([BP_density_range_dim_1, BP_density_range_dim_2])
            BP_G_entropy = np.mean([BP_G_1_entropy, BP_G_2_entropy])

        BP_density_range_dict = {"BP_density_range": BP_density_range, "BP_density_range_dim_1": BP_density_range_dim_1, "BP_density_range_dim_2": BP_density_range_dim_2}
        BP_entropy_dict = {"BP_G_entropy": BP_G_entropy, "BP_G_1_entropy": BP_G_1_entropy, "BP_G_2_entropy": BP_G_2_entropy, "BP_D_entropy": BP_D_entropy, "affine_BP_G_prop": affine_BP_G_prop}

        results_dict.update(BP_density_range_dict)
        results_dict.update(BP_entropy_dict)

        timer.Print(msg="BP density weighted by delta-slopes")


        """ =========================================================================================================== """
        """ Losses of optimization """
        """ =========================================================================================================== """
        loss_G = idd["loss_G"].item()
        loss_D = idd["loss_D"].item()
        results_dict.update({"loss_G": loss_G, "loss_D": loss_D})

        """ =========================================================================================================== """
        """ Estimate global Lipschitz constant """
        """ =========================================================================================================== """
        L_G = L_D = np.nan
        if ((t % 10 == 0 and total_row_num > existing_row_num and args.data not in ["mnist", "cifar"]) \
                or (args.data in ["mnist", "cifar"] and idd["iter"] == 600000)) and save_weight_mat:
            """ Saving G, D weights """
            mat_name_G = f"{exp_data_name}_mat{t}_tot{total_row_num}_G.mat"
            mat_name_D = f"{exp_data_name}_mat{t}_tot{total_row_num}_D.mat"
            weights_G = [state_dict_G["hidden_layer.weight"].cpu().double().numpy(), state_dict_G["output_layer.weight"].cpu().double().numpy()]
            weights_D = [state_dict_D["hidden_layer.weight"].cpu().double().numpy(), state_dict_D["output_layer.weight"].cpu().double().numpy()]
            os.makedirs(os.path.join(summary_dir, task_dir, weight_mat_dir), exist_ok=True)
            weight_path_G = os.path.join(summary_dir, task_dir, weight_mat_dir, mat_name_G)
            weight_path_D = os.path.join(summary_dir, task_dir, weight_mat_dir, mat_name_D)
            savemat(weight_path_G, {"weights": np.array(weights_G, dtype=np.object)})
            savemat(weight_path_D, {"weights": np.array(weights_D, dtype=np.object)})

            if args.data in ["mnist", "cifar"]:
                form = "network-rand"
                form = "layer"
                num_neurons = 1
            else:
                form = "neuron"
                num_neurons = 100

            # L_G = Estimate_Lipschitz(weight_path_G, form=form, num_neurons=num_neurons)
            # L_D = Estimate_Lipschitz(weight_path_D, form=form, num_neurons=num_neurons)

        results_dict.update({"L_G": L_G, "L_D": L_D})
        timer.Print(msg="Estimate global Lipschitz constant")

        """ =========================================================================================================== """
        """ Ratio of neurons whose activation pattern change """
        """ =========================================================================================================== """
        rel_act_diff_G = rel_act_diff_D = np.nan
        if (total_row_num > existing_row_num) \
                and (state_dict_G is not None and state_dict_D is not None) \
                and ((not add_new_columns) or calculate_all_metrics):
        # if (total_row_num > existing_row_num or (act_pattern_G_init is None or act_pattern_D_init is None)) \
        #         and (state_dict_G is not None and state_dict_D is not None) \
        #         and ((add_new_columns) or calculate_all_metrics):
            if arch == "mlp":
                """ Compute * first, then + , then >. """
                act_pattern_G = (state_dict_G['hidden_layer.weight'] @ z_test_torch.transpose(0, 1) + state_dict_G['hidden_layer.bias'].unsqueeze(
                    1) > 0).int().detach().cpu().numpy().ravel()
                act_pattern_D = (state_dict_D['hidden_layer.weight'] @ x_test_torch.view(x_test_torch.shape[0], -1).transpose(0, 1) + state_dict_D[
                    'hidden_layer.bias'].unsqueeze(1) > 0).int().detach().cpu().numpy().ravel()

                if act_pattern_G_init is None:
                    act_pattern_G_init = act_pattern_G
                if act_pattern_D_init is None:
                    act_pattern_D_init = act_pattern_D

                rel_act_diff_G = np.linalg.norm(act_pattern_G - act_pattern_G_init, ord=1) / len(act_pattern_G_init)
                rel_act_diff_D = np.linalg.norm(act_pattern_D - act_pattern_D_init, ord=1) / len(act_pattern_D_init)
            else:
                relu_output_vec_G = Get_all_relu_layer_output_as_vector(G, z_test_torch, get_pre_act=True)
                relu_output_vec_D = Get_all_relu_layer_output_as_vector(D, x_test_torch, get_pre_act=True)
                if relu_output_vec_G_init is None:
                    relu_output_vec_G_init = relu_output_vec_G
                if relu_output_vec_D_init is None:
                    relu_output_vec_D_init = relu_output_vec_D

                rel_act_diff_G = Get_relative_activation_pattern_change_from_relu_outputs(relu_output_vec_G_init, relu_output_vec_G, from_pre_act=True)
                rel_act_diff_D = Get_relative_activation_pattern_change_from_relu_outputs(relu_output_vec_D_init, relu_output_vec_D, from_pre_act=True)

        results_dict.update({"rel_act_diff_G": rel_act_diff_G, "rel_act_diff_D": rel_act_diff_D})

        timer.Print(msg="Ratio of neurons whose activation pattern change")

        """ =========================================================================================================== """
        """ (Mean, Std) of norms of change of parameters (with per-group stats and BPs for MLPs) """
        """ =========================================================================================================== """
        update_angle_BP_directions_G_tot_deg_mean = np.nan
        update_BP_distances_G_tot_norm_mean = np.nan
        update_BP_delta_slopes_G_tot_norm_mean = np.nan
        update_angle_BP_directions_D_tot_deg_mean = np.nan
        update_BP_distances_D_tot_norm_mean = np.nan
        update_BP_delta_slopes_D_tot_norm_mean = np.nan

        update_angle_BP_directions_G_tot_deg_std = np.nan
        update_BP_distances_G_tot_norm_std = np.nan
        update_BP_delta_slopes_G_tot_norm_std = np.nan
        update_angle_BP_directions_D_tot_deg_std = np.nan
        update_BP_distances_D_tot_norm_std = np.nan
        update_BP_delta_slopes_D_tot_norm_std = np.nan

        update_weights_G_norm_all = np.nan
        update_biases_G_norm_all = np.nan
        update_vweights_G_norm_all = np.nan
        update_weights_D_norm_all = np.nan
        update_biases_D_norm_all = np.nan
        update_vweights_D_norm_all = np.nan

        update_weights_G_tot_norm_mean = np.nan
        update_biases_G_tot_norm_mean = np.nan
        update_vweights_G_tot_norm_mean = np.nan
        update_weights_D_tot_norm_mean = np.nan
        update_biases_D_tot_norm_mean = np.nan
        update_vweights_D_tot_norm_mean = np.nan

        update_weights_G_tot_norm_std = np.nan
        update_biases_G_tot_norm_std = np.nan
        update_vweights_G_tot_norm_std = np.nan
        update_weights_D_tot_norm_std = np.nan
        update_biases_D_tot_norm_std = np.nan
        update_vweights_D_tot_norm_std = np.nan

        BP_distances_G_mean_final = np.nan
        BP_delta_slopes_G_Lp_norm_final_list = [np.nan for p in BP_Lp_norm_list]
        BP_distances_D_mean_final = np.nan
        BP_delta_slopes_D_Lp_norm_final_list = [np.nan for p in BP_Lp_norm_list]

        weights_G_norm_final = np.nan
        biases_G_norm_final = np.nan
        vweights_G_norm_final = np.nan
        weights_D_norm_final = np.nan
        biases_D_norm_final = np.nan
        vweights_D_norm_final = np.nan

        params_G_norm_final = np.nan
        params_D_norm_final = np.nan

        update_params_G_norm_all = np.nan
        update_params_G_tot_norm_mean = np.nan
        update_params_G_tot_norm_std = np.nan

        update_params_D_norm_all = np.nan
        update_params_D_tot_norm_mean = np.nan
        update_params_D_tot_norm_std = np.nan

        if state_dict_G is not None and state_dict_D is not None and (not add_new_columns or calculate_all_metrics):
        # if state_dict_G is not None and state_dict_D is not None and (add_new_columns or calculate_all_metrics):
            params_G = torch.cat([_.flatten() for _ in G.parameters()]).view(1, -1)
            params_D = torch.cat([_.flatten() for _ in D.parameters()]).view(1, -1)

            if params_G_prev is None:
                params_G_prev = params_G
                params_G_init = params_G

            if params_D_prev is None:
                params_D_prev = params_D
                params_D_init = params_D

            if t == 0:
                params_G_norm_init = torch.norm(params_G).item()
                params_D_norm_init = torch.norm(params_D).item()

            params_G_norm_final = torch.norm(params_G).item()
            params_D_norm_final = torch.norm(params_D).item()

            update_params_G_norm_all = torch.norm(params_G - params_G_init, dim=1, p=2).item()
            update_params_D_norm_all = torch.norm(params_D - params_D_init, dim=1, p=2).item()

            update_params_G_list = torch.abs(params_G - params_G_init).detach().cpu().numpy()
            update_params_D_list = torch.abs(params_D - params_D_init).detach().cpu().numpy()

            update_params_G_tot_norm_mean = np.mean(update_params_G_list.ravel())
            update_params_G_tot_norm_std = np.mean(update_params_D_list.ravel())

            update_params_D_tot_norm_mean = np.std(update_params_G_list.ravel())
            update_params_D_tot_norm_std = np.std(update_params_D_list.ravel())

            """     Updating parameters """
            params_G_prev = params_G
            params_D_prev = params_D

            if arch == "mlp":
                """ ----------------------------------------------------------------------------------------------------------- """
                """ (Mean, Std) of norms of change of BPs and params per group """
                """ ----------------------------------------------------------------------------------------------------------- """
                if BP_directions_G is None or BP_signed_distances_G is None or BP_delta_slopes_G is None:
                    BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"], state_dict_G["hidden_layer.bias"], state_dict_G["output_layer.weight"])
                if BP_directions_D is None or BP_signed_distances_D is None or BP_delta_slopes_D is None:
                    BP_directions_D, BP_signed_distances_D, BP_delta_slopes_D = Get_BP_params(state_dict_D["hidden_layer.weight"], state_dict_D["hidden_layer.bias"], state_dict_D["output_layer.weight"])

                if BP_directions_G_prev is None:
                    BP_directions_G_prev = BP_directions_G
                    BP_directions_G_init = BP_directions_G

                if BP_signed_distances_G_prev is None:
                    BP_signed_distances_G_prev = BP_signed_distances_G
                    BP_signed_distances_G_init = BP_signed_distances_G

                if BP_delta_slopes_G_prev is None:
                    BP_delta_slopes_G_prev = BP_delta_slopes_G
                    BP_delta_slopes_G_init = BP_delta_slopes_G

                if BP_directions_D_prev is None:
                    BP_directions_D_prev = BP_directions_D
                    BP_directions_D_init = BP_directions_D

                if BP_signed_distances_D_prev is None:
                    BP_signed_distances_D_prev = BP_signed_distances_D
                    BP_signed_distances_D_init = BP_signed_distances_D

                if BP_delta_slopes_D_prev is None:
                    BP_delta_slopes_D_prev = BP_delta_slopes_D
                    BP_delta_slopes_D_init = BP_delta_slopes_D

                if weights_G_prev is None:
                    weights_G_prev = state_dict_G["hidden_layer.weight"]
                    weights_G_init = state_dict_G["hidden_layer.weight"]

                if biases_G_prev is None:
                    biases_G_prev = state_dict_G["hidden_layer.bias"]
                    biases_G_init = state_dict_G["hidden_layer.bias"]

                if vweights_G_prev is None:
                    vweights_G_prev = state_dict_G["output_layer.weight"]
                    vweights_G_init = state_dict_G["output_layer.weight"]

                if weights_D_prev is None:
                    weights_D_prev = state_dict_D["hidden_layer.weight"]
                    weights_D_init = state_dict_D["hidden_layer.weight"]

                if biases_D_prev is None:
                    biases_D_prev = state_dict_D["hidden_layer.bias"]
                    biases_D_init = state_dict_D["hidden_layer.bias"]

                if vweights_prev is None:
                    vweights_prev = state_dict_D["output_layer.weight"]
                    vweights_init = state_dict_D["output_layer.weight"]



                """     Calculating BP descriptors """
                """         Calculate BP update norm """
                if t == 0:
                    BP_distances_G_mean_init = np.mean(np.abs(BP_signed_distances_G.ravel())) # average of absolute distance to origin
                    BP_delta_slopes_G_Lp_norm_init_list = [np.linalg.norm(BP_delta_slopes_G.ravel(), ord=p) for p in BP_Lp_norm_list]
                    BP_distances_D_mean_init = np.mean(np.abs(BP_signed_distances_D.ravel()))  # average of absolute distance to origin
                    BP_delta_slopes_D_Lp_norm_init_list = [np.linalg.norm(BP_delta_slopes_D.ravel(), ord=p) for p in BP_Lp_norm_list]

                    weights_G_norm_init = np.linalg.norm(state_dict_G["hidden_layer.weight"].cpu().numpy().ravel())
                    biases_G_norm_init = np.linalg.norm(state_dict_G["hidden_layer.bias"].cpu().numpy().ravel())
                    vweights_G_norm_init = np.linalg.norm(state_dict_G["output_layer.weight"].cpu().numpy().ravel())
                    weights_D_norm_init = np.linalg.norm(state_dict_D["hidden_layer.weight"].cpu().numpy().ravel())
                    biases_D_norm_init = np.linalg.norm(state_dict_D["hidden_layer.bias"].cpu().numpy().ravel())
                    vweights_D_norm_init = np.linalg.norm(state_dict_D["output_layer.weight"].cpu().numpy().ravel())


                BP_distances_G_mean_final = np.mean(np.abs(BP_signed_distances_G.ravel()))  # average of absolute distance to origin
                BP_delta_slopes_G_Lp_norm_final_list = [np.linalg.norm(BP_delta_slopes_G.ravel(), ord=p) for p in BP_Lp_norm_list]
                BP_distances_D_mean_final = np.mean(np.abs(BP_signed_distances_D.ravel()))  # average of absolute distance to origin
                BP_delta_slopes_D_Lp_norm_final_list = [np.linalg.norm(BP_delta_slopes_D.ravel(), ord=p) for p in BP_Lp_norm_list]

                weights_G_norm_final = np.linalg.norm(state_dict_G["hidden_layer.weight"].cpu().numpy().ravel())
                biases_G_norm_final = np.linalg.norm(state_dict_G["hidden_layer.bias"].cpu().numpy().ravel())
                vweights_G_norm_final = np.linalg.norm(state_dict_G["output_layer.weight"].cpu().numpy().ravel())
                weights_D_norm_final = np.linalg.norm(state_dict_D["hidden_layer.weight"].cpu().numpy().ravel())
                biases_D_norm_final = np.linalg.norm(state_dict_D["hidden_layer.bias"].cpu().numpy().ravel())
                vweights_D_norm_final = np.linalg.norm(state_dict_D["output_layer.weight"].cpu().numpy().ravel())

                update_angle_BP_directions_G = Rad_to_Deg(np.arccos(Cossim(BP_directions_G, BP_directions_G_prev, output_vec=True)))
                update_BP_distances_G = BP_signed_distances_G - BP_signed_distances_G_prev
                update_BP_delta_slopes_G = BP_delta_slopes_G - BP_delta_slopes_G_prev
                update_angle_BP_directions_D = Rad_to_Deg(np.arccos(Cossim(BP_directions_D, BP_directions_D_prev, output_vec=True)))
                update_BP_distances_D = BP_signed_distances_D - BP_signed_distances_D_prev
                update_BP_delta_slopes_D = BP_delta_slopes_D - BP_delta_slopes_D_prev

                update_angle_BP_directions_G_tot_deg += np.linalg.norm(update_angle_BP_directions_G, ord=1)
                update_BP_distances_G_tot_norm += np.linalg.norm(update_BP_distances_G)
                update_BP_delta_slopes_G_tot_norm += np.linalg.norm(update_BP_delta_slopes_G)
                update_angle_BP_directions_D_tot_deg += np.linalg.norm(update_angle_BP_directions_D, ord=1)
                update_BP_distances_D_tot_norm += np.linalg.norm(update_BP_distances_D)
                update_BP_delta_slopes_D_tot_norm += np.linalg.norm(update_BP_delta_slopes_D)

                update_angle_BP_directions_G_tot_list = np.abs(Rad_to_Deg(np.arccos(Cossim(BP_directions_G, BP_directions_G_init, output_vec=True))))
                update_BP_distances_G_tot_list = np.abs(BP_signed_distances_G - BP_signed_distances_G_init)
                update_BP_delta_slopes_G_tot_list = np.linalg.norm(BP_delta_slopes_G - BP_delta_slopes_G_init, axis=0)
                update_angle_BP_directions_D_tot_list = np.abs(Rad_to_Deg(np.arccos(Cossim(BP_directions_D, BP_directions_D_init, output_vec=True))))
                update_BP_distances_D_tot_list = np.abs(BP_signed_distances_D - BP_signed_distances_D_init)
                update_BP_delta_slopes_D_tot_list = np.linalg.norm(BP_delta_slopes_D - BP_delta_slopes_D_init, axis=0)

                update_angle_BP_directions_G_tot_deg_mean = np.mean(update_angle_BP_directions_G_tot_list.ravel())
                update_BP_distances_G_tot_norm_mean = np.mean(update_BP_distances_G_tot_list.ravel())
                update_BP_delta_slopes_G_tot_norm_mean = np.mean(update_BP_delta_slopes_G_tot_list.ravel())
                update_angle_BP_directions_D_tot_deg_mean = np.mean(update_angle_BP_directions_D_tot_list.ravel())
                update_BP_distances_D_tot_norm_mean = np.mean(update_BP_distances_D_tot_list.ravel())
                update_BP_delta_slopes_D_tot_norm_mean = np.mean(update_BP_delta_slopes_D_tot_list.ravel())

                update_angle_BP_directions_G_tot_deg_std = np.std(update_angle_BP_directions_G_tot_list.ravel())
                update_BP_distances_G_tot_norm_std = np.std(update_BP_distances_G_tot_list.ravel())
                update_BP_delta_slopes_G_tot_norm_std = np.std(update_BP_delta_slopes_G_tot_list.ravel())
                update_angle_BP_directions_D_tot_deg_std = np.std(update_angle_BP_directions_D_tot_list.ravel())
                update_BP_distances_D_tot_norm_std = np.std(update_BP_distances_D_tot_list.ravel())
                update_BP_delta_slopes_D_tot_norm_std = np.std(update_BP_delta_slopes_D_tot_list.ravel())

                """         Calculate parameters update norm """
                update_weights_G = (state_dict_G["hidden_layer.weight"] - weights_G_init).cpu().numpy()
                update_biases_G = (state_dict_G["hidden_layer.bias"] - biases_G_init).cpu().numpy()
                update_vweights_G = (state_dict_G["output_layer.weight"] - vweights_G_init).cpu().numpy()
                update_weights_D = (state_dict_D["hidden_layer.weight"] - weights_D_init).cpu().numpy()
                update_biases_D = (state_dict_D["hidden_layer.bias"] - biases_D_init).cpu().numpy()
                update_vweights_D = (state_dict_D["output_layer.weight"] - vweights_init).cpu().numpy()

                update_weights_G_tot_list = np.linalg.norm(update_weights_G, axis=1)
                update_biases_G_tot_list = np.abs(update_biases_G)
                update_vweights_G_tot_list = np.linalg.norm(update_vweights_G, axis=0)
                update_weights_D_tot_list = np.linalg.norm(update_weights_D, axis=1)
                update_biases_D_tot_list = np.abs(update_biases_D)
                update_vweights_D_tot_list = np.linalg.norm(update_vweights_D, axis=0)

                update_weights_G_norm_all = np.linalg.norm(update_weights_G.ravel())
                update_biases_G_norm_all = np.linalg.norm(update_biases_G.ravel())
                update_vweights_G_norm_all = np.linalg.norm(update_vweights_G.ravel())
                update_weights_D_norm_all = np.linalg.norm(update_weights_D.ravel())
                update_biases_D_norm_all = np.linalg.norm(update_biases_D.ravel())
                update_vweights_D_norm_all = np.linalg.norm(update_vweights_D.ravel())

                update_weights_G_tot_norm_mean = np.mean(update_weights_G_tot_list.ravel())
                update_biases_G_tot_norm_mean = np.mean(update_biases_G_tot_list.ravel())
                update_vweights_G_tot_norm_mean = np.mean(update_vweights_G_tot_list.ravel())
                update_weights_D_tot_norm_mean = np.mean(update_weights_D_tot_list.ravel())
                update_biases_D_tot_norm_mean = np.mean(update_biases_D_tot_list.ravel())
                update_vweights_D_tot_norm_mean = np.mean(update_vweights_D_tot_list.ravel())

                update_weights_G_tot_norm_std = np.std(update_weights_G_tot_list.ravel())
                update_biases_G_tot_norm_std = np.std(update_biases_G_tot_list.ravel())
                update_vweights_G_tot_norm_std = np.std(update_vweights_G_tot_list.ravel())
                update_weights_D_tot_norm_std = np.std(update_weights_D_tot_list.ravel())
                update_biases_D_tot_norm_std = np.std(update_biases_D_tot_list.ravel())
                update_vweights_D_tot_norm_std = np.std(update_vweights_D_tot_list.ravel())

                """     Updating BPs """
                BP_directions_G_prev = BP_directions_G
                BP_signed_distances_G_prev = BP_signed_distances_G
                BP_delta_slopes_G_prev = BP_delta_slopes_G
                BP_directions_D_prev = BP_directions_D
                BP_signed_distances_D_prev = BP_signed_distances_D
                BP_delta_slopes_D_prev = BP_delta_slopes_D

                """     Updating parameters """
                weights_G_prev = state_dict_G["hidden_layer.weight"]
                biases_G_prev = state_dict_G["hidden_layer.bias"]
                vweights_G_prev = state_dict_G["output_layer.weight"]
                weights_D_prev = state_dict_D["hidden_layer.weight"]
                biases_D_prev = state_dict_D["hidden_layer.bias"]
                vweights_prev = state_dict_D["output_layer.weight"]

        fc_2_layer_NN_params_dict = {"weights_G_norm_init": weights_G_norm_init, "biases_G_norm_init": biases_G_norm_init, "vweights_G_norm_init": vweights_G_norm_init, "weights_D_norm_init": weights_D_norm_init, "biases_D_norm_init": biases_D_norm_init, "vweights_D_norm_init": vweights_D_norm_init, "weights_G_norm_final": weights_G_norm_final, "biases_G_norm_final": biases_G_norm_final, "vweights_G_norm_final": vweights_G_norm_final, "weights_D_norm_final": weights_D_norm_final, "biases_D_norm_final": biases_D_norm_final, "vweights_D_norm_final": vweights_D_norm_final}

        fc_2_layer_BP_params_dict = {"update_angle_BP_directions_G_tot_deg": update_angle_BP_directions_G_tot_deg, "update_BP_distances_G_tot_norm": update_BP_distances_G_tot_norm, "update_BP_delta_slopes_G_tot_norm": update_BP_delta_slopes_G_tot_norm, "update_angle_BP_directions_D_tot_deg": update_angle_BP_directions_D_tot_deg, "update_BP_distances_D_tot_norm": update_BP_distances_D_tot_norm, "update_BP_delta_slopes_D_tot_norm": update_BP_delta_slopes_D_tot_norm, "update_angle_BP_directions_G_tot_deg_mean": update_angle_BP_directions_G_tot_deg_mean, "update_BP_distances_G_tot_norm_mean": update_BP_distances_G_tot_norm_mean, "update_BP_delta_slopes_G_tot_norm_mean": update_BP_delta_slopes_G_tot_norm_mean , "update_angle_BP_directions_D_tot_deg_mean": update_angle_BP_directions_D_tot_deg_mean, "update_BP_distances_D_tot_norm_mean": update_BP_distances_D_tot_norm_mean, "update_BP_delta_slopes_D_tot_norm_mean": update_BP_delta_slopes_D_tot_norm_mean, "update_angle_BP_directions_G_tot_deg_std": update_angle_BP_directions_G_tot_deg_std, "update_BP_distances_G_tot_norm_std": update_BP_distances_G_tot_norm_std, "update_BP_delta_slopes_G_tot_norm_std": update_BP_delta_slopes_G_tot_norm_std, "update_angle_BP_directions_D_tot_deg_std": update_angle_BP_directions_D_tot_deg_std, "update_BP_distances_D_tot_norm_std": update_BP_distances_D_tot_norm_std, "update_BP_delta_slopes_D_tot_norm_std": update_BP_delta_slopes_D_tot_norm_std, "update_weights_G_tot_norm_mean": update_weights_G_tot_norm_mean, "update_biases_G_tot_norm_mean": update_biases_G_tot_norm_mean, "update_vweights_G_tot_norm_mean": update_vweights_G_tot_norm_mean, "update_weights_D_tot_norm_mean": update_weights_D_tot_norm_mean, "update_biases_D_tot_norm_mean": update_biases_D_tot_norm_mean, "update_vweights_D_tot_norm_mean": update_vweights_D_tot_norm_mean, "update_weights_G_tot_norm_std": update_weights_G_tot_norm_std, "update_biases_G_tot_norm_std": update_biases_G_tot_norm_std, "update_vweights_G_tot_norm_std": update_vweights_G_tot_norm_std, "update_weights_D_tot_norm_std": update_weights_D_tot_norm_std, "update_biases_D_tot_norm_std": update_biases_D_tot_norm_std, "update_vweights_D_tot_norm_std": update_vweights_D_tot_norm_std, "update_weights_G_norm_all": update_weights_G_norm_all, "update_biases_G_norm_all": update_biases_G_norm_all, "update_vweights_G_norm_all": update_vweights_G_norm_all, "update_weights_D_norm_all": update_weights_D_norm_all, "update_biases_D_norm_all": update_biases_D_norm_all, "update_vweights_D_norm_all": update_vweights_D_norm_all, "BP_distances_G_mean_init": BP_distances_G_mean_init, "BP_distances_D_mean_init": BP_distances_D_mean_init, "BP_distances_G_mean_final": BP_distances_G_mean_final, "BP_distances_D_mean_final": BP_distances_D_mean_final}
        BP_delta_slopes_G_Lp_norm_init_dict = dict(zip([f"BP_delta_slopes_G_L{p:.1f}_norm_init" for p in BP_Lp_norm_list], BP_delta_slopes_G_Lp_norm_init_list))
        BP_delta_slopes_D_Lp_norm_init_dict = dict(zip([f"BP_delta_slopes_D_L{p:.1f}_norm_init" for p in BP_Lp_norm_list], BP_delta_slopes_D_Lp_norm_init_list))
        BP_delta_slopes_G_Lp_norm_final_dict = dict(zip([f"BP_delta_slopes_G_L{p:.1f}_norm_final" for p in BP_Lp_norm_list], BP_delta_slopes_G_Lp_norm_final_list))
        BP_delta_slopes_D_Lp_norm_final_dict = dict(zip([f"BP_delta_slopes_D_L{p:.1f}_norm_final" for p in BP_Lp_norm_list], BP_delta_slopes_D_Lp_norm_final_list))

        results_dict.update(fc_2_layer_NN_params_dict)
        results_dict.update(fc_2_layer_BP_params_dict)
        results_dict.update(BP_delta_slopes_G_Lp_norm_init_dict)
        results_dict.update(BP_delta_slopes_D_Lp_norm_init_dict)
        results_dict.update(BP_delta_slopes_G_Lp_norm_final_dict)
        results_dict.update(BP_delta_slopes_D_Lp_norm_final_dict)

        param_change_dict = {"update_params_G_norm_all": update_params_G_norm_all, "update_params_G_tot_norm_mean": update_params_G_tot_norm_mean, "update_params_G_tot_norm_std": update_params_G_tot_norm_std, "update_params_D_norm_all": update_params_D_norm_all, "update_params_D_tot_norm_mean": update_params_D_tot_norm_mean, "update_params_D_tot_norm_std": update_params_D_tot_norm_std, "params_G_norm_init": params_G_norm_init, "params_G_norm_final": params_G_norm_final, "params_D_norm_init": params_D_norm_init, "params_D_norm_final": params_D_norm_final}
        results_dict.update(param_change_dict)

        timer.Print(msg="Param changes")

        """ =========================================================================================================== """
        """ NTK effective rank  https://github.com/tfjgeorge/nngeometry + https://github.com/tfjgeorge/ntk_alignment """
        """ =========================================================================================================== """
        ntk_G_centered_effective_rank = ntk_D_centered_effective_rank = np.nan
        if (total_row_num > existing_row_num) \
                and (state_dict_G is not None and state_dict_D is not None) \
                and ((not add_new_columns) or calculate_all_metrics) and arch == "mlp":

            ntk_G_centered = Get_NTK_using_nngeometry(G, args.x_dim, z_test_torch_no_grad, centering=True)
            ntk_D_centered = Get_NTK_using_nngeometry(D, 1, x_test_torch_no_grad, centering=True)

            ntk_G_centered_effective_rank, _ = Effective_rank_torch(ntk_G_centered)
            ntk_D_centered_effective_rank, _ = Effective_rank_torch(ntk_D_centered)

        ntk_effective_rank_dict = {"ntk_G_centered_effective_rank": ntk_G_centered_effective_rank, "ntk_D_centered_effective_rank": ntk_D_centered_effective_rank}
        results_dict.update(ntk_effective_rank_dict)

        timer.Print(msg="NTK effective rank")

        """ =========================================================================================================== """
        """ NTK change https://github.com/bobby-he/Neural_Tangent_Kernel/ """
        """ =========================================================================================================== """
        ntk_G_1_reldiff = ntk_G_2_reldiff = ntk_D_reldiff = np.nan
        if calculate_NTK and total_row_num > existing_row_num:
            grad_mat_G_1 = torch.zeros(test_data_num, n_params_G).to(device)
            grad_mat_G_2 = torch.zeros(test_data_num, n_params_G).to(device)
            grad_mat_D = torch.zeros(test_data_num, n_params_D).to(device)
            for i, (z_test_i, x_test_i) in enumerate(zip(z_test, x_test)):
                z_test_i_torch = torch.from_numpy(z_test_i).to(device)
                x_test_i_torch = torch.from_numpy(x_test_i).to(device).float().view(1, -1)
                # x_fake_test_i = G(z_test_i_torch)
                x_fake_test_i = Get_output_by_batch(G, z_test_i_torch)
                # D_x_test_i = D(x_test_i_torch)
                D_x_test_i = Get_output_by_batch(D, x_test_i_torch)

                grad_mat_G_1[i, :] = torch.cat([_.flatten() for _ in torch.autograd.grad(x_fake_test_i[0], G.parameters(), retain_graph=True)]).view(-1)
                grad_mat_G_2[i, :] = torch.cat([_.flatten() for _ in torch.autograd.grad(x_fake_test_i[1], G.parameters(), retain_graph=True)]).view(-1)
                grad_mat_D[i, :] = torch.cat([_.flatten() for _ in torch.autograd.grad(D_x_test_i, D.parameters(), retain_graph=True)]).view(-1)

            ntk_G_1 = grad_mat_G_1 @ grad_mat_G_1.T
            ntk_G_2 = grad_mat_G_2 @ grad_mat_G_2.T
            ntk_D = grad_mat_D @ grad_mat_D.T

            if ntk_G_1_init is None:
                ntk_G_1_init = ntk_G_1
            if ntk_G_2_init is None:
                ntk_G_2_init = ntk_G_2
            if ntk_D_init is None:
                ntk_D_init = ntk_D

            ntk_G_1_reldiff = np.linalg.norm((ntk_G_1 - ntk_G_1_init).cpu().detach().numpy().ravel()) / (np.linalg.norm(ntk_G_1_init.cpu().detach().numpy().ravel()) + EPS_BUFFER)
            ntk_G_2_reldiff = np.linalg.norm((ntk_G_2 - ntk_G_2_init).cpu().detach().numpy().ravel()) / (np.linalg.norm(ntk_G_2_init.cpu().detach().numpy().ravel()) + EPS_BUFFER)
            ntk_D_reldiff = np.linalg.norm((ntk_D - ntk_D_init).cpu().detach().numpy().ravel()) / (np.linalg.norm(ntk_D.cpu().detach().numpy().ravel()) + EPS_BUFFER)

        ntk_metrics_dict = {"ntk_G_1_reldiff": ntk_G_1_reldiff, "ntk_G_2_reldiff": ntk_G_2_reldiff, "ntk_D_reldiff": ntk_D_reldiff}
        results_dict.update(ntk_metrics_dict)

        timer.Print(msg="Calculated BP summaries")


        """ =========================================================================================================== """
        """ Jacobian frobenius norm of dG/dz, and gradient norm of dD/dx """
        """ =========================================================================================================== """
        # if args.data in ["grid5", "random9-6_2"]:
        #     selected_mode_tuple_list = [(4, 5), (4, 8), (5, 8), (7, 5)]
        # else:
        #     selected_mode_tuple_list = [(2, 3), (3, 4), (2, 5), (4, 5)]

        dG_dz_avg_between_modes_dict = {}
        dG_dz_mean = np.nan
        dD_dx_mean = np.nan
        KL = KL_mode = covered_mode_num = prop_neg_samples = np.nan

        proportion_outliers_interpolated = np.nan
        mean_of_dG_dz_interpolated_max = np.nan
        mean_of_dD_dx_interpolated_max = np.nan
        """ --- Find z such that G(z) is classified as the following modes """
        if state_dict_G is not None and ((not add_new_columns) or calculate_all_metrics) and (is_last_t or t == 0 or (not only_last_t)):
        # if state_dict_G is not None and ((add_new_columns) or calculate_all_metrics) and (is_last_t or t == 0 or (not only_last_t)):
            z_test = deepviz.attr["z_test"]

            if args.data not in ["mnist", "cifar"]:
                dG_dz_test = Estimate_Jacobian_norm(G, z_test_torch, n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx)
                dD_dx_test = Estimate_Jacobian_norm(D, x_test_torch, n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx)

                dG_dz_mean = np.mean(dG_dz_test.ravel())
                dD_dx_mean = np.mean(dD_dx_test.ravel())
            else:
                try:
                    dG_dz_test = Estimate_Jacobian_norm(G, z_test_torch[:64, ...], n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx, using_loop=True)
                    dD_dx_test = Estimate_Jacobian_norm(D, x_test_torch[:64, ...], n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx, using_loop=True)

                    dG_dz_mean = np.mean(dG_dz_test.ravel())
                    dD_dx_mean = np.mean(dD_dx_test.ravel())
                except:
                    print("RuntimeError: CUDA out of memory")

            if args.data not in ["mnist", "cifar"]:
                with torch.no_grad():
                    pred_logits = aux_classifier_loaded(x_test_torch)
                    pred_probs = F.softmax(pred_logits, dim=1).data.cpu().numpy()
                    pred_data_prob = np.mean(pred_probs, axis=0)
                    pred_classes = np.argmax(pred_probs, axis=1)
                    pred_classes_one_hot = np.zeros_like(pred_probs)
                    pred_classes_one_hot[np.arange(len(pred_probs)), pred_classes] = 1
                    pred_classes_count = np.sum(pred_classes_one_hot, axis=0)
                    covered_mode_num = np.sum(pred_classes_count[:-1] > 0)


                KL = KL_divergence(real_data_prob, pred_data_prob)

                if KL > 25.43:
                    print(f"{'=' * 20}\nKL {KL}, log KL {np.log(KL)}\nreal_data_prob {real_data_prob}\npred_data_prob {pred_data_prob}\n{'=' * 20}")

                KL_mode = KL_divergence(real_data_prob[:-1], Normalize(pred_data_prob[:-1]))
                print("real prob", real_data_prob[:-1])
                print("pred prob", Normalize(pred_data_prob[:-1]))
                print("KL_mode", KL_mode)
                prop_neg_samples = pred_classes_count[-1] / x_test_torch.shape[0]

                if (add_new_columns) or calculate_all_metrics:
                    pred_labels = np.argmax(pred_probs, axis=1)

                    mode_to_z_dict = {}
                    for c in range(dataset.n):
                        mode_to_z_dict[c] = []
                    for z, label in zip(z_test, pred_labels):
                        if label < dataset.n:
                            mode_to_z_dict[label].append(z)

                    mode_list = np.arange(dataset.n)
                    prob_vec = Normalize([len(mode_to_z_dict[mode]) for mode in mode_list], ord=1)
                    print("prob_vec", prob_vec)
                    if np.abs(np.sum(prob_vec) - 1) > 1e-4 or np.sum(prob_vec > 0) <= 1:
                        proportion_outliers_interpolated = 1
                    else:
                        selected_mode_tuple_list = [np.random.choice(mode_list, size=2, replace=False, p=prob_vec) for _ in range(100)]

                        """ --- Interpolate between z's and calculate gradient norm """
                        dG_dz_interpolated_max_list = []
                        dD_dx_interpolated_max_list = []

                        num_realistic_sample = 0
                        for ss, selected_mode_tuple in enumerate(selected_mode_tuple_list):
                            if len(mode_to_z_dict[selected_mode_tuple[0]]) > 0 and len(mode_to_z_dict[selected_mode_tuple[1]]) > 0:
                                for ll in range(n_grad_norm_sample):
                                    random_z_0 = random.choice(mode_to_z_dict[selected_mode_tuple[0]])
                                    random_z_1 = random.choice(mode_to_z_dict[selected_mode_tuple[1]])
                                    z_interpolated = np.linspace(random_z_0, random_z_1, n_interpolate)
                                    z_interpolated_torch = torch.from_numpy(z_interpolated).to(device)
                                    z_interpolated_torch.requires_grad_(True)

                                    """ JFN """
                                    dG_dz_interpolated = Estimate_Jacobian_norm(G, z_interpolated_torch, n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx)
                                    dG_dz_interpolated_max_list.append(np.max(dG_dz_interpolated.ravel()))

                                    """ GN """
                                    # fake_out_on_z_interpolated = G(z_interpolated_torch)
                                    fake_out_on_z_interpolated = Get_output_by_batch(G, z_interpolated_torch)
                                    dD_dx_interpolated = Estimate_Jacobian_norm(D, fake_out_on_z_interpolated, n_grad_approx=n_grad_approx, std_grad_approx=std_grad_approx)
                                    dD_dx_interpolated_max_list.append(np.max(dD_dx_interpolated.ravel()))

                                    """ Get realistic samples """
                                    pred_logits_on_G_z_interpolated = aux_classifier_loaded(fake_out_on_z_interpolated)
                                    pred_probs_on_G_z_interpolated = F.softmax(pred_logits_on_G_z_interpolated, dim=1).data.cpu().numpy()
                                    pred_labels_on_G_z_interpolated = np.argmax(pred_probs_on_G_z_interpolated, axis=1)
                                    realistic_samples = pred_labels_on_G_z_interpolated[(pred_labels_on_G_z_interpolated == selected_mode_tuple[0]) | (pred_labels_on_G_z_interpolated == selected_mode_tuple[1])]
                                    num_realistic_sample += len(realistic_samples)


                        proportion_outliers_interpolated = (len(selected_mode_tuple_list) * n_grad_norm_sample * n_interpolate - num_realistic_sample) / (len(selected_mode_tuple_list) * n_grad_norm_sample * n_interpolate)
                        mean_of_dG_dz_interpolated_max = np.mean(dG_dz_interpolated_max_list)
                        mean_of_dD_dx_interpolated_max = np.mean(dD_dx_interpolated_max_list)


        timer.Print(msg="Calculated JFN")
        """ =========================================================================================================== """
        """ Saving data """
        """ =========================================================================================================== """

        if task_type == "2d":
            results_dict.update({"KL": KL, "KL_mode": KL_mode, "covered_mode_num": covered_mode_num, "dG_dz_mean": dG_dz_mean, "dD_dx_mean": dD_dx_mean, "prop_neg_samples": prop_neg_samples, "mean_of_dG_dz_interpolated_max": mean_of_dG_dz_interpolated_max, "mean_of_dD_dx_interpolated_max": mean_of_dD_dx_interpolated_max, "proportion_outliers_interpolated": proportion_outliers_interpolated})
        elif task_type == "real":
            results_dict.update({"is_mean": idd["inception_score"][0], "is_std": idd["inception_score"][1], "dG_dz_mean": dG_dz_mean, "dD_dx_mean": dD_dx_mean})

        key_list = [key for key in results_dict]
        results_str_list = [str(results_dict[key]) for key in results_dict]

        if existing_row_num > 0:
            set_header = False

        print("total_row_num", total_row_num)
        print("existing_row_num", existing_row_num)
        if total_row_num > existing_row_num:
            with open(result_path, "a") as f:
                if set_header:
                    f.write(f"{','.join(key_list)}\n")
                    set_header = False
                f.write(f"{','.join(results_str_list)}\n")

        timer.Print(msg="Saved to csv")
        idd = idd_next
        if is_last_t:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", type=str, default="", help="task_dir")
    parser.add_argument("--add_new_columns", type=int, default=-1, help="whether to add_new_columns")
    parser.add_argument("--calculate_all_metrics", type=int, default=-1, help="whether to calculate_all_metrics")
    parser.add_argument("--summary_dir", type=str, default="", help="summary_dir")
    parser.add_argument("--task_type", type=str, default="", help="task_type")
    parser.add_argument("--write_local", action='store_true', help="whether to write results at local directory")
    parser.add_argument("--get_init_from_expname_list", action='store_true', help="whether to get_init_from_expname_list")

    args = parser.parse_args()
    task_dir = "GAN_training_results_examples"

    if args.task_type == "":
        if task_dir.split("_")[1] in ["mnist", "cifar", "cifar10"]:
            task_type = "real"
        else:
            task_type = "2d"
    else:
        task_type = args.task_type


    summary_dir = "Summaries"

    if args.summary_dir != "":
        summary_dir = args.summary_dir

    if args.task_dir != "":
        task_dir = args.task_dir

    debug = False
    save_weight_mat = False
    append_to_csv = False
    calculate_all_metrics = False
    add_new_columns = False
    only_last_t = True

    calculate_per_num = 10

    special_name = ""
    # special_name = "_new-BP-updates"

    if args.calculate_all_metrics != -1:
        calculate_all_metrics = bool(args.calculate_all_metrics)

    if args.add_new_columns != -1:
        add_new_columns = bool(args.add_new_columns)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    set_header = True
    exp_data_path_list = glob.glob(os.path.join(summary_dir, task_dir, f"*.pickle"))
    print("exp_data_path_list", exp_data_path_list)
    print("summary_dir", summary_dir)
    print("task_dir", task_dir)

    if platform.system() == "Darwin":
        print("Using MacOS.")
        exp_data_name_list = [exp_data_path.split("/")[-1][:-7] for exp_data_path in exp_data_path_list]
    elif platform.system() == "Linux":
        print("Using Linux.")
        exp_data_name_list = [exp_data_path.split("/")[-1][:-7] for exp_data_path in exp_data_path_list]
    else:
        print("Using Windows.")
        exp_data_name_list = [exp_data_path.split("\\")[-1][:-7] for exp_data_path in exp_data_path_list]

    exp_data_name_list = [exp_data_name for exp_data_name in exp_data_name_list if "_init_b" not in exp_data_name]

    print(exp_data_name_list)
    timer = Timer()

    """ Detect if results.csv already exists """
    new_column_suffix = ""
    if add_new_columns:
        new_column_suffix = "_new-columns"

    if calculate_all_metrics:
        new_column_suffix += "_all-metrics"

    if args.write_local:
        output_path = "Summaries"
    else:
        output_path = summary_dir

    result_path = os.path.join(output_path, task_dir, f"results{new_column_suffix}{special_name}.csv")
    os.makedirs(os.path.join(output_path, task_dir), exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result_line_list = f.readlines()
        existing_row_num = len(result_line_list) - 1
    else:
        existing_row_num = 0

    if append_to_csv:
        existing_row_num = 0
        set_header = False

    for i, exp_data_name in enumerate(exp_data_name_list):
        print(f"[Processing {i + 1} / {len(exp_data_name_list)} ...] {exp_data_name}")
        Save_deepviz_data_as_csv(result_path, exp_data_name, task_dir, task_type, set_header, device=device, debug=debug, save_weight_mat=save_weight_mat, existing_row_num=existing_row_num, summary_dir=summary_dir, add_new_columns=add_new_columns, only_last_t=only_last_t, calculate_per_num=calculate_per_num, calculate_all_metrics=calculate_all_metrics, get_init_from_expname_list=args.get_init_from_expname_list)
        timer.Print()
        if set_header:
            set_header = False
