from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.autograd as autograd
from torch.nn import functional as F

import numpy as np
import scipy.sparse.linalg
import glob
import random

from deepviz import *
from Run_Aux_Training import Get_classification_logit_prob_class, Load_aux_classifier, aux_model_folder
from Run_GAN_Training import Get_models, transform, real_data_folder
from Synthetic_Dataset import Synthetic_Dataset
from ComputationalTools import Timer
from BPs import Get_2D_line_points
from models import MLP

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

import scipy.stats as st

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['savefig.facecolor'] = "0.8"
# font = {"size": 50}
#
# matplotlib.rc('font', **font)
plt.rcParams['font.size'] = '80'


if platform.system() == "Darwin":
    print("Using MacOS.")
    plt.rcParams['animation.ffmpeg_path'] = "/usr/local/bin/ffmpeg"
elif platform.system() == "Linux":
    print("Using Linux.")
    plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"
else:
    print("Using Windows.")
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/juyil/ffmpeg/bin/ffmpeg.exe'

summary_dir = summary_folder = "Summaries"

def Get_diverging_color(value, cmap_pos="Blues", cmap_neg="Reds", color_scale=15):
    cmap_name = cmap_pos if value >= 0 else cmap_neg
    return cm.get_cmap(cmap_name)(np.abs(value) * color_scale)

def KDE(x_range, y_range, point_list, weight_list=None, bw_method=None):
    xmin, xmax = x_range[0], x_range[1]
    ymin, ymax = y_range[0], y_range[1]
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] # grid
    positions = np.vstack([xx.ravel(), yy.ravel()])
    x = point_list[:, 0]; y = point_list[:, 1] # data pints
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values, weights=weight_list, bw_method=bw_method)
    density_KDE = np.reshape(kernel(positions).T, xx.shape) # density on grid
    return xx, yy, density_KDE

def Classify_experiments_name_by_str(name, by_str):
    print(f"by_str: {by_str}")
    return name.replace(f"_{by_str}_", "_")

def Classify_experiments_name_by_str_list(name, by_str_list):
    for by_str in by_str_list:
        name = Classify_experiments_name_by_str(name, by_str)
    return name

def Classify_experiments_name_by_keyword(name, by_keyword):
    if by_keyword == "opt_type":
        by_str_list = ["sgd", "rmsprop_gm0.999"]
    else:
        print("Unknown by_keyword")
        raise NotImplementedError

    return Classify_experiments_name_by_str_list(name, by_str_list)


class DeepVisuals_Summary_weighted_BP_KDE_and_grad_norm(DeepVisuals_2D):
    def __init__(self, task_dir, dataset_name, cumul=False):
        super(DeepVisuals_Summary_weighted_BP_KDE_and_grad_norm, self).__init__(name=task_dir, handle=-1)
        self.dataset_name = dataset_name
        self.cumul = cumul

        self.exp_data_file_path_list = [ele for ele in glob.glob(os.path.join(summary_folder, task_dir, f"{dataset_name}*.pickle"))]
        self.gamma_list = [0.8, 0.9, 0.99, 0.999, 0.9999, 1]
        self.seed_list = [1, 2, 3, 4]
        exp_config_list = [f"gamma={gamma},seed={seed}" for gamma in self.gamma_list for seed in self.seed_list]
        self.gamma_seed_deepviz_dict = dict.fromkeys(exp_config_list)

        for exp_data_file_path in self.exp_data_file_path_list:
            # try:
            #     deepviz = DeepVisuals_2D(data_folder="")
            #     deepviz.Load_data(exp_data_file_path)
            #     print("New deepviz")
            # except:
            #     deepviz = DeepVisuals_2D()
            #     deepviz.Load_data(exp_data_file_path, data_folder="")
            #     print("Old deepviz")

            deepviz = DeepVisuals_2D(data_folder="")
            deepviz.Load_data(exp_data_file_path)
            print("New deepviz")

            if deepviz.attr["args"].opt_type == "sgd":
                deepviz.attr["args"].gamma = 1

            gamma = deepviz.attr["args"].gamma
            seed = deepviz.attr["args"].seed
            self.gamma_seed_deepviz_dict[f"gamma={gamma},seed={seed}"] = deepviz

        self.args = list(self.gamma_seed_deepviz_dict.values())[0].attr["args"]
        self.dataset = Synthetic_Dataset(self.args.data, std=self.args.mog_std, scale=self.args.mog_scale, sample_per_mode=1000)
        self.dataset_config = f"{self.args.data}-{self.args.mog_scale}-{self.args.mog_std}"
        self.aux_classifier_loaded, self.real_data_prob, self.real_mode_num = Load_aux_classifier(self.dataset_config)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def Init_figure(self):
        self.ims = []
        self.ax_dict = {}
        span_figure_r = 3
        span_figure_c = 2


        row_per_sec = 2
        col_per_sec = 5
        row_num = len(self.gamma_list) * row_per_sec
        col_num = len(self.seed_list) * col_per_sec
        base_size = 6
        width_to_height_ratio = 1
        plot_lim = 30

        self.fig = plt.figure(figsize=(base_size * col_num * width_to_height_ratio, base_size * row_num))

        for i, seed in enumerate(self.seed_list):
            for j, gamma in enumerate(self.gamma_list):
                self.ax_dict[f"gamma{gamma}_seed{seed}_output"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j, col_per_sec * i), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_text"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j + 1, col_per_sec * i), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_0"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j, col_per_sec * i + 1), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_1"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j + 1, col_per_sec * i + 1), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_0"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j, col_per_sec * i + 2), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_1"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j + 1, col_per_sec * i + 2), rowspan=1, colspan=1)

                self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_aspect("equal")
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_0"].set_aspect("equal")
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_1"].set_aspect("equal")
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_0"].set_aspect("equal")
                self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_1"].set_aspect("equal")

                self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_0"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j, col_per_sec * i + 3), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_2"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j + 1, col_per_sec * i + 3), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_1"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j, col_per_sec * i + 4), rowspan=1, colspan=1)
                self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_3"] = plt.subplot2grid((row_num, col_num), (row_per_sec * j + 1, col_per_sec * i + 4), rowspan=1, colspan=1)

        self.axis_range = [-1 * self.args.z_std, 1 * self.args.z_std]
        self.output_range = [-1.5, 1.5]
        self.x_range = self.axis_range
        self.y_range = self.axis_range
        self.KL_range = [-2, 3]
        self.n_grad_norm_sample = 3
        self.n_interpolate = 101
        self.grad_norm_amplitude = 18
        self.x_out_density_amplitude = 0.5
        self.density_amplitude = 50
        self.bandwidth = 0.1

        for ax_name in self.ax_dict:
            self.ax_dict[ax_name].set_xlim(self.axis_range)
            self.ax_dict[ax_name].set_ylim(self.axis_range)
            self.ax_dict[ax_name].set_yticklabels([])

            for label in (self.ax_dict[ax_name].get_xticklabels() + self.ax_dict[ax_name].get_yticklabels()):
                label.set_fontsize(20)

        self.fig.set_tight_layout(True)
        plt.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)
        self.fig.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)
        print("Figure intialized.")

        # plt.savefig(os.path.join(summary_dir, f"{task_dir}_updates.png"), dpi=400)

    def Plot_step(self, t, loading=False):
        """ Begin plotting """
        imgs = []

        for i, seed in enumerate(self.seed_list):
            for j, gamma in enumerate(self.gamma_list):
                deepviz = self.gamma_seed_deepviz_dict[f"gamma={gamma},seed={seed}"]
                if deepviz is None:
                    continue
                args = deepviz.attr["args"]
                if args.data in ["cifar"]:
                    output_shape = [3, 32, 32]
                elif args.data in ["mnist"]:
                    output_shape = [1, 28, 28]
                else:
                    output_shape = None

                if not hasattr(args, "alpha_mobility_D"):
                    args.alpha_mobility_D = 1

                G, D = Get_models(args, output_shape)
                G = G.to(self.device)
                # D = D.to(self.device)

                idd = next(deepviz.data_container_generator)
                state_dict_G = idd["state_dict_G"]
                G.load_state_dict(state_dict_G)

                """ log KL text """
                pred_data_prob, covered_mode_num = Get_classification_logit_prob_class(self.aux_classifier_loaded, idd["x_out"])
                KL = KL_divergence(self.real_data_prob, pred_data_prob)
                log_KL = np.log(KL)
                KL_text = self.ax_dict[f"gamma{gamma}_seed{seed}_text"].text(0.4, 0.5, f"log KL = {log_KL:.4f}\niteration = {t * args.plot_iter}", {"ha": "center", "va": "center"},
                                                        horizontalalignment="left", verticalalignment="top",
                                                        transform=self.ax_dict[f"gamma{gamma}_seed{seed}_text"].transAxes, fontsize=30)
                imgs.append(KL_text)

                """ KL bar plot """
                mode_label_list = [str(i) for i in range(1, self.dataset.n + 1)] + ["-"]
                mode_freq_bar = self.ax_dict[f"gamma{gamma}_seed{seed}_text"].bar(x=list(range(1, 1 + len(pred_data_prob))), height=pred_data_prob, color="#000000", label="Predicted Pr for each mode", tick_label=mode_label_list)
                imgs.extend(mode_freq_bar)

                self.ax_dict[f"gamma{gamma}_seed{seed}_text"].yaxis.tick_right()
                self.ax_dict[f"gamma{gamma}_seed{seed}_text"].set_xlim(0, 1 + len(pred_data_prob))
                self.ax_dict[f"gamma{gamma}_seed{seed}_text"].set_ylim(0, 1)

                """ Gradient norm w.r.t. z between modes """
                """ --- Find z such that G(z) is classified as the following modes """
                z_test = deepviz.attr["z_test"]
                x_out = G(torch.from_numpy(z_test).to(self.device))
                with torch.no_grad():
                    pred_logits = self.aux_classifier_loaded(x_out)
                    # print("pred_logits", pred_logits.shape, pred_logits)
                    pred_probs = F.softmax(pred_logits, dim=1).data.cpu().numpy()
                    # print("pred_probs", pred_probs.shape, pred_probs)

                pred_labels = np.argmax(pred_probs, axis=1)

                mode_to_z_dict = {}
                for c in range(self.dataset.n + 1):
                    mode_to_z_dict[c] = []
                for z, label in zip(z_test, pred_labels):
                    mode_to_z_dict[label].append(z)

                """ --- Interpolate between z's and calculate gradient norm """
                selected_mode_tuple_list = [(4, 5), (4, 8), (5, 8), (7, 5)]
                for ss, selected_mode_tuple in enumerate(selected_mode_tuple_list):
                    if len(mode_to_z_dict[selected_mode_tuple[0]]) > 0 and len(mode_to_z_dict[selected_mode_tuple[1]]) > 0:
                        dG_dz_interpolated_list = []
                        for ll in range(self.n_grad_norm_sample):
                            random_z_0 = random.choice(mode_to_z_dict[selected_mode_tuple[0]])
                            random_z_1 = random.choice(mode_to_z_dict[selected_mode_tuple[1]])
                            z_interpolated = np.linspace(random_z_0, random_z_1, self.n_interpolate)
                            z_interpolated_torch = torch.from_numpy(z_interpolated).to(self.device)
                            z_interpolated_torch.requires_grad_(True)

                            fake_out_on_z_interpolated = G(z_interpolated_torch)
                            dx_0_dz = autograd.grad(fake_out_on_z_interpolated[:, 0], z_interpolated_torch, create_graph=True, retain_graph=True,
                                                    grad_outputs=torch.ones_like(fake_out_on_z_interpolated[:, 0]))[0]
                            dx_1_dz = autograd.grad(fake_out_on_z_interpolated[:, 1], z_interpolated_torch, create_graph=True, retain_graph=True,
                                                    grad_outputs=torch.ones_like(fake_out_on_z_interpolated[:, 1]))[0]

                            dG_dz_interpolated = torch.sqrt(torch.sum(torch.pow(dx_0_dz, 2) + torch.pow(dx_1_dz, 2), dim=1)).cpu().detach().numpy()
                            dG_dz_interpolated_list.append(dG_dz_interpolated)

                            grad_norm_plot, = self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].plot(np.linspace(0, 1, self.n_interpolate), dG_dz_interpolated)
                            imgs.append(grad_norm_plot)

                        dG_dz_interpolated_mat = np.array(dG_dz_interpolated_list)
                        dG_dz_interpolated_mean = np.mean(dG_dz_interpolated_mat, axis=0)
                        dG_dz_interpolated_std = np.std(dG_dz_interpolated_mat, axis=0)

                    self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_xlim(0, 1)
                    self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_ylim(0, self.grad_norm_amplitude)
                    self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_xticklabels([])
                    self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_xlabel(f"From {selected_mode_tuple[0] + 1} to {selected_mode_tuple[1] + 1}", fontdict={"size": 40})
                    if ss < 2:
                        self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_yticklabels([])
                    else:
                        self.ax_dict[f"gamma{gamma}_seed{seed}_G_grad_norm_{ss}"].set_ylabel(r"$||dG/dz||^2$", fontdict={"size": 40})

                """ Output KDE """
                kernel = stats.gaussian_kde(idd['x_out'].T)
                xx_x, yy_x = np.mgrid[self.output_range[0]:self.output_range[1]:50j, self.output_range[0]:self.output_range[1]:50j]
                positions_z = np.vstack([xx_x.ravel(), yy_x.ravel()])
                G_z_density_surf = np.reshape(kernel(positions_z).T, xx_x.shape)

                """ --- Contour for G density @ output"""
                levels = np.linspace(0, self.x_out_density_amplitude, 11)
                x_out_ct = self.ax_dict[f"gamma{gamma}_seed{seed}_output"].contourf(xx_x, yy_x, G_z_density_surf, cmap=deepviz.cmap, alpha=0.8, levels=levels)
                imgs.extend(x_out_ct.collections)
                divider = make_axes_locatable(self.ax_dict[f"gamma{gamma}_seed{seed}_output"])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(x_out_ct, cax=cax)

                """ --- label for each mode """
                for c, center in enumerate(self.dataset.centers):
                    self.ax_dict[f"gamma{gamma}_seed{seed}_output"].text(center[0], center[1], f"{c + 1}", {"ha": "center", "va": "center"}, fontsize=30)

                """ BP loc """
                BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"],
                                                                                          state_dict_G["hidden_layer.bias"],
                                                                                          state_dict_G["output_layer.weight"])
                BP_loc_G = BP_directions_G * BP_signed_distances_G[:, np.newaxis]
                D_output_grid = idd['D_output_grid']
                color_list = [Get_diverging_color(weight, color_scale=5) for weight in BP_delta_slopes_G[0, :]]
                BP_loc_0_scatter = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_0"].scatter(BP_loc_G[:, 0], BP_loc_G[:, 1], c=color_list)
                imgs.append(BP_loc_0_scatter)

                color_list = [Get_diverging_color(weight, color_scale=5) for weight in BP_delta_slopes_G[1, :]]
                BP_loc_1_scatter = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_1"].scatter(BP_loc_G[:, 0], BP_loc_G[:, 1], c=color_list)
                imgs.append(BP_loc_1_scatter)


                """ --- BP lines """
                G_hidden_layer_weights_np = idd['state_dict_G']["hidden_layer.weight"].cpu().numpy()
                G_hidden_layer_biases_np = idd['state_dict_G']["hidden_layer.bias"].cpu().numpy()
                for i, (w, b) in enumerate(zip(G_hidden_layer_weights_np, G_hidden_layer_biases_np)):
                    BP_line_points = Get_2D_line_points(w, b, plot_lim=np.max(np.abs(self.axis_range)))
                    BP_line_plot_0, = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_0"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1,
                                                                c=Get_diverging_color(BP_delta_slopes_G[0, i], color_scale=5), alpha=0.7)
                    imgs.append(BP_line_plot_0)

                    BP_line_plot_1, = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_loc_1"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1,
                                                                c=Get_diverging_color(BP_delta_slopes_G[1, i], color_scale=5), alpha=0.7)
                    imgs.append(BP_line_plot_1)

                density_range = [-self.density_amplitude, self.density_amplitude]
                levels = np.linspace(density_range[0], density_range[1], 21)

                """ BP density - dim 0 """
                weight_list_pos = np.maximum(BP_delta_slopes_G[0, :], 0)
                weight_list_neg = np.minimum(BP_delta_slopes_G[0, :], 0)

                xx, yy, density_KDE = KDE(self.x_range, self.y_range, BP_loc_G, bw_method=self.bandwidth)
                xx, yy, density_KDE_pos = KDE(self.x_range, self.y_range, BP_loc_G, weight_list_pos, bw_method=self.bandwidth)
                xx, yy, density_KDE_neg = KDE(self.x_range, self.y_range, BP_loc_G, weight_list_neg, bw_method=self.bandwidth)

                ct = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_0"].contourf(xx, yy, density_KDE_pos - density_KDE_neg,
                                                                         cmap='seismic', levels=levels)
                imgs.extend(ct.collections)
                divider = make_axes_locatable(self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_0"])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(ct, cax=cax)

                """ BP density - dim 1 """
                weight_list_pos = np.maximum(BP_delta_slopes_G[1, :], 0)
                weight_list_neg = np.minimum(BP_delta_slopes_G[1, :], 0)

                xx, yy, density_KDE = KDE(self.x_range, self.y_range, BP_loc_G, bw_method=self.bandwidth)
                xx, yy, density_KDE_pos = KDE(self.x_range, self.y_range, BP_loc_G, weight_list_pos, bw_method=self.bandwidth)
                xx, yy, density_KDE_neg = KDE(self.x_range, self.y_range, BP_loc_G, weight_list_neg, bw_method=self.bandwidth)

                ct = self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_1"].contourf(xx, yy, density_KDE_pos - density_KDE_neg,
                                                                         cmap='seismic', levels=levels)
                imgs.extend(ct.collections)
                divider = make_axes_locatable(self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_1"])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.fig.colorbar(ct, cax=cax)

                self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_title(f"gamma = {gamma}", fontdict={"size": 40})

                if j == 0:
                    # self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_ylabel(f"iteration = {t * args.plot_iter}", fontdict={"size": 40})
                    pass
                else:
                    self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_yticklabels([])
                    self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_0"].set_yticklabels([])
                    self.ax_dict[f"gamma{gamma}_seed{seed}_BP_density_1"].set_yticklabels([])


                self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_xlim(self.output_range)
                self.ax_dict[f"gamma{gamma}_seed{seed}_output"].set_ylim(self.output_range)


                if (not self.legend_drawn) and t != 1:
                    # self.ax_dict[f"gamma{gamma}_seed{seed}_text"].legend(loc="upper right")

                    handles_list = []
                    labels_list = []
                    #

                    # self.ax_dict[title].legend(handles_list, labels_list, loc="upper right")
                    # self.ax_dict[f"BP_G_density_{title}"]for ax_name in self.ax_dict:
                    #                         handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                    #                         handles_list.extend(handles)
                    #                         labels_list.extend(labels).legend(loc="upper right")
                    # self.ax_dict[f"BP_D_density_{title}"].legend(loc="upper right")
                    #
                    # handles_list = []
                    # labels_list = []
                    # for ax_name in [f"KL_{title}", f"BP_entropy_{title}"]:
                    #     handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                    #     handles_list.extend(handles)
                    #     labels_list.extend(labels)
                    # self.ax_dict[f"KL_{title}"].legend(handles_list, labels_list, loc="upper right")

        plt.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)
        self.fig.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)

        if loading:
            self.ims.append(tuple(imgs))
            # print(f"len(self.ims) {len(self.ims)}")
            if (not self.legend_drawn) and t > 0:
                self.legend_drawn = True


        frame_folder = "Viz"
        name = self.attr["name"]
        save_folder = os.path.join(summary_dir, name, frame_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.fig.savefig(os.path.join(save_folder, f"{name}_{str(t).zfill(3)}.png"), dpi=200)
        plt.close(fig=self.fig)


    def Generate_summary_video_from_file(self, my_part=None, num_parts=None, iter_start=0, iter_end=np.inf, skip_frame=1):
        generating_start_time = time.time()

        self.Init_figure()
        self.num_parts = num_parts
        self.skip_frame = skip_frame

        base_start_pos = start_pos = 0
        base_end_pos = end_pos = self.attr["max_t"]
        if my_part is not None and num_parts is not None:
            start_pos, end_pos = Get_start_and_end_pos_for_worker(my_part, num_parts, base_start_pos, base_end_pos)
            self.attr["name"] += f"_{my_part}-{num_parts}"

        print(f"part {my_part} / {num_parts}: ({start_pos}, {end_pos})")

        for t in range(self.attr["max_t"]):
            """ If this part of video is not started from the beginning, plot the previous segments line plots """
            if t % skip_frame != 0:
                continue

            if t >= start_pos and t < end_pos:
                if t % (self.attr["max_t"] // 5) == 0:
                    print(f't {t}, max_t {self.attr["max_t"]}')

                self.Plot_step(t, loading=True)
                self.total_frame += 1

            else:
                self.Plot_step(t, loading=False)

        print(f"Video production time: {time.time() - generating_start_time}")
        plt.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)
        self.fig.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)


    def Save_plot(self, title=None, fps=10, figures_folder=figures_folder):
        save_plot_start_time = time.time()

        if title is None:
            self.attr["name"] += f"_{Now()}"
            title = self.attr["name"]
        else:
            title += f"_{Now()}"

        self.fig.suptitle(title)
        self.attr['name'] = title
        # subplots_adjust(top=.95)
        plt.subplots_adjust(top=0.9, hspace=0.1, wspace=0.1)
        self.fig.subplots_adjust(top=0.9, hspace=0.01, wspace=0.01)
        tight_layout()

        mywriter = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me', title=title), bitrate=1000)
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=2, blit=False, repeat_delay=1000)


        if not os.path.exists(figures_folder):
            os.makedirs(figures_folder)


        if platform.system() == "Darwin":
            print("Using MacOS.")
        elif platform.system() == "Linux":
            print("Using Linux.")
        else:
            print("Using Windows.")


        output_filename = os.path.join(figures_folder, title + ".mp4")
        print(f"output_filename: {output_filename}")

        def progress_callback_func(i, n):
            prog_num_5 = np.round(self.total_frame / 5, 0).astype(int)
            if prog_num_5 == 0:
                prog_num_5 += 1
            if i % prog_num_5 == 0 or i == self.total_frame - 1:
                print(f'Saving frame {i + 1} of {self.total_frame}')

        if self.total_frame > 0:
            ani.save(output_filename, writer=mywriter, progress_callback=progress_callback_func, dpi=50) # , dpi=50
        print(f"video saving time: {time.time() - save_plot_start_time}")



if __name__ == "__main__":
    timer = Timer()
    task_dir = "Grid5-width128-gamma0.8-0.9-0.99-0.999-0.9999-1"

    dataset_name = "random9-6_1"
    dataset_name = "grid5"
    deepVisuals_Summary = DeepVisuals_Summary_weighted_BP_KDE_and_grad_norm(task_dir, dataset_name, cumul=True)


    print(deepVisuals_Summary.gamma_seed_deepviz_dict)
    # print(deepVisuals_Summary.deepVisuals_classified_name_dict)
    # print(len(deepVisuals_Summary.deepVisuals_classified_name_dict))
    # print(deepVisuals_Summary.deepVisuals_matching_name_dict)
    # print(len(deepVisuals_Summary.deepVisuals_matching_name_dict))
    deepVisuals_Summary.Init_figure()
    # max_t = deepVisuals_Summary.attr["max_t"]
    max_t = 100
    # max_t = 3
    for t in range(max_t):
        print(f"Plotting frame {t + 1} / {max_t}")
        deepVisuals_Summary.Init_figure()
        deepVisuals_Summary.Plot_step(t, loading=True)
        deepVisuals_Summary.total_frame += 1
    #
    # deepVisuals_Summary.Save_plot(figures_folder=os.path.join(summary_dir, task_dir), fps=5)
    timer.Print()
