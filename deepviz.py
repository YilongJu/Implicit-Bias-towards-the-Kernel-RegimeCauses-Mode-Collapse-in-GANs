from matplotlib import use
# use("Qt5Agg")
# use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D                         # Not explicitly used, but necessary
from matplotlib.transforms import Affine2D                      # Not explicitly used, but necessary
import mpl_toolkits.axisartist.floating_axes as floating_axes   # Not explicitly used, but necessary

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
from matplotlib import animation
from matplotlib import cm
import matplotlib.colors as mc

from scipy.spatial.transform import Rotation

import pylab as pl
import pickle

import os
import platform
import time
import datetime

from BPs import *
from ComputationalTools import *
from utils import Get_models

os.environ['KMP_DUPLICATE_LIB_OK']='True'
plt.rcParams['savefig.facecolor'] = "0.8"

if platform.system() == "Darwin":
    print("Using MacOS.")
    plt.rcParams['animation.ffmpeg_path'] = "/usr/local/bin/ffmpeg"
elif platform.system() == "Linux":
    print("Using Linux.")
    plt.rcParams['animation.ffmpeg_path'] = "/usr/bin/ffmpeg"
else:
    print("Using Windows.")
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/juyil/ffmpeg/bin/ffmpeg.exe'

data_folder = "Data"
figures_folder = "Figures"

def Torch_loss_list_val_list(loss_list):
    if isinstance(loss_list[0], float):
        return loss_list
    else:
        return [ele.item() for ele in loss_list]

""" Calculate gaussian kde estimate for a dataset """
def kde(mu, tau, bbox=[-5, 5, -5, 5], save_file="", xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox) # set axis range by [xmin, xmax, ymin, ymax]
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2])) # set axis value ratio manually to get equal length
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)

""" Create a python generator for a pickle file """
def Load_all_pickles(title, data_folder=data_folder):
    # print("Load_all_pickles title", title)

    if platform.system() == "Darwin":
        print("Using MacOS.")
    elif platform.system() == "Linux":
        print("Using Linux.")
    else:
        print("Using Windows.")

    filepath = os.path.join(os.getcwd(), data_folder, title + ".pickle")
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except:
                    print(f"End of file: {title}")
                    break
    else:
        raise ValueError("File not found")

""" Record positio of panels of viz """
fig_size = 6
grid_span = 6
span_figure_r = 3
span_figure_c = 6


#%%
class DeepVisuals_2D():
    def __init__(self, args=None, z_mesh=None, x_real=None, xx_D=None, yy_D=None, xx_z=None, yy_z=None, bbox_x=[-5, 5, -5, 5], bbox_z=[-3, 3, -3, 3], name="", z_test=None, attr_seq_name_list=None, handle=None, data_folder=data_folder, dataset=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        if attr_seq_name_list is None:
            attr_seq_name_list = ["iter", "loss_G", "loss_D", "loss_G_tot", "loss_D_tot", "grad_raw_norm_x", "grad_raw_norm_y", "grad_corr_norm_x", "grad_corr_norm_y",
         "update_tot_norm_x", "update_tot_norm_y", "wall_time", "phase_factor", "conditioning_factor"]

        """ Time-invariant members """
        self.attr = {}
        self.attr['z_mesh'] = z_mesh
        self.attr['x_real'] = x_real
        self.attr['xx_D'] = xx_D
        self.attr['yy_D'] = yy_D
        self.attr['xx_z'] = xx_z
        self.attr['yy_z'] = yy_z
        self.attr['bbox_x'] = bbox_x
        self.attr['bbox_z'] = bbox_z
        self.attr['timestamp'] = Now()
        self.attr['name'] = name
        self.attr['z_test'] = z_test
        self.attr['dataset'] = dataset
        self.attr['device'] = device
        self.attr['args'] = args

        """ Time-variant members """
        self.attr_seq = {}
        for item in attr_seq_name_list:
            self.attr_seq[item] = []

        """ Utils members """
        self.data_container_dict = {}
        self.last_contour_plot = None
        self.legend_drawn = False
        self.cmap = 'Blues'
        self.cmap_D = 'Reds'
        self.first_frame = True
        self.num_parts = 1
        self.skip_frame = 1
        self.total_frame = 0
        self.data_folder = data_folder

        self.image_min = None  # min pixel intensity for normalizing images
        self.image_max = None  # max pixel intensity for normalizing images

        self.handle = handle

        if self.attr["args"] is not None:
            if not hasattr(self.attr["args"], "save_path"):
                print("self.attr['args']", self.attr['args'])
                self.attr['args'].save_path == ""

            if not os.path.exists(self.attr['args'].save_path):
                os.makedirs(self.attr['args'].save_path)

            self.save_file_path = os.path.join(self.attr['args'].save_path, f"{self.attr['name']}_{self.attr['timestamp']}.pickle")

        if name != "" and self.handle is None:
            self.handle = open(self.save_file_path, "wb")

            """ Adding items into data_container """
            self.data_container_dict["attr"] = self.attr
            self.data_container_dict["attr_seq"] = self.attr_seq
            pickle.dump(self.data_container_dict, self.handle)

        self.Calculate_max_t()


    def Calculate_max_t(self):
        if self.attr["args"] is not None:
            self.attr["max_iter_div_5"] = self.attr["args"].iteration // 5
            if self.attr["max_iter_div_5"] == 0:
                self.attr["max_iter_div_5"] += 1

            self.attr["max_t"] = self.attr["args"].iteration // self.attr["args"].plot_iter + 1

    def Init_figure(self):
        self.Calculate_max_t()

        self.ims = []
        self.ax_dict = {}
        self.figure_nrow = span_figure_r
        self.figure_ncol = span_figure_c

        self.fig = pl.figure(figsize=(self.figure_ncol * fig_size, self.figure_nrow * fig_size))
        """ Factors line plot """
        """     conditioning factor """
        self.ax_dict["conditioning_factor"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span + 4, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        self.ax_dict["conditioning_factor"].set_xlabel(r"Iteration")
        self.ax_dict["conditioning_factor"].set_ylabel(r"Value")

        """     Phase factor """
        self.ax_dict["phase_factor"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span + 2, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        # self.ax_dict["phase_factor"].set_xlabel(r"Value")
        self.ax_dict["phase_factor"].set_ylabel(r"Count")

        # self.ax_dict["traj_angle"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        # self.ax_dict["traj_angle"].set_xlabel("Iteration")
        # self.ax_dict["traj_angle"].set_ylabel("Trajectory Angle")
        # self.ax_dict["ref_angle"] = self.ax_dict["traj_angle"].twinx()
        # self.ax_dict["ref_angle"].set_ylabel("Reference Angle")
        # self.ax_dict["l2_dist_GD"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span + 4, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        # self.ax_dict["l2_dist_GD"].set_xlabel("Wall time (s)")
        # self.ax_dict["l2_dist_GD"].set_ylabel("L2 Distance to Params")
        # self.ax_dict["grad_angle_GD"] = self.ax_dict["l2_dist_GD"].twinx()
        # self.ax_dict["grad_angle_GD"].set_ylabel("Grad Angle")

        """ Minimax Criterion """
        self.ax_dict["eig_vals_Hyy_g"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        # self.ax_dict["eig_vals_Hyy_g"].set_xlabel(r"Value")
        self.ax_dict["eig_vals_Hyy_g"].set_ylabel(r"Count")
        # self.ax_dict["eig_vals_Hyy_g"].set_title(r"Histogram of $\lambda$(H_{DD}) or $\lambda$(H_{yy})")
        self.ax_dict["minimax_eig_2"] = self.ax_dict["eig_vals_Hyy_g"].twinx()

        """ Maximin Criterion """
        self.ax_dict["eig_vals_Hxx_f"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span + 4, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        # self.ax_dict["eig_vals_Hxx_f"].set_xlabel(r"Value")
        self.ax_dict["eig_vals_Hxx_f"].set_ylabel(r"Count")
        # self.ax_dict["eig_vals_Hxx_f"].set_title(r"Histogram of $\lambda$(H_{GG}) or $\lambda$(H_{xx})")
        self.ax_dict["minimax_eig_1"] = self.ax_dict["eig_vals_Hxx_f"].twinx()

        """ Grad norm curve """
        self.ax_dict["grad_norm"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span + 2, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        self.ax_dict["grad_norm"].set_xlabel(r"Iteration")
        self.ax_dict["grad_norm"].set_ylabel(r"$||\nabla_x f||_2$, $||\nabla_y g||_2$")

        """ Grad norm curve """
        self.ax_dict["grad_corr_norm"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span + 2, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        self.ax_dict["grad_corr_norm"].set_xlabel("Wall time (s)")
        self.ax_dict["grad_corr_norm"].set_ylabel(r"Grad Correction Norm")
        # self.ax_dict["l2_dist"] = self.ax_dict["grad_corr_norm"].twinx()
        # self.ax_dict["l2_dist"].set_ylabel(r"$||\theta_T - \theta_t||_2$")

        """ Learning curve """
        self.ax_dict["loss_G"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span, 0 * grid_span), rowspan=2, colspan=2 * grid_span)
        self.ax_dict["loss_G"].set_xlabel(r"Iteration")
        self.ax_dict["loss_G"].set_ylabel(r"Loss_G")
        self.ax_dict["loss_D"] = self.ax_dict["loss_G"].twinx()
        self.ax_dict["loss_D"].set_ylabel(r"Loss_D")

        if self.attr['args'].divergence == "standard":
            self.opt_loss_G_ref_val = np.log(2)
        else:
            self.opt_loss_G_ref_val = -np.log(2)

        """ Eigenvalue histogram """
        # self.ax_dict["eig_mod"] = plt.subplot2grid((self.figure_nrow * grid_size, self.figure_ncol * grid_size), (0 * grid_size + 3, 2 * grid_size), rowspan=3, colspan=1 * grid_size)
        # # self.ax_dict["eig_mod"].set_xlabel(r"Value")
        # self.ax_dict["eig_mod"].set_ylabel(r"Count")
        # # self.ax_dict["eig_mod"].set_title(r"Histogram of $\lambda$")
        # self.ax_dict["eig_real"] = plt.subplot2grid((self.figure_nrow * grid_size, self.figure_ncol * grid_size), (0 * grid_size + 3, 3 * grid_size), rowspan=3, colspan=1 * grid_size)
        # # self.ax_dict["eig_real"].set_xlabel(r"Value")
        # self.ax_dict["eig_real"].set_ylabel(r"Count")
        # # self.ax_dict["eig_real"].set_title(r"Histogram of $\lambda$")
        # self.ax_dict["eig_imag"] = plt.subplot2grid((self.figure_nrow * grid_size, self.figure_ncol * grid_size), (0 * grid_size + 3, 4 * grid_size), rowspan=3, colspan=1 * grid_size)
        # # self.ax_dict["eig_imag"].set_xlabel(r"Value")
        # self.ax_dict["eig_imag"].set_ylabel(r"Count")
        # # self.ax_dict["eig_imag"].set_title(r"Histogram of $\lambda$")

        """ Eigenvalue scatter """
        # self.ax_dict["eig"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span, 5 * grid_span), rowspan=1 * grid_span,
        #                                        colspan=1 * grid_span)
        # self.ax_dict["eig"].set_xlabel(r"$\Re(\lambda)$")
        # self.ax_dict["eig"].set_ylabel(r"$\Im(\lambda)$")
        # self.ax_dict["eig"].add_artist(Circle((0, 0), 1, color="#00FF00", fill=False))
        # self.ax_dict["eig"].set_aspect("equal")

        if self.attr['args'].data in ["mnist", "cifar"]:
            self.show_num = 32
            col_num = 8

            for i in range(self.show_num):
                row_idx = i // col_num
                col_idx = i % col_num
                self.ax_dict[f"out_{i}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span + 3 * row_idx, 2 * grid_span + 3 * col_idx), rowspan=3, colspan=3)
                self.ax_dict[f"out_{i}"].set_xlabel(None)
                self.ax_dict[f"out_{i}"].set_ylabel(None)
                self.ax_dict[f"out_{i}"].set_xticklabels([])
                self.ax_dict[f"out_{i}"].set_yticklabels([])
                self.ax_dict[f"out_{i}"].xaxis.set_visible(False)
                self.ax_dict[f"out_{i}"].yaxis.set_visible(False)


        else:
            """ Input plot """
            self.ax_dict["in"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span, 4 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            self.ax_dict["in"].axis(self.attr["bbox_z"])  # set axis range by [xmin, xmax, ymin, ymax]
            self.ax_dict["in"].set_aspect(abs(self.attr["bbox_z"][1] - self.attr["bbox_z"][0]) / abs(
                self.attr["bbox_z"][3] - self.attr["bbox_z"][2]))  # set axis value ratio manually to get equal length
            self.ax_dict["in"].set_xlabel(r"$z_1$")
            self.ax_dict["in"].set_ylabel(r"$z_2$")
            # plasma, winter, RdPu
            self.z_color_map = "plasma"
            self.z_color_map_val = np.abs(self.attr["z_mesh"][:, 0]) + np.abs(self.attr["z_mesh"][:, 1])
            self.z_color_map_val = np.angle(self.attr["z_mesh"][:, 0] + 1j * self.attr["z_mesh"][:, 1])  # np.abs(self.attr["z_mesh"][:, 0]) + np.abs(self.attr["z_mesh"][:, 1])
            self.ax_dict["in"].scatter(self.attr["z_mesh"][:, 0], self.attr["z_mesh"][:, 1], linewidth=4.0, alpha=0.8, cmap=self.z_color_map, c=self.z_color_map_val)
            self.ax_dict["in"].scatter(self.attr["z_test"][:, 0], self.attr["z_test"][:, 1], linewidth=1.0, alpha=0.7, c="#00FFFF")

            """ U plot """
            self.ax_dict["U"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span, 5 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span, projection='3d')
            self.ax_dict["U"].set_xlabel(r"$z_1$")
            self.ax_dict["U"].set_ylabel(r"$z_2$")
            self.ax_dict["U"].set_zlabel(r"$U(z)$")
            self.ax_dict["U"].set_xlim(self.attr["bbox_z"][0], self.attr["bbox_z"][1])
            self.ax_dict["U"].set_ylim(self.attr["bbox_z"][2], self.attr["bbox_z"][3])
            self.ax_dict["U"].set_zlim(-50, 50)

            """ Gz1 plot """
            self.ax_dict["G_z_1"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span, 3 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span, projection='3d')
            self.ax_dict["G_z_1"].set_xlabel(r"$x_1$, $G(z)_1$")
            self.ax_dict["G_z_1"].set_ylabel(r"$z_1$")
            self.ax_dict["G_z_1"].set_zlabel(r"$z_2$")
            self.ax_dict["G_z_1"].set_xlim(self.attr["bbox_x"][0], self.attr["bbox_x"][1])
            self.ax_dict["G_z_1"].set_ylim(self.attr["bbox_z"][0], self.attr["bbox_z"][1])
            self.ax_dict["G_z_1"].set_zlim(self.attr["bbox_z"][2], self.attr["bbox_z"][3])

            """ Gz2 plot """
            self.ax_dict["G_z_2"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span, 4 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span, projection='3d')
            self.ax_dict["G_z_2"].set_xlabel(r"$z_1$")
            self.ax_dict["G_z_2"].set_ylabel(r"$z_2$")
            self.ax_dict["G_z_2"].set_zlabel(r"$x_2$, $G(z)_2$")
            self.ax_dict["G_z_2"].set_xlim(self.attr["bbox_z"][0], self.attr["bbox_z"][1])
            self.ax_dict["G_z_2"].set_ylim(self.attr["bbox_z"][2], self.attr["bbox_z"][3])
            self.ax_dict["G_z_2"].set_zlim(self.attr["bbox_x"][2], self.attr["bbox_x"][3])

            """ Output plot with D contour"""
            self.ax_dict["out"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span, 3 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            self.ax_dict["out"].scatter(self.attr["x_real"][:, 0], self.attr["x_real"][:, 1], color="#000000", linewidth=2.0, alpha=1)
            self.ax_dict["out"].axis(self.attr["bbox_x"])  # set axis range by [xmin, xmax, ymin, ymax]
            self.ax_dict["out"].set_aspect(abs(self.attr["bbox_x"][1] - self.attr["bbox_x"][0]) / abs(
                self.attr["bbox_x"][3] - self.attr["bbox_x"][2]))  # set axis value ratio manually to get equal length
            self.ax_dict["out"].set_xlabel(r"$x_1$, $G(z)_1$")
            self.ax_dict["out"].set_ylabel(r"$x_2$, $G(z)_2$")

            """ Output plot with kde estimate"""
            self.ax_dict["data"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span, 2 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            # self.ax_dict["data"].scatter(self.attr["x_real"][:, 0], self.attr["x_real"][:, 1], color="#000000", linewidth=2.0, alpha=0.7)
            self.ax_dict["data"].axis(self.attr["bbox_x"])  # set axis range by [xmin, xmax, ymin, ymax]
            self.ax_dict["data"].set_aspect(abs(self.attr["bbox_x"][1] - self.attr["bbox_x"][0]) / abs(self.attr["bbox_x"][3] - self.attr["bbox_x"][2]))  # set axis value ratio manually to get equal length
            self.ax_dict["data"].set_xlabel(r"$x_1$, $G(z)_1$")
            self.ax_dict["data"].set_ylabel(r"$x_2$, $G(z)_2$")

            """ BP plots """
            """     Generator """
            self.ax_dict["delta_slope_G_z_2"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span + 4, 5 * grid_span), rowspan=2, colspan=1 * grid_span)
            self.ax_dict["delta_slope_G_z_1"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span + 2, 5 * grid_span), rowspan=2, colspan=1 * grid_span)
            self.ax_dict["signed_distance_G"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span, 5 * grid_span), rowspan=2, colspan=1 * grid_span)

            self.ax_dict["BP_G_z_1"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (2 * grid_span, 2 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            self.ax_dict["BP_G_z_1"].set_xlabel(r"$z_1$")
            self.ax_dict["BP_G_z_1"].set_ylabel(r"$z_2$")
            self.ax_dict["BP_G_z_1"].axis(self.attr["bbox_z"])
            self.ax_dict["BP_G_z_1"].set_aspect(abs(self.attr["bbox_z"][1] - self.attr["bbox_z"][0]) / abs(self.attr["bbox_z"][3] - self.attr["bbox_z"][2]))

            self.ax_dict["BP_G_z_2"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (1 * grid_span, 5 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            self.ax_dict["BP_G_z_2"].set_xlabel(r"$z_1$")
            self.ax_dict["BP_G_z_2"].set_ylabel(r"$z_2$")
            self.ax_dict["BP_G_z_2"].axis(self.attr["bbox_z"])
            self.ax_dict["BP_G_z_2"].set_aspect(abs(self.attr["bbox_z"][1] - self.attr["bbox_z"][0]) / abs(self.attr["bbox_z"][3] - self.attr["bbox_z"][2]))

            """     Discriminator """
            self.ax_dict["delta_slope_D"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span + 3, 4 * grid_span), rowspan=3, colspan=1 * grid_span)
            self.ax_dict["signed_distance_D"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span, 4 * grid_span), rowspan=3, colspan=1 * grid_span)

            self.ax_dict["BP_D_x"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (0 * grid_span, 3 * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
            self.ax_dict["BP_D_x"].axis(self.attr["bbox_x"])
            self.ax_dict["BP_D_x"].set_xlabel(r"$x_1$")
            self.ax_dict["BP_D_x"].set_ylabel(r"$x_2$")
            self.ax_dict["BP_D_x"].set_aspect(abs(self.attr["bbox_x"][1] - self.attr["bbox_x"][0]) / abs(self.attr["bbox_x"][3] - self.attr["bbox_x"][2]))

        self.fig.set_tight_layout(True)
        print("Figure intialized.")


    def Plot_step(self, idd, loading=False):
        toc = time.time()

        if not loading:
            """ Adding items into data_container """
            if self.handle is not None:
                del idd["G"]
                del idd["D"]

                pickle.dump(idd, self.handle)

            if idd["iter"] % self.attr["max_iter_div_5"] == 0:
                plot_time = time.time() - toc
                print(f"> Mid [iter {idd['iter']} / {self.attr['args'].iteration}], plot time taken: {plot_time:.3f}")
            return

        for item in self.attr_seq:
            val = idd.get(item, None)
            self.attr_seq[item].append(val)

        line_animated = True

        imgs = []

        """ ====================== Optional Viz ====================== """
        if self.attr['args'].data in ["mnist", "cifar"]:

            for i in range(self.show_num):
                if self.attr['args'].data in ["mnist"]:
                    image_normed = (idd['x_out'][i, 0, ...] - self.image_min) / (self.image_max - self.image_min)
                    image_show_i = self.ax_dict[f"out_{i}"].imshow(image_normed, cmap=cm.get_cmap("gray"))
                elif self.attr['args'].data in ["cifar"]:
                    # print(np.transpose(idd['x_out'][i, ...], (1, 2, 0)))
                    image_show_i = self.ax_dict[f"out_{i}"].imshow(np.transpose(idd['x_out'][i, ...], (1, 2, 0)))
                else:
                    raise NotImplementedError

                imgs.append(image_show_i)
        else:
            try:
                """ Compute density for G(z) """
                kernel = stats.gaussian_kde(idd['x_out'].T)
                xx_x, yy_x = np.mgrid[self.attr["bbox_x"][0]:self.attr["bbox_x"][1]:50j, self.attr["bbox_x"][2]:self.attr["bbox_x"][3]:50j]
                positions_z = np.vstack([xx_x.ravel(), yy_x.ravel()])
                G_z_density_surf = np.reshape(kernel(positions_z).T, xx_x.shape)

                """ Contour for G density @ output"""
                cfset = self.ax_dict["data"].contourf(xx_x, yy_x, G_z_density_surf, cmap=self.cmap, alpha=0.8)
                imgs.extend(cfset.collections)
            except:
                print("KDE error")

            """ Contour for D @ output """
            D_prob_grid = 1 / (1 + np.exp(-idd['D_output_grid']))
            cfset_D = self.ax_dict["out"].contourf(self.attr['xx_D'], self.attr['yy_D'], D_prob_grid, alpha=0.3, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], colors=["#110000", "#440000", "#770000", "#AA0000", "#DD0000", "#00FF00", "#00DD00", "#00AA00", "#007700", "#004400", "#001100"])
            imgs.extend(cfset_D.collections)

            """ z_test scatter """
            if idd.get('x_fake_mesh_vec_out', None) is not None:
                """ G(z) mesh """
                G_z_mesh_scatter = self.ax_dict["out"].scatter(idd['x_fake_mesh_vec_out'][:, 0], idd['x_fake_mesh_vec_out'][:, 1], linewidth=3.0, alpha=0.3, cmap=self.z_color_map, c=self.z_color_map_val)
                imgs.append(G_z_mesh_scatter)

            """ Scatter for G output """
            x_out_scatter = self.ax_dict["data"].scatter(idd['x_out'][:, 0], idd['x_out'][:, 1], linewidth=1.5, alpha=0.5, c="#00FFFF")
            imgs.append(x_out_scatter)

            # G_z_1_view_elev = -60
            # G_z_1_view_azim = 75
            G_z_1_view_elev = None # View angle for 3D plots
            G_z_1_view_azim = None

            is_brenier = False
            if hasattr(self.attr["args"], "brenier"):
                if self.attr["args"].brenier:
                    is_brenier = True

            if idd.get('state_dict_G', None) is not None and is_brenier:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                G, D = Get_models(self.attr["args"], None)
                G = G.to(device)
                G.load_state_dict(idd["state_dict_G"])

                U_mesh_vec_out = G(torch.from_numpy(self.attr["z_mesh"]).to(device).float()).cpu().detach().numpy()
                """ G(z)_2 scatter """
                U_scatter = self.ax_dict["U"].scatter(self.attr["z_mesh"][:, 0], self.attr["z_mesh"][:, 1], U_mesh_vec_out, linewidth=4.0, alpha=0.8, cmap=self.z_color_map, c=self.z_color_map_val)
                imgs.append(U_scatter)

                stride = 5
                U_grid = np.reshape(U_mesh_vec_out.T, self.attr['xx_z'].shape)
                U_wireframe = self.ax_dict["U"].plot_wireframe(self.attr['xx_z'], self.attr['yy_z'], U_grid, rstride=stride, cstride=stride)
                imgs.append(U_wireframe)


            if idd.get('x_fake_mesh_vec_out', None) is not None:
                # print("x_fake_mesh_vec_out")
                rot_z_1 = Rotation.from_rotvec([np.pi / 2, 0, 0])
                rot_z_2 = Rotation.from_rotvec([0, np.pi / 2, 0])
                point_cloud = np.vstack([self.attr["z_mesh"][:, 0], self.attr["z_mesh"][:, 1], idd['x_fake_mesh_vec_out'][:, 0]]).T
                point_cloud_rotated = rot_z_1.apply(rot_z_2.apply(point_cloud))

                """ G(z)_1 scatter """
                G_z_1_scatter = self.ax_dict["G_z_1"].scatter(point_cloud_rotated[:, 0], point_cloud_rotated[:, 1], point_cloud_rotated[:, 2], linewidth=4.0, alpha=0.8, cmap=self.z_color_map, c=self.z_color_map_val)
                self.ax_dict["G_z_1"].view_init(elev=G_z_1_view_elev, azim=G_z_1_view_azim)
                imgs.append(G_z_1_scatter)

                """ G(z)_2 scatter """
                G_z_2_scatter = self.ax_dict["G_z_2"].scatter(self.attr["z_mesh"][:, 0], self.attr["z_mesh"][:, 1], idd['x_fake_mesh_vec_out'][:, 1], linewidth=4.0, alpha=0.8, cmap=self.z_color_map, c=self.z_color_map_val)
                imgs.append(G_z_2_scatter)

                stride = 5
                """ G(z)_1 wireframe """
                xx_z_rotated = np.reshape(point_cloud_rotated[:, 0].T, self.attr['xx_z'].shape)
                yy_z_rotated = np.reshape(point_cloud_rotated[:, 1].T, self.attr['yy_z'].shape)
                x_fake_1_grid_rotated = np.reshape(point_cloud_rotated[:, 2].T, self.attr['xx_z'].shape)

                G_z_1_wireframe = self.ax_dict["G_z_1"].plot_wireframe(xx_z_rotated, yy_z_rotated, x_fake_1_grid_rotated, rstride=stride, cstride=stride)
                imgs.append(G_z_1_wireframe)

                """ G(z)_2 wireframe """
                G_z_2_wireframe = self.ax_dict["G_z_2"].plot_wireframe(self.attr['xx_z'], self.attr['yy_z'], idd['x_fake_2_grid'], rstride=stride, cstride=stride)
                imgs.append(G_z_2_wireframe)

                """ Calculate BP parameters """
                if self.attr['args'].arch != "mlp" or is_brenier:
                    is_pure_mlp = False
                else:
                    is_pure_mlp = True

                if idd.get('state_dict_G', None) is not None and is_brenier:
                    BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(idd['state_dict_G']["hidden_layer.weight"], idd['state_dict_G']["hidden_layer.bias"], idd['state_dict_G']["output_layer.weight"])
                    BP_delta_slopes_G = BP_delta_slopes_G.ravel()
                    _, _, delta_slope_G_z_1_hist = self.ax_dict["delta_slope_G_z_1"].hist(BP_delta_slopes_G[np.isfinite(BP_delta_slopes_G)], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\mu_G^{(1)}$", bins=30)
                    imgs.extend(delta_slope_G_z_1_hist)

                    _, _, signed_distance_G_hist = self.ax_dict["signed_distance_G"].hist(BP_signed_distances_G[np.isfinite(BP_signed_distances_G)], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\gamma_G$", bins=30)
                    imgs.extend(signed_distance_G_hist)

                    G_hidden_layer_weights_np = idd['state_dict_G']["hidden_layer.weight"].cpu().numpy()
                    G_hidden_layer_biases_np = idd['state_dict_G']["hidden_layer.bias"].cpu().numpy()

                    G_alt_act_num = 0
                    if self.attr['args'] is not None:
                        if hasattr(self.attr['args'], "alt_act_prop"):
                            if self.attr['args'].alt_act_prop is not None:
                                G_alt_act_num = np.floor(self.attr['args'].alt_act_prop * self.attr['args'].g_hidden).astype(int)
                        else:
                            setattr(self.attr['args'], "alt_act_prop", None)

                    for i, (w, b) in enumerate(zip(G_hidden_layer_weights_np, G_hidden_layer_biases_np)):
                        BP_line_points = Get_2D_line_points(w, b, plot_lim=self.attr['args'].plot_lim_z)
                        if i == G_alt_act_num - 1:
                            BP_label = "G alt BP"
                        elif i == G_alt_act_num:
                            BP_label = "G BP"
                        else:
                            BP_label = ""

                        BP_line_plot, = self.ax_dict["U"].plot(BP_line_points[:, 0], BP_line_points[:, 1], np.ones_like(BP_line_points[:, 0]) * (-50), '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                        BP_line_plot, = self.ax_dict["BP_G_z_2"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                if idd.get('state_dict_G', None) is not None and is_pure_mlp:
                    BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(idd['state_dict_G']["hidden_layer.weight"], idd['state_dict_G']["hidden_layer.bias"], idd['state_dict_G']["output_layer.weight"])
                    _, _, delta_slope_G_z_1_hist = self.ax_dict["delta_slope_G_z_1"].hist(BP_delta_slopes_G[0, :][np.isfinite(BP_delta_slopes_G[0, :])], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\mu_G^{(1)}$", bins=30)
                    imgs.extend(delta_slope_G_z_1_hist)

                    _, _, delta_slope_G_z_2_hist = self.ax_dict["delta_slope_G_z_2"].hist(BP_delta_slopes_G[1, :][np.isfinite(BP_delta_slopes_G[1, :])], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\mu_G^{(2)}$", bins=30)
                    imgs.extend(delta_slope_G_z_2_hist)

                    _, _, signed_distance_G_hist = self.ax_dict["signed_distance_G"].hist(BP_signed_distances_G[np.isfinite(BP_signed_distances_G)], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\gamma_G$", bins=30)
                    imgs.extend(signed_distance_G_hist)

                    G_hidden_layer_weights_np = idd['state_dict_G']["hidden_layer.weight"].cpu().numpy()
                    G_hidden_layer_biases_np = idd['state_dict_G']["hidden_layer.bias"].cpu().numpy()

                    G_alt_act_num = 0
                    if self.attr['args'] is not None:
                        if hasattr(self.attr['args'], "alt_act_prop"):
                            if self.attr['args'].alt_act_prop is not None:
                                G_alt_act_num = np.floor(self.attr['args'].alt_act_prop * self.attr['args'].g_hidden).astype(int)
                        else:
                            setattr(self.attr['args'], "alt_act_prop", None)

                    for i, (w, b) in enumerate(zip(G_hidden_layer_weights_np, G_hidden_layer_biases_np)):
                        BP_line_points = Get_2D_line_points(w, b, plot_lim=self.attr['args'].plot_lim_z)
                        if i == G_alt_act_num - 1:
                            BP_label = "G alt BP"
                        elif i == G_alt_act_num:
                            BP_label = "G BP"
                        else:
                            BP_label = ""

                        BP_line_plot, = self.ax_dict["G_z_1"].plot(np.ones_like(BP_line_points[:, 0]) * self.attr["bbox_x"][0], BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[0, i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                        BP_line_plot, = self.ax_dict["G_z_2"].plot(BP_line_points[:, 0], BP_line_points[:, 1], np.ones_like(BP_line_points[:, 0]) * self.attr["bbox_x"][0], '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[1, i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                        BP_line_plot, = self.ax_dict["BP_G_z_1"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[0, i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                        BP_line_plot, = self.ax_dict["BP_G_z_2"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1, color="#00FF00" if i < G_alt_act_num else Get_diverging_color(BP_delta_slopes_G[1, i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)

                if idd.get('state_dict_D', None) is not None:
                    BP_directions_D, BP_signed_distances_D, BP_delta_slopes_D = Get_BP_params(idd['state_dict_D']["hidden_layer.weight"], idd['state_dict_D']["hidden_layer.bias"], idd['state_dict_D']["output_layer.weight"])
                    _, _, delta_slope_D_hist = self.ax_dict["delta_slope_D"].hist(BP_delta_slopes_D.ravel()[np.isfinite(BP_delta_slopes_D.ravel())], animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\mu_D$", bins=30)
                    imgs.extend(delta_slope_D_hist)

                    BP_signed_distances_D_nona = BP_signed_distances_D[np.isfinite(BP_signed_distances_D)]
                    _, _, signed_distance_D_hist = self.ax_dict["signed_distance_D"].hist(BP_signed_distances_D_nona, animated=True, density=False, color="#000000", alpha=0.7, log=False, label=r"$\gamma_D$", bins=30)
                    imgs.extend(signed_distance_D_hist)

                    D_hidden_layer_weights_np = idd['state_dict_D']["hidden_layer.weight"].cpu().numpy()
                    D_hidden_layer_biases_np = idd['state_dict_D']["hidden_layer.bias"].cpu().numpy()
                    for i, (w, b) in enumerate(zip(D_hidden_layer_weights_np, D_hidden_layer_biases_np)):
                        BP_line_points = Get_2D_line_points(w, b, plot_lim=self.attr['args'].plot_lim_x)
                        BP_label = r"D BPs" if i == 0 else None

                        BP_line_plot, = self.ax_dict["BP_D_x"].plot(BP_line_points[:, 0], BP_line_points[:, 1], '-', linewidth=1, color=Get_diverging_color(BP_delta_slopes_D[0, i]), animated=True, label=BP_label, alpha=0.7)
                        imgs.append(BP_line_plot)


        """ ====================== Generic Viz ====================== """
        """ Eigenvalues scatter """
        if (idd.get('eig_vals_Hxx_f', None) is not None) and (idd.get('eig_vals_Hyy_g', None) is not None):
            bin_num = "sqrt"
            Hxx_f_sym = r"$\lambda(H_{xx}f)$"
            Hyy_g_sym = r"$\lambda(H_{yy}g)$"
            if len(idd['eig_vals_Hxx_f']) <= 20:
                gg_eig_vals_bar = self.ax_dict["eig_vals_Hxx_f"].bar(x=list(range(1, 1 + len(idd['eig_vals_Hxx_f']))), height=np.sort(idd['eig_vals_Hxx_f'].real), color="#2222DD", alpha=0.7, label=Hxx_f_sym, log=False, width=0.6)
                imgs.extend(gg_eig_vals_bar)
                self.ax_dict["eig_vals_Hxx_f"].set_ylabel("Magnitude")
            else:
                _, _, gg_eig_vals_hist = self.ax_dict["eig_vals_Hxx_f"].hist(idd['eig_vals_Hxx_f'].real, animated=True, bins=bin_num, density=False, color="#2222DD", alpha=0.7, log=True, label=Hxx_f_sym)
                imgs.extend(gg_eig_vals_hist)

            gg_eig_vals_hist_text = self.ax_dict["eig_vals_Hxx_f"].text(0.75, 0.5, f"{Hxx_f_sym}\nmax: {np.max(idd['eig_vals_Hxx_f'].real):.4f}\nmin: {np.min(idd['eig_vals_Hxx_f'].real):.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["eig_vals_Hxx_f"].transAxes, fontsize=13)
            imgs.append(gg_eig_vals_hist_text)

            if len(idd['eig_vals_Hyy_g']) <= 20:
                dd_eig_vals_bar = self.ax_dict["eig_vals_Hyy_g"].bar(x=list(range(1, 1 + len(idd['eig_vals_Hyy_g']))), height=np.sort(idd['eig_vals_Hyy_g'].real), color="#DD22DD", alpha=0.7, label=Hyy_g_sym, log=False, width=0.6)
                imgs.extend(dd_eig_vals_bar)
                self.ax_dict["eig_vals_Hyy_g"].set_ylabel("Magnitude")
            else:
                _, _, dd_eig_vals_hist = self.ax_dict["eig_vals_Hyy_g"].hist(idd['eig_vals_Hyy_g'].real, animated=True, bins=bin_num, density=False, color="#DD22DD", alpha=0.7, log=True, label=Hyy_g_sym)
                imgs.extend(dd_eig_vals_hist)
            dd_eig_vals_hist_text = self.ax_dict["eig_vals_Hyy_g"].text(0.75, 0.5, f"{Hyy_g_sym}\nmax: {np.max(idd['eig_vals_Hyy_g'].real):.4f}\nmin: {np.min(idd['eig_vals_Hyy_g'].real):.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["eig_vals_Hyy_g"].transAxes, fontsize=13)
            imgs.append(dd_eig_vals_hist_text)

            if idd.get('eig_vals_Hxx_f_Schur', None) is not None:
                mc1_sym = r"$\lambda(H_{yy}g - H_{yx}g H_{xx}^{-1}f H_{xy}f)$"
                # print("idd['eig_vals_Hxx_f_Schur']", idd['eig_vals_Hxx_f_Schur'])
                if len(idd['eig_vals_Hxx_f_Schur']) <= 20:
                    minimax_eig_1_bar = self.ax_dict["eig_vals_Hxx_f"].bar(x=list(range(1, 1 + len(idd['eig_vals_Hxx_f_Schur']))), height=np.sort(idd['eig_vals_Hxx_f_Schur'].real), color="#222222", alpha=0.7, label=mc1_sym, log=False, width=0.3)
                    imgs.extend(minimax_eig_1_bar)
                else:
                    _, _, minimax_eig_1_hist = self.ax_dict["minimax_eig_1"].hist(idd['eig_vals_Hxx_f_Schur'].real, animated=True, bins=bin_num, density=False, color="#222222", alpha=0.4, log=True, label=mc1_sym)
                    imgs.extend(minimax_eig_1_hist)

                minimax_eig_1_hist_text = self.ax_dict["minimax_eig_1"].text(0.25, 0.5, f"{mc1_sym}\nmax: {np.max(idd['eig_vals_Hxx_f_Schur'].real):.4f}\nmin: {np.min(idd['eig_vals_Hxx_f_Schur'].real):.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["eig_vals_Hxx_f"].transAxes, fontsize=13)
                imgs.append(minimax_eig_1_hist_text)

            if idd.get('eig_vals_Hyy_g_Schur', None) is not None:
                mc2_sym = r"$\lambda(H_{xx}f - H_{xy}f H_{yy}^{-1}g H_{yx}g)$"
                # print("idd['eig_vals_Hyy_g_Schur']", idd['eig_vals_Hyy_g_Schur'])
                if len(idd['eig_vals_Hyy_g_Schur']) <= 20:
                    minimax_eig_2_bar = self.ax_dict["eig_vals_Hyy_g"].bar(x=list(range(1, 1 + len(idd['eig_vals_Hyy_g_Schur']))), height=np.sort(idd['eig_vals_Hyy_g_Schur'].real), color="#222222", alpha=0.7, label=mc2_sym, log=False, width=0.3)
                    imgs.extend(minimax_eig_2_bar)
                else:
                    _, _, minimax_eig_2_hist = self.ax_dict["minimax_eig_2"].hist(idd['eig_vals_Hyy_g_Schur'].real, animated=True, bins=bin_num, density=False, color="#222222", alpha=0.4, log=True, label=mc2_sym)
                    imgs.extend(minimax_eig_2_hist)

                minimax_eig_2_hist_text = self.ax_dict["minimax_eig_2"].text(0.25, 0.5, f"{mc2_sym}\nmax: {np.max(idd['eig_vals_Hyy_g_Schur'].real):.4f}\nmin: {np.min(idd['eig_vals_Hyy_g_Schur'].real):.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["eig_vals_Hyy_g"].transAxes, fontsize=13)
                imgs.append(minimax_eig_2_hist_text)

        if idd.get('eig_vals_J', None) is not None:
            if self.attr['args'].data not in ["mnist", "cifar"]:
                eig_vals_scatter = self.ax_dict["eig"].scatter(idd['eig_vals_J'].real, idd['eig_vals_J'].imag, color="#000000", alpha=0.3)
                imgs.append(eig_vals_scatter)

            """ Eigenvalues histogram """
            bin_num = "sqrt"
            eig_vals_modula = np.absolute(idd['eig_vals_J'])
            # _, _, eig_vals_modula_hist = self.ax_dict["eig_mod"].hist(eig_vals_modula, animated=True, bins=bin_num,
            #                                                           density=False, color="#000000", alpha=0.7,
            #                                                           log=True, label=r"$||\lambda||$")
            # imgs.extend(eig_vals_modula_hist)
            #
            # eig_vals_modula_text = self.ax_dict["eig_mod"].text(0.7, 0.5, f"max: {np.max(eig_vals_modula):.4f}\nmin: {np.min(eig_vals_modula):.4f}", {"ha": "center", "va": "center"},
            #                                                     horizontalalignment="left", verticalalignment="top",
            #                                                     transform=self.ax_dict["eig_mod"].transAxes, fontsize=13)
            # imgs.append(eig_vals_modula_text)
            #
            #
            # _, _, eig_vals_real_hist = self.ax_dict["eig_real"].hist(idd['eig_vals_J'].real, animated=True, bins=bin_num,
            #                                                          density=False, color="#22DD22", alpha=0.6,
            #                                                          log=True, label=r"$\Re(\lambda)$")
            # imgs.extend(eig_vals_real_hist)
            #
            # eig_vals_real_text = self.ax_dict["eig_real"].text(0.7, 0.5, f"max: {np.max(idd['eig_vals_J'].real):.4f}\nmin: {np.min(idd['eig_vals_J'].real):.4f}", {"ha": "center", "va": "center"},
            #                                                    horizontalalignment="left", verticalalignment="top",
            #                                                    transform=self.ax_dict["eig_real"].transAxes, fontsize=13)
            # imgs.append(eig_vals_real_text)
            #
            #
            # _, _, eig_vals_imag_hist = self.ax_dict["eig_imag"].hist(idd['eig_vals_J'].imag, animated=True, bins=bin_num,
            #                                                          density=False, color="#DD2222", alpha=0.6,
            #                                                          log=True, label=r"$\Im(\lambda)$")
            # imgs.extend(eig_vals_imag_hist)
            #
            # eig_vals_imag_text = self.ax_dict["eig_imag"].text(0.7, 0.5, f"max: {np.max(idd['eig_vals_J'].imag):.4f}\nmin: {np.min(idd['eig_vals_J'].imag):.4f}", {"ha": "center", "va": "center"},
            #                                                    horizontalalignment="left", verticalalignment="top",
            #                                                    transform=self.ax_dict["eig_imag"].transAxes, fontsize=13)
            # imgs.append(eig_vals_imag_text)

            """ Histogram for phase factor """
            phase_factor_list = np.nan_to_num(np.abs(idd['eig_vals_J'].imag / idd['eig_vals_J'].real))

            _, _, phase_factor_hist = self.ax_dict["phase_factor"].hist(phase_factor_list, animated=True, bins=bin_num, density=False, color="#AAAA22", alpha=0.6, log=True, label=r"Phase factor")
            imgs.extend(phase_factor_hist)
            phase_factor_text = self.ax_dict["phase_factor"].text(0.75, 0.5, f"max: {np.max(phase_factor_list):.4f}\nmin: {np.min(phase_factor_list):.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["phase_factor"].transAxes, fontsize=13)
            imgs.append(phase_factor_text)

            """ Line plot for conditioning factor """
            eig_vals_modula_nonzero = eig_vals_modula[eig_vals_modula > 0]
            conditioning_factor = np.nan_to_num(np.abs(np.max(eig_vals_modula_nonzero) / np.min(eig_vals_modula_nonzero)))
            self.attr_seq['conditioning_factor'].append(conditioning_factor)

            conditioning_factor_curve, = self.ax_dict["conditioning_factor"].semilogy(self.attr_seq['iter'], self.attr_seq['conditioning_factor'], '-.', linewidth=1.5, color="#0000FF", animated=True, label=r"Conditioning factor", alpha=0.7)
            imgs.append(conditioning_factor_curve)
            conditioning_factor_sci = "{:.4E}".format(self.attr_seq['conditioning_factor'][-1])
            conditioning_factor_text = self.ax_dict["conditioning_factor"].text(0.75, 0.5, f"value: {conditioning_factor_sci}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["conditioning_factor"].transAxes, fontsize=13)
            imgs.append(conditioning_factor_text)

        if (self.attr_seq['grad_corr_norm_x'] is not None) and (self.attr_seq['grad_corr_norm_y'] is not None):
            if idd.get('corr_norm_x_ma', None) is not None:
                x_corr_text = f"x corr norm MA: {idd['corr_norm_x_ma']:.4f}"
            else:
                x_corr_text = ""

            if idd.get('corr_rel_norm_x_ma', None) is not None:
                x_corr_text += f"\nrel x corr norm MA: {idd['corr_rel_norm_x_ma']:.4f}"

            if idd.get('use_x_corr', None) is not None:
                if idd['use_x_corr']:
                    x_corr_text += f"\nx corr: on"
                else:
                    x_corr_text += f"\nx corr: off"

            corr_norm_x_plot, = self.ax_dict["grad_corr_norm"].semilogy(self.attr_seq['wall_time'], self.attr_seq['grad_corr_norm_x'], '-', linewidth=2, color="#000000", animated=line_animated, label=r"$||H_{xy}H_{yy}^{-1}\nabla_y f||_2 / ||\nabla_x f||_2$", alpha=0.7, marker="x")
            imgs.append(corr_norm_x_plot)
            corr_norm_y_plot, = self.ax_dict["grad_corr_norm"].semilogy(self.attr_seq['wall_time'], self.attr_seq['grad_corr_norm_y'], '--', linewidth=2, color="#000000", animated=line_animated, label=r"$||H_{yy}^{-1}H_{yx}\nabla_x f||_2 / ||\nabla_y f||_2$", alpha=0.7, marker="1")
            imgs.append(corr_norm_y_plot)

            corr_norm_x_text = self.ax_dict["grad_corr_norm"].text(0.75, 0.5, x_corr_text, {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["grad_corr_norm"].transAxes, fontsize=13)
            imgs.append(corr_norm_x_text)

        """ Grad norms """
        grad_norm_G_plot, = self.ax_dict["grad_norm"].semilogy(self.attr_seq['iter'], self.attr_seq['grad_raw_norm_x'], '-', linewidth=1.0, color="#2222FF", animated=line_animated, label=r"$||\nabla_x f||_2$", alpha=0.7)
        imgs.append(grad_norm_G_plot)
        if self.attr_seq['update_tot_norm_x'] is not None:
            grad_norm_G_plot, = self.ax_dict["grad_norm"].semilogy(self.attr_seq['iter'], self.attr_seq['update_tot_norm_x'], '-.', linewidth=1.5, color="#2222FF", animated=line_animated, label=r"$||\nabla_x \tilde{f}||_2$", alpha=0.7)
            imgs.append(grad_norm_G_plot)

        grad_norm_D_plot, = self.ax_dict["grad_norm"].semilogy(self.attr_seq['iter'], self.attr_seq['grad_raw_norm_y'], '-', linewidth=1.0, color="#FF22FF", animated=line_animated, label=r"$||\nabla_y g||_2$", alpha=0.7)
        imgs.append(grad_norm_D_plot)
        if self.attr_seq['update_tot_norm_y'] is not None:
            grad_norm_D_plot, = self.ax_dict["grad_norm"].semilogy(self.attr_seq['iter'], self.attr_seq['update_tot_norm_y'], '-.', linewidth=1.5, color="#FF22FF", animated=line_animated, label=r"$||\nabla_y \tilde{g}||_2$", alpha=0.7)
            imgs.append(grad_norm_D_plot)

        """ Learning curve G """
        if self.attr['args'].divergence == "standard":

            learning_curve_G, = self.ax_dict["loss_G"].semilogy(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_G']), ':', linewidth=1.5, color="#0000FF", animated=line_animated, label=r"loss_G", alpha=0.7)
            imgs.append(learning_curve_G)

            learning_curve_G_tot, = self.ax_dict["loss_G"].semilogy(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_G_tot']), '-', linewidth=2.5, color="#0000FF", animated=line_animated, label=r"loss_G_tot")
            imgs.append(learning_curve_G_tot)

            """ Reference line for optimal G loss """
            opt_loss_G_ref, = self.ax_dict["loss_G"].semilogy(self.attr_seq['iter'], np.ones_like(np.array(Torch_loss_list_val_list(self.attr_seq['loss_G']))) * self.opt_loss_G_ref_val, 'r-', linewidth=1, color="#000055", animated=line_animated) # , label=r"loss_G$^*$"
            imgs.append(opt_loss_G_ref)

            opt_loss_G_val_text = self.ax_dict["loss_G"].text(1, self.opt_loss_G_ref_val, f"opt_loss_G = {self.opt_loss_G_ref_val:.5f}", fontsize=10)
            imgs.append(opt_loss_G_val_text)

            """ Learning curve D """
            learning_curve_D, = self.ax_dict["loss_D"].semilogy(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_D']), ':', linewidth=1.5, color="#FF00FF", animated=line_animated, label=r"loss_D", alpha=0.7)
            imgs.append(learning_curve_D)

            learning_curve_D_tot, = self.ax_dict["loss_D"].semilogy(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_D_tot']), '-', linewidth=2.5, color="#FF00FF", animated=line_animated, label=r"loss_D_tot")
            imgs.append(learning_curve_D_tot)

            """ Reference line for optimal D loss """
            opt_loss_D_ref, = self.ax_dict["loss_D"].semilogy(self.attr_seq['iter'], np.ones_like(np.array(Torch_loss_list_val_list(self.attr_seq['loss_D']))) * 2 * np.log(2), 'r-', linewidth=1, color="#550055", animated=line_animated, label=r"loss_D$^*$")
            imgs.append(opt_loss_D_ref)

            opt_loss_D_val_text = self.ax_dict["loss_D"].text(1, 2 * np.log(2), r"opt_loss_D = 1.38629", fontsize=10)
            imgs.append(opt_loss_D_val_text)
        else:
            # print("self.attr_seq['iter']", self.attr_seq['iter'])
            # print("self.attr_seq['loss_G']", self.attr_seq['loss_G'])
            learning_curve_G, = self.ax_dict["loss_G"].plot(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_G']), ':', linewidth=1.5, color="#0000FF", animated=line_animated, label=r"loss_G", alpha=0.7)
            imgs.append(learning_curve_G)

            learning_curve_G_tot, = self.ax_dict["loss_G"].plot(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_G_tot']), '-', linewidth=2.5, color="#0000FF", animated=line_animated, label=r"loss_G_tot")
            imgs.append(learning_curve_G_tot)

            """ Reference line for optimal G loss """
            opt_loss_G_ref, = self.ax_dict["loss_G"].plot(self.attr_seq['iter'], np.ones_like(np.array(Torch_loss_list_val_list(self.attr_seq['loss_G']))) * self.opt_loss_G_ref_val, 'r-', linewidth=1, color="#000055", animated=line_animated) # , label=r"loss_G$^*$"
            imgs.append(opt_loss_G_ref)

            opt_loss_G_val_text = self.ax_dict["loss_G"].text(1, self.opt_loss_G_ref_val, f"opt_loss_G = {self.opt_loss_G_ref_val:.5f}", fontsize=10)
            imgs.append(opt_loss_G_val_text)

            """ Learning curve D """
            learning_curve_D, = self.ax_dict["loss_D"].plot(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_D']), ':', linewidth=1.5, color="#FF00FF", animated=line_animated, label=r"loss_D", alpha=0.7)
            imgs.append(learning_curve_D)

            learning_curve_D_tot, = self.ax_dict["loss_D"].plot(self.attr_seq['iter'], Torch_loss_list_val_list(self.attr_seq['loss_D_tot']), '-', linewidth=2.5, color="#FF00FF", animated=line_animated, label=r"loss_D_tot")
            imgs.append(learning_curve_D_tot)

            """ Reference line for optimal D loss """
            opt_loss_D_ref, = self.ax_dict["loss_D"].plot(self.attr_seq['iter'], np.ones_like(np.array(Torch_loss_list_val_list(self.attr_seq['loss_D']))) * 2 * np.log(2), 'r-', linewidth=1, color="#550055", animated=line_animated, label=r"loss_D$^*$")
            imgs.append(opt_loss_D_ref)

            opt_loss_D_val_text = self.ax_dict["loss_D"].text(1, 2 * np.log(2), r"opt_loss_D = 1.38629", fontsize=10)
            imgs.append(opt_loss_D_val_text)

        """ Text """
        iter_info = self.ax_dict["loss_G"].text(0.5, 0.5, f"iter: {self.attr_seq['iter'][-1]}\nloss_G: {Torch_loss_list_val_list(self.attr_seq['loss_G_tot'])[-1]:.4f}\nloss_D: {Torch_loss_list_val_list(self.attr_seq['loss_D_tot'])[-1]:.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["loss_G"].transAxes, fontsize=8)
        imgs.append(iter_info)

        spanning_init_text = ""
        if (idd.get('spanning_init', None) is not None) and self.attr['args'].spanning_init:
            if idd['spanning_init']:
                spanning_init_text = "\nspanning ..."
            else:
                spanning_init_text = "\nspanning completed"

        time_info = self.ax_dict["loss_G"].text(0.7, 0.5,  f"wall time (s): {idd['cumul_training_time']:.2f}\nPer iter (s): {idd['cumul_training_time'] / (idd['iter'] + 1):.4f}{spanning_init_text}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["loss_G"].transAxes, fontsize=8)
        imgs.append(time_info)

        """ Legends """
        if not self.legend_drawn:
            for ax_name in self.ax_dict:
                if ax_name[:3] != "out":
                    self.ax_dict[ax_name].legend(loc="upper right")

            self.legend_drawn = True

        """ ================================== """
        self.ims.append(tuple(imgs))

        plot_time = time.time() - toc
        if idd['iter'] % self.attr["max_iter_div_5"] == 0:
            print(f"> [End iter {idd['iter']} / {self.attr['args'].iteration}], plot time taken: {plot_time:.3f}")

    def Load_data(self, filename):

        if filename[-7:] == ".pickle":
            filename = filename[:-7]

        self.data_container_generator = Load_all_pickles(filename, data_folder=self.data_folder)
        try:
            self.data_container_dict = next(self.data_container_generator)
        except:
            print("self.data_container_generator is None")
            self.data_container_dict = None

        if self.data_container_dict is None:
            return -1
        else:
            self.attr = self.data_container_dict["attr"]
            self.attr_seq = self.data_container_dict["attr_seq"]
            print("Data assigned to members.")
            self.Calculate_max_t()

            if self.attr['args'].data in ["mnist"]:
                if self.image_min is None or self.image_max is None:
                    print("Determining image intensity range")
                    x_out_list = []
                    for t in range(self.attr["args"].iteration // self.attr["args"].plot_iter + 1):
                        try:
                            idd = next(self.data_container_generator)
                        except:
                            continue

                        x_out_list.append(idd["x_out"])

                    x_out_list_np = np.array(x_out_list)
                    if len(x_out_list_np) == 0 or x_out_list_np is None:
                        self.image_min = 0
                        self.image_max = 1
                    else:
                        self.image_min = np.min(x_out_list_np.ravel())
                        self.image_max = np.max(x_out_list_np.ravel())

                    if self.image_min is None:
                        self.image_min = 0
                    if self.image_max is None:
                        self.image_max = 1
                    print(f"Image intensity range: ({self.image_min:.3f}, {self.image_max:.3f})")


                    self.Load_data(filename)

            return 0

    def Generate_video_from_file(self, title, my_part=None, num_parts=None, iter_start=0, iter_end=np.inf, skip_frame=1):

        generating_start_time = time.time()
        self.Load_data(title)

        max_t = 0
        max_iter = 0
        for t in range(self.attr["max_t"]):
            if t % (np.max([self.attr["max_t"] // 5, 1]).astype(int)) == 0:
                print("Checking... ", t, self.attr["max_t"], self.attr["args"].iteration, self.attr["args"].plot_iter)
            try:
                idd = next(self.data_container_generator)
                max_iter = idd["iter"]
                max_t = t
            except:
                print("file end")

        self.Load_data(title)
        self.attr["max_t"] = max_t
        self.attr["max_iter_div_5"] = self.attr["max_t"] // 5
        if self.attr["max_iter_div_5"] == 0:
            self.attr["max_iter_div_5"] += 1

        if iter_end == np.inf:
            iter_end = max_iter

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
            try:
                idd = next(self.data_container_generator)
            except:
                print("file end")
                return

            if t % skip_frame != 0:
                continue

            for item in self.attr_seq:
                self.attr_seq[item].append(idd.get(item, None))

            """ If this part of video is not started from the beginning, plot the previous segments line plots """
            if idd["iter"] >= iter_start and idd["iter"] <= iter_end and t >= start_pos and t < end_pos:
                if t % (np.max([self.attr["max_t"] // 5, 1]).astype(int)) == 0:
                    print(f't {t}, max_t {self.attr["max_t"]}, iteration {self.attr["args"].iteration}, plot_iter {self.attr["args"].plot_iter}')

                self.Plot_step(idd, loading=True)
                self.total_frame += 1

        print(f"Video production time: {time.time() - generating_start_time}")

    def Save_plot(self, title=None, fps=10, figures_folder=figures_folder):
        save_plot_start_time = time.time()

        if title is None:
            self.attr["name"] += f"_{Now()}"
            title = self.attr["name"]
        else:
            title += f"_{Now()}"

        self.fig.suptitle(title)
        self.attr['name'] = title
        subplots_adjust(top=.95)
        # tight_layout()

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
            ani.save(output_filename, writer=mywriter, progress_callback=progress_callback_func) # , dpi=50
        print(f"video saving time: {time.time() - save_plot_start_time}")

if __name__ == "__main__":
    pass