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
    if platform.system() == "Darwin":
        print("Using MacOS.")
    elif platform.system() == "Linux":
        print("Using Linux.")
    else:
        print("Using Windows.")

    with open(os.path.join(os.getcwd(), data_folder, title + ".pickle"), "rb") as f:
        while True:
            yield pickle.load(f)
            # try:
            #     yield pickle.load(f)
            # except EOFError:
            #     break

""" Record positio of panels of viz """
grid_size = 6
plt.rcParams.update({'font.size': grid_size * 2})

span_figure_r = 1
span_figure_c = 1
fig_size = 6
fig_size_h = 7
fig_size_w = 7
eig_loc = 0
in_loc = 1
out_loc = 2
ls_loc = 3
eig_mod_loc = 4
eig_real_loc = 5
eig_imag_loc = 6
phase_factor_loc = 7
conditioning_factor_loc = 8
grad_loc = 2

#%%
class DeepVisuals_1D():
    def __init__(self, args=None, dataset=None, name="", real_time_video=False, z_test=None, x_real=None, x_real_label=None, z_linspace=None, x_linspace=None, init_figure=True):
        self.real_time_video = real_time_video
        self.ims = []
        self.ax_dict = {}

        if init_figure:
            if self.real_time_video:
                self.fig = pl.figure(figsize=(2 * 4, 1 * 4))
                """ Layout of panels """
            else:
                self.fig = pl.figure(figsize=(span_figure_c * fig_size_w, span_figure_r * fig_size_h))
                """ Layout of panels """
                self.ax_dict["horizontal_z"] = plt.subplot2grid((span_figure_r * grid_size, span_figure_c * grid_size), (0 * grid_size, 0 * grid_size), rowspan=1 * grid_size, colspan=1 * grid_size)

        """ Time-invariant members """
        self.attr = {}
        self.attr["args"] = args
        self.attr["name"] = name
        self.attr["dataset"] = dataset
        self.attr["z_test"] = z_test
        self.attr["x_real"] = x_real
        self.attr["z_linspace"] = z_linspace
        self.attr["x_linspace"] = x_linspace
        self.attr["timestamp"] = Now()

        """ Time-variant members """
        self.attr_seq = {}
        attr_seq_list = ["iter", "loss_G", "loss_D", "loss_G_tot", "loss_D_tot", "grad_raw_norm_x", "grad_raw_norm_y", "grad_corr_norm_x", "grad_corr_norm_y",
         "update_tot_norm_x", "update_tot_norm_y", "cumul_training_time"]

        for item in attr_seq_list:
            self.attr_seq[item] = []

        """ Utils members """
        self.data_container_dict = {}
        self.legend_drawn = False
        self.total_frame = 0
        self.BP_colors = None

        self.handle = None
        if name != "":
            if not os.path.exists(data_folder):
                os.makedirs(data_folder)

            self.handle = open(os.path.join(os.getcwd(), data_folder, self.attr["name"] + f"_{self.attr['timestamp']}.pickle"), "wb")

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

        """ Test data """
        """ Real data """
        self.ax_dict["horizontal_z"].hist(self.attr["z_test"], bins=100, label=r"$z$", bottom=self.attr["dataset"].xlim[0], color="#AAAACC", alpha=0.6, density=True)
        self.ax_dict["horizontal_z"].set_xlabel(r"$z$")
        # self.ax_dict["horizontal_z"].set_ylabel(r"$y$")
        # self.ax_dict["horizontal_z"].set_aspect(1)
        self.ax_dict["horizontal_z"].set_xlim(-4 * self.attr["args"].z_std, 4 * self.attr["args"].z_std)
        self.ax_dict["horizontal_z"].set_ylim(self.attr["dataset"].xlim)

        cdf_z = np.array(norm.cdf(self.attr["z_linspace"]))
        x_gt_np, _ = self.attr["dataset"].MG.Solve_inverse_mixed_CDF_acc(cdf_z, precise_gt=True)
        # self.ax_dict["horizontal_z"].set_ylim(self.attr["dataset"].ylim)
        self.ax_dict["horizontal_z"].plot(self.attr["z_linspace"], x_gt_np, c='#00FF00', label=r"$G_+^*$", linewidth=1)
        self.ax_dict["horizontal_z"].plot(-self.attr["z_linspace"], x_gt_np, c='#00AA00', label=r"$G_-^*$", linewidth=1)

        self.ax_dict["vertical_x"] = self.ax_dict["horizontal_z"].twiny()
        self.ax_dict["vertical_x"].set_xticklabels([])
        self.ax_dict["vertical_x"].set_ylabel(r"x")
        self.ax_dict["vertical_x"].set_ylim(self.attr["dataset"].xlim)
        self.ax_dict["vertical_x"].hist(self.attr["x_real"], bins=100, label=r"$x_{\mathrm{real}}$", color="#66AAAA", alpha=0.6, orientation="horizontal", density=True)

        print("Figure intialized.")


    # def Plot_step(self, iter, preds, preds_linspace, loss, loading=False):
    def Plot_step(self, idd, loading=False, plotting=False):
        toc = time.time()

        G = idd.get("G", None)
        D = idd.get("D", None)

        if not loading:
            """ Adding items into data_container """
            if self.handle is not None:
                del idd["G"]
                del idd["D"]
                pickle.dump(idd, self.handle) # idd = iter_data_dict
            if not self.real_time_video:
                if idd["iter"] % self.attr["max_iter_div_5"] == 0:
                    plot_time = time.time() - toc
                    print(f"> Mid [iter {idd['iter']} / {self.attr['args'].iteration}], plot time taken: {plot_time:.3f}")
                return

        for item in self.attr_seq:
            self.attr_seq[item].append(idd[item])
        line_animated = True

        """ BPs """
        BP_directions, BP_signed_distances, BP_delta_slopes = Get_BP_params(idd["state_dict_G"]["hidden_layer.weight"], idd["state_dict_G"]["hidden_layer.bias"], idd["state_dict_G"]["output_layer.weight"])

        if self.BP_colors is None:
            color_map = plt.get_cmap("gist_rainbow")
            BP_signed_distances_01 = np.linspace(0, 1, self.attr["args"].g_hidden)
            self.BP_signed_distances_argsort = np.argsort(BP_signed_distances)
            self.BP_colors = [color_map(BP_signed_distances_01[np.where(self.BP_signed_distances_argsort == i)[0][0]]) for i in range(len(self.BP_signed_distances_argsort))]
            # self.ax_dict["horizontal_z"].scatter(BP_signed_distances[:self.attr['alt_act_num']], np.zeros_like(BP_signed_distances[:self.attr['alt_act_num']]), c=self.BP_colors[:self.attr['alt_act_num']], marker="1", alpha=0.7, s=2)
            # self.ax_dict["horizontal_z"].scatter(BP_signed_distances[self.attr['alt_act_num']:], np.zeros_like(BP_signed_distances[self.attr['alt_act_num']:]), c=self.BP_colors[self.attr['alt_act_num']:], marker="|", alpha=0.7, s=2)
            self.ax_dict["horizontal_z"].scatter(BP_signed_distances, np.zeros_like(BP_signed_distances), c=self.BP_colors, marker="|", alpha=0.7, s=2)
            print(f"iter {idd['iter']}, self.BP_colors calculated <<<<")


        if plotting:
            imgs = []
            """ G(z) function """
            G_z_plot, = self.ax_dict["horizontal_z"].plot(self.attr["z_linspace"], idd['G_z_linspace'], '-', linewidth=1, color="#0000FF", animated=True, label=r"$G(z)$", alpha=0.7)
            imgs.append(G_z_plot)

            """ Histogram of x fake """
            _, _, x_fake_hist = self.ax_dict["vertical_x"].hist(idd['x_out'], animated=True, density=True, color="#0000FF", alpha=0.7, log=False, label=r"$x_{\mathrm{fake}}$", orientation="horizontal", bins=120)
            imgs.extend(x_fake_hist)

            iter_info = self.ax_dict["horizontal_z"].text(0.2, 0.2, f"iter: {idd['iter']}\nloss_G_tot: {idd['loss_G_tot']:.4f}\nloss_D_tot: {idd['loss_D_tot']:.4f}\nspiky init: {idd['performing_spiky_init']}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict["horizontal_z"].transAxes, fontsize=8)
            imgs.append(iter_info)

            """ BP loc """
            # BP_loc_scatter = self.ax_dict["horizontal_z"].scatter(BP_signed_distances[:self.attr['alt_act_num']], BP_delta_slopes[0][:self.attr['alt_act_num']] * self.attr['args'].alt_act_factor, c=self.BP_colors[:self.attr['alt_act_num']], marker="v", s=30)
            # imgs.append(BP_loc_scatter)
            # BP_loc_scatter = self.ax_dict["horizontal_z"].scatter(BP_signed_distances[self.attr['alt_act_num']:], BP_delta_slopes[0][self.attr['alt_act_num']:], c=self.BP_colors[self.attr['alt_act_num']:], s=4, alpha=0.7)
            # imgs.append(BP_loc_scatter)

            BP_loc_scatter = self.ax_dict["horizontal_z"].scatter(BP_signed_distances, BP_delta_slopes[0], c=self.BP_colors, s=4, alpha=0.7)
            imgs.append(BP_loc_scatter)

            """ Legends """
            if not self.legend_drawn:
                handles_list = []
                labels_list = []
                for ax_name in [f"horizontal_z", f"vertical_x"]:
                    handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                    handles_list.extend(handles)
                    labels_list.extend(labels)

                self.ax_dict["horizontal_z"].legend(handles_list, labels_list, loc="upper right")
                self.legend_drawn = True

            """ ================================== """
            self.ims.append(tuple(imgs))

        plot_time = time.time() - toc
        if idd['iter'] % self.attr["max_iter_div_5"] == 0:
            print(f"> [End iter {idd['iter']} / {self.attr['args'].iteration}], plot time taken: {plot_time:.3f}")

    def Load_data(self, title, data_folder=data_folder):
        self.data_container_generator = Load_all_pickles(title, data_folder=data_folder)
        self.data_container_dict = next(self.data_container_generator)

        self.attr = self.data_container_dict["attr"]
        self.attr_seq = self.data_container_dict["attr_seq"]
        print("Data assigned to members.")
        print("args:\n", self.attr["args"])

    def Generate_video_from_file(self, title, my_part=None, num_parts=None, iter_start=0, iter_end=np.inf, skip_frame=1):

        generating_start_time = time.time()
        self.Load_data(title)
        self.Calculate_max_t()
        """ Get actual number of frames """
        max_t = 0
        max_iter = 0
        for t in range(self.attr["max_t"]):
            if t % np.max([1, self.attr["max_t"] // 5]) == 0:
                print("Checking... ", t, self.attr["max_t"], self.attr["args"].iteration, self.attr["args"].plot_iter)
            try:
                idd = next(self.data_container_generator)
                max_iter = idd["iter"]
                max_t = t
            except:
                print("file end")

        self.Load_data(title)
        self.attr["max_t"] = max_t
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

            """ If this part of video is not started from the beginning, plot the previous segments line plots """
            if idd["iter"] >= iter_start and idd["iter"] <= iter_end and t >= start_pos and t < end_pos:
                if t % np.max([1, self.attr["max_t"] // 5]) == 0:
                    print(f't {t}, max_t {self.attr["max_t"]}, iteration {self.attr["args"].iteration}, plot_iter {self.attr["args"].plot_iter}')
                self.Plot_step(idd, loading=True, plotting=True)
                self.total_frame += 1
            else:
                self.Plot_step(idd, loading=True, plotting=False)

        print(f"Video production time: {time.time() - generating_start_time}")

    def Save_plot(self, title=None, fps=10):
        save_plot_start_time = time.time()

        self.fig.suptitle(self.attr["name"])
        if title is None:
            self.attr["name"] += f"_{Now()}"
            title = self.attr["name"]
        else:
            title += f"_{Now()}"

        self.attr["title"] = title
        subplots_adjust(top=.9)
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

        if len(title) > 250:
            title = title[:250]
        output_filename = os.path.join(figures_folder, title + ".mp4")
        print(f"output_filename: {output_filename}")

        def progress_callback_func(i, n):
            prog_num_5 = np.round(self.total_frame / 5, 0).astype(int)
            if prog_num_5 == 0:
                prog_num_5 += 1
            if i % prog_num_5 == 0 or i == self.total_frame - 1:
                print(f'Saving frame {i + 1} of {self.total_frame}')

        ani.save(output_filename, writer=mywriter, progress_callback=progress_callback_func) # , dpi=50
        print(f"video saving time: {time.time() - save_plot_start_time:.3f}")

if __name__ == "__main__":
    pass