from matplotlib import use
# use("Qt5Agg")
# use('TkAgg')
from mpl_toolkits.mplot3d import Axes3D                         # Not explicitly used, but necessary
from matplotlib.transforms import Affine2D                      # Not explicitly used, but necessary
import mpl_toolkits.axisartist.floating_axes as floating_axes   # Not explicitly used, but necessary

import numpy as np
import random
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

import torch
from torch.nn import functional as F

from BPs import *
from ComputationalTools import *
from deepviz import *
from models import ResNet18
import Run_Aux_Training
# from Run_Aux_Training import Get_classification_logit_prob_class, Load_aux_classifier

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

""" Create a python generator for a pickle file """
def Load_all_pickles(title, data_folder=data_folder):
    print("Load_all_pickles title", title)

    if platform.system() == "Darwin":
        print("Using MacOS.")
    elif platform.system() == "Linux":
        print("Using Linux.")
    else:
        print("Using Windows.")

    with open(os.path.join(os.getcwd(), data_folder, title + ".pickle"), "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except:
                print(f"End of file: {title}")
                break

""" Record positio of panels of viz """
fig_size = 6
grid_span = 6
span_figure_r = 3
span_figure_c = 6


#%%
class DeepVisuals_real(DeepVisuals_2D):
    def __init__(self, *args, **kwargs):
        super(DeepVisuals_real, self).__init__(*args, **kwargs)
        """ Time-invariant members """ # self.attr = {}
        self.attr["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        """ Time-variant members """ # self.attr_seq = {[], [], ...}
        """ Utils members """ # self.data_container_dict = {}
        # self.last_contour_plot = None
        # self.legend_drawn = False
        # self.cmap = 'Blues'
        # self.cmap_D = 'Reds'
        # self.first_frame = True
        # self.num_parts = 1
        # self.skip_frame = 1
        # self.total_frame = 0

    def Plot_step(self, idd, loading=False):
        toc = time.time()

        G = idd.get("G", None)
        D = idd.get("D", None)


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
            self.attr_seq[item].append(idd.get(item, None))


        """ BP observables (1-hidden-layer FC network only) """
        if self.attr["args"].arch == "mlp":
            pass
        else:
            pass

        """ Image quality metric """
        # idd["inception_score"] = (is_mean, is_std)
        if self.attr["args"].data in ["mnist"]:
            pass


        """ Grad norm metric """
        z_test = self.attr["z_test"] # Fixed noise
        x_out = idd["x_out"] # Output from fixed noise
        dataset = self.attr["dataset"]

        if self.attr["args"].data in ["mnist"]:
            selected_mode_tuple_list = [(2, 8), (0, 2), (5, 6), (0, 6), (3, 5), (1, 9), (1, 8), (1, 3), (7, 8), (6, 7), (1, 4), (8, 9), (3, 8), (6, 7), (4, 5)]
            aux_classifier_loaded = ResNet18().to(self.attr["device"])
            num_mode = 10
        else:
            if self.attr["args"].data in ["grid5", "random9-6_2"]:
                selected_mode_tuple_list = [(4, 5), (4, 8), (5, 8), (7, 5)]
            else:
                selected_mode_tuple_list = [(2, 3), (3, 4), (2, 5), (4, 5)]

            args = self.attr["args"]
            dataset_config = f"{args.data}-{args.mog_scale}-{args.mog_std}"
            aux_classifier_loaded, real_data_prob, real_mode_num = Run_Aux_Training.Load_aux_classifier(dataset_config)
            num_mode = dataset.n + 1

        """ --- Find z such that G(z) is classified as the following modes """
        with torch.no_grad():
            pred_logits = aux_classifier_loaded(torch.from_numpy(x_out).to(self.attr["device"]))
            # pred_data_prob, covered_mode_num = Run_Aux_Training.Get_classification_logit_prob_class(aux_classifier_loaded, idd["x_out"])
            pred_probs = F.softmax(pred_logits, dim=1).data.cpu().numpy()

        pred_labels = np.argmax(pred_probs, axis=1)
        # print("pred_labels", pred_labels)

        mode_to_z_dict = {}

        for c in range(num_mode):
            mode_to_z_dict[c] = []
        for z, label in zip(z_test, pred_labels):
            mode_to_z_dict[label].append(z)

        """ --- Interpolate between z's and calculate gradient norm """
        n_grad_norm_sample = self.attr["args"].n_grad_norm_sample
        n_interpolate = self.attr["args"].n_interpolate

        grad_norm_timer = Timer()
        dG_dz_avg_between_modes_dict = {}
        for ss, selected_mode_tuple in enumerate(selected_mode_tuple_list):
            if len(mode_to_z_dict[selected_mode_tuple[0]]) > 0 and len(mode_to_z_dict[selected_mode_tuple[1]]) > 0\
                    and n_grad_norm_sample * n_interpolate > 0:
                dG_dz_interpolated_max_list = []
                for ll in range(n_grad_norm_sample):
                    random_z_0 = random.choice(mode_to_z_dict[selected_mode_tuple[0]])
                    random_z_1 = random.choice(mode_to_z_dict[selected_mode_tuple[1]])
                    z_interpolated = np.linspace(random_z_0, random_z_1, n_interpolate)
                    z_interpolated_torch = torch.from_numpy(z_interpolated).to(self.attr["device"])
                    z_interpolated_torch.requires_grad_(True)

                    """ TODO - Calculate grad norm for arbitrary network """
                    # dx_0_dz = autograd.grad(fake_out_on_z_interpolated[:, 0], z_interpolated_torch, create_graph=True, retain_graph=True,
                    #                         grad_outputs=torch.ones_like(fake_out_on_z_interpolated[:, 0]))[0]
                    # dx_1_dz = autograd.grad(fake_out_on_z_interpolated[:, 1], z_interpolated_torch, create_graph=True, retain_graph=True,
                    #                         grad_outputs=torch.ones_like(fake_out_on_z_interpolated[:, 1]))[0]
                    if ll == 0:
                        print(f"z_interpolated_torch.shape = {z_interpolated_torch.shape}")

                    out_jacobian_list = Get_list_of_Jacobian_for_a_batch_of_input(G, z_interpolated_torch)
                    out_batch_jacobian_flat = torch.cat(out_jacobian_list, 0)
                    dG_dz_interpolated = torch.norm(out_batch_jacobian_flat, dim=1, p=2).detach().cpu().numpy()

                    """ TODO - Change max to 95% percentile """
                    # dG_dz_interpolated_max_list.append(np.max(dG_dz_interpolated.ravel()))
                    dG_dz_interpolated_max_list.append(np.sort(dG_dz_interpolated.ravel())[np.floor(0.95 * len(dG_dz_interpolated.ravel())).astype(int) - 1])

                dG_dz_interpolated_avg_of_max = np.mean(dG_dz_interpolated_max_list)
            else:
                dG_dz_interpolated_avg_of_max = np.nan

            dG_dz_avg_between_modes_dict[f"Mode_{selected_mode_tuple[0]}_to_{selected_mode_tuple[1]}"] = dG_dz_interpolated_avg_of_max


        # print("dG_dz_avg_between_modes_dict", dG_dz_avg_between_modes_dict)
        grad_norm_timer.Print(print_time=True, msg=f"n_grad_norm_sample = {n_grad_norm_sample}, n_interpolate = {n_interpolate}, z_test.shape = {z_test.shape}")




if __name__ == "__main__":
    pass