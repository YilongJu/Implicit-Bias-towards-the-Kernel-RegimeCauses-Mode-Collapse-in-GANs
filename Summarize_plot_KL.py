from utils.deepviz import *
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import os
import time
import glob
import argparse
import pandas as pd

import tensorflow as tf
from keras import backend as K

from utils.data import MOG_2D
from follow_ridge_2D import test_data_num
from follow_ridge_2D import activation_fn
from follow_ridge_2D import epsilon

try:
    import winsound  # for sound
except:
    print("Not using Windows")



parser = argparse.ArgumentParser()
parser.add_argument("--iteration", "--iter", type=int, default=1000, help="number of iterations of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--learning_rate", "--lr", type=float, default=0.0001, help="classifier learning rate")
parser.add_argument("--neurons", type=int, default=32, help="dimension of hidden units")
parser.add_argument("--layers", type=int, default=5, help="num of hidden layer")
parser.add_argument("--act", type=str, default="relu", help="which activation function for disc")  # elu, relu, tanh
parser.add_argument("--x_dim", type=int, default=2, help="data dimension")
parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
parser.add_argument("--data", type=str, default="grid", help="which dataset")
parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
parser.add_argument("--seed", type=int, default=2020, help="random seed")
parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument("--plot_iter", type=int, default=100, help="number of iter to plot")
parser.add_argument("--save_iter", type=int, default=1000, help="number of iter to save")
parser.add_argument("--summary_dir", type=str, default="Summaries", help="which summary dir")
parser.add_argument("--task_dir", type=str, default="Apr6_fr_fr3_noext", help="which task dir")
parser.add_argument("--save_csv", action='store_true', help="whether to save csv")
opt = parser.parse_args()


summary_dir = opt.summary_dir
task_dir = opt.task_dir
# task_dir = "Apr5_2"
# task_dir = "Apr6_fr_fr3_simgd_noext"
# task_dir = "Apr11_random2"
# task_dir = "Apr11_grid5"
# task_dir = "Apr11_circle2"
# task_dir = "Apr11_circle2c"
# task_dir = "Apr11_separated"
# task_dir = "Apr12_seed0"
# task_dir = "Apr12_seed2020_lr0.01"
# task_dir = "Apr16_5layers_seed2020"
# task_dir = "Apr16_5layers_seed0"
# task_dir = "Apr16_random0"
# task_dir = "Apr16_random9_seed2020"
task_dir = "May27_seed2020_nopre"


save_csv = opt.save_csv


#circle2c_fr2_JS_it160000_G2-32_D2-32_lrG0.001_lrD0.001_wd0.0001_dpTrue_adFalse_ad2False_th0.01_rth0.0001_noextTrue_bs128_ds0.1_siFalse_silr0.01_dzFalse_sd2020_int5_183631.pickle
#circle2c_fr2_JS_it160000_G2-32_D2-32_lrG0.001_lrD0.001_wd0.0001_dpTrue_adFalse_ad2False_th0.01_rth0.0001_noextTrue_bs128_ds0.1_siFalse_silr0.01_dzFalse_sd2020_int5_183631

data_dir = os.path.join(summary_dir, task_dir)
file_list = os.listdir(data_dir)

def Remove_extension(str_in, delimiter=".", ext="pickle"):
    str_split_list = str_in.split(delimiter)
    if str_split_list[-1] == ext:
        str_out = delimiter.join([str_split for str_split in str_split_list[:-1]])
    else:
        str_out = str_in

    return str_out

print(file_list)
summary_dict_keys = ["data", "method", "spanning", "opt", "divergence", "iter", "wall_time", "iter_per_sec", "loss_G_tot", "loss_D_tot", "grad_G_tot", "grad_D_tot", "max_H_GG", "min_H_GG", "max_H_DD", "min_H_DD", "max_H_MM", "min_H_MM"]
summary_dict = dict.fromkeys(summary_dict_keys)
for key in summary_dict:
    summary_dict[key] = []
    
plot_dict_list_keys = ["name", "config", "wall_time_list", "iter_list", "grad_norm_list", "grad_norm_y_list", "params_l2_dist_list", "corr_norm_x_list", "corr_rel_norm_x_list", "x_out_list"]
plot_dict_list = []
unique_data_type_dict = {}

real_data_dict = {}
real_data_prob_dict = {}
real_data_mode_num_dict = {}

checkpoint_dict = {"grid5": "Aux_grid5_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05", "random2": "Aux_random2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05", "circle2": "Aux_circle2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05", "circle2c": "Aux_circle2c_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05", "separated": "Aux_separated_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05", "random-3_1": "Aux_random-3_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_190236", "random-3_2": "Aux_random-3_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_191759", "random-6_1": "Aux_random-6_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_193310", "random-6_2": "Aux_random-6_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_194837", "random-9_1": "Aux_random-9_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_200427", "random-9_2": "Aux_random-9_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_201944", "random-12_1": "Aux_random-12_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_203439", "random-12_2": "Aux_random-12_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200420_204852", "random9-3_1": "Aux_random9-3_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_011206", "random9-3_2": "Aux_random9-3_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_012358", "random9-6_1": "Aux_random9-6_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_013536", "random9-6_2": "Aux_random9-6_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_014708", "random9-9_1": "Aux_random9-9_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_015832", "random9-9_2": "Aux_random9-9_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_021004", "random9-12_1": "Aux_random9-12_1_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_022140", "random9-12_2": "Aux_random9-12_2_ds0.1_sd2020_5-32_it200000_bs128_lr5e-05_20200427_023315"}


for i, data_file in enumerate(file_list):
    print(f"{i} / {len(file_list)}", data_file)
    print(Remove_extension(data_file))
    deepVisuals_2D = DeepVisuals_2D()
    deepVisuals_2D.Load_data(Remove_extension(data_file), data_folder=data_dir)

    plot_dict = dict.fromkeys(plot_dict_list_keys)
    for key in plot_dict:
        plot_dict[key] = []
    plot_dict["name"] = deepVisuals_2D.name
    plot_dict["data"] = deepVisuals_2D.Get_data_type()
    plot_dict["adapt"] = False
    plot_dict["adapt2"] = False
    if "adTrue" in plot_dict["name"] or "ad_" in plot_dict["name"]:
        plot_dict["adapt"] = True
    if "ad2True" in plot_dict["name"] or "ad2_" in plot_dict["name"]:
        plot_dict["adapt2"] = True

    plot_dict["method"] = deepVisuals_2D.method
    plot_dict["spanning"] = deepVisuals_2D.spanning_init_0
    plot_dict["opt"] = deepVisuals_2D.opt_type
    plot_dict["divergence"] = deepVisuals_2D.divergence
    plot_dict["stop_x_corr_iter"] = -1
    plot_dict["stop_x_corr_time"] = -1
    plot_dict["stop_span_iter"] = -1
    plot_dict["stop_span_time"] = -1


    use_x_corr_0 = False
    use_span_0 = False

    last_iter = 0
    last_time = 0.
    for t in range(deepVisuals_2D.max_t):
        try:
            idd = next(deepVisuals_2D.data_container_generator)
        except:
            continue
        if t > 0 and (np.isinf(idd["path_grad_norm_list"][10]) or np.isnan(idd["path_grad_norm_list"][10]) or np.isnan(np.sqrt(idd["grads_norm_tot_G_out"] ** 2 + idd["grads_norm_tot_D_out"] ** 2))):
            break

        plot_dict["iter_list"].append(idd["iter"])
        print("cumul_training_time", idd["cumul_training_time"])
        plot_dict["wall_time_list"].append(idd["cumul_training_time"])
        plot_dict["grad_norm_list"].append(np.sqrt(idd["grads_norm_tot_G_out"] ** 2 + idd["grads_norm_tot_D_out"] ** 2))
        plot_dict["grad_norm_y_list"].append(idd["grads_norm_tot_D_out"])
        if "corr_norm_x" not in idd:
            idd["corr_norm_x"] = 0.
        if "corr_rel_norm_x" not in idd:
            idd["corr_rel_norm_x"] = 0.
        plot_dict["corr_norm_x_list"].append(idd["corr_norm_x"])
        plot_dict["corr_rel_norm_x_list"].append(idd["corr_rel_norm_x"])
        plot_dict["x_out_list"].append(idd["x_out"])

        if t == 0:
            if "use_x_corr" not in idd:
                idd["use_x_corr"] = False
            use_x_corr_0 = idd["use_x_corr"]
            if "spanning_init" not in idd:
                idd["spanning_init"] = False
            use_span_0 = idd["spanning_init"]

        if t > 0:
            plot_dict["params_l2_dist_list"] = idd["params_l2_dist_list"]

            if use_x_corr_0 and (not idd["use_x_corr"]):
                plot_dict["stop_x_corr_iter"] = last_iter
                plot_dict["stop_x_corr_time"] = last_time
                use_x_corr_0 = False

            if use_span_0 and (not idd["spanning_init"]):
                plot_dict["stop_span_iter"] = last_iter
                plot_dict["stop_span_time"] = last_time
                use_span_0 = False

        # print(idd["use_x_corr"])

        if t == deepVisuals_2D.max_t - 1:
            plot_dict["params_l2_dist_list"] = idd["params_l2_dist_list"]
            # print(idd)
            summary_dict["data"].append(deepVisuals_2D.Get_data_type())
            summary_dict["method"].append(deepVisuals_2D.method)
            summary_dict["spanning"].append(deepVisuals_2D.spanning_init_0)
            summary_dict["opt"].append(deepVisuals_2D.opt_type)
            summary_dict["divergence"].append(deepVisuals_2D.divergence)
            summary_dict["iter"].append(idd["iter"])
            summary_dict["wall_time"].append(idd["cumul_training_time"])
            summary_dict["iter_per_sec"].append(idd["cumul_training_time"] / idd["iter"])
            summary_dict["loss_G_tot"].append(idd["g_loss_tot"])
            summary_dict["loss_D_tot"].append(idd["d_loss_tot"])
            summary_dict["grad_G_tot"].append(idd["grads_norm_tot_G_out"])
            summary_dict["grad_D_tot"].append(idd["grads_norm_tot_D_out"])
            summary_dict["max_H_GG"].append(np.max(idd["gg_eig_vals"].real))
            summary_dict["min_H_GG"].append(np.min(idd["gg_eig_vals"].real))
            summary_dict["max_H_DD"].append(np.max(idd["dd_eig_vals"].real))
            summary_dict["min_H_DD"].append(np.min(idd["dd_eig_vals"].real))
            summary_dict["max_H_MM"].append(np.max(idd["eig_vals_2"].real))
            summary_dict["min_H_MM"].append(np.min(idd["eig_vals_2"].real))
        try:
            pass
        except:
            print(f"[Failed] {Remove_extension(data_file)}")

        last_iter = idd["iter"]
        last_time = idd["cumul_training_time"]

    print("stop_x_corr_iter", plot_dict["stop_x_corr_iter"])
    print("stop_x_corr_time", plot_dict["stop_x_corr_time"])
    print("stop_span_iter", plot_dict["stop_span_iter"])
    print("stop_span_time", plot_dict["stop_span_time"])
    print("grad_norm_list", plot_dict["grad_norm_list"])
    print("params_l2_dist_list", plot_dict["params_l2_dist_list"])
    plot_dict_list.append(plot_dict)
    if plot_dict["data"] not in unique_data_type_dict:
        unique_data_type_dict[plot_dict["data"]] = [plot_dict]
    else:
        unique_data_type_dict[plot_dict["data"]].append(plot_dict)


print(plot_dict_list)

line_type_dict = {"fr": "-", "fr2": "--", "fr3": "-.", "fr4": ":", "fr5": ":", "simgd": "-", \
                  "jare": "-", "jare1": "-", "jare.5": "--", "jare-.5": "--", "jare.2": "-.", "jare.1": ":", "cgd": "-"}  # "#AA6622"
line_color_dict = {"fr": "#FF0000", "fr2": "#00FFFF", "fr3": "#00FF00", "fr4": "#AAAA00", "fr5": "#0000FF", "simgd": "#000000", \
                   "jare": "#666622", "jare1": "#666622", "jare.5": "#996622", "jare-.5": "#226699", "jare.2": "#CC6622", "jare.1": "#FF6622", \
                   "cgd": "#FF00AA", \
                   "grad_x": "#000000", "grad_x2": "#FF00FF", "grad_y": "#00FF00", "grad_y2": "#00FFFF"}
method_name_dict = {"fr": "FR", "fr2": "AltFR", "fr3": "BothFR", "fr4": "AdaptFR-1", "fr5": "AdaptFR-2", \
                    "simgd": "SimGD", \
                    "jare": "JARE", "jare1": "JARE 1", "jare.5": "JARE 0.5", "jare-.5": "JARE -0.5", \
                    "jare.2": "JARE 0.2", "jare.1": "JARE 0.1", \
                    "cgd": "CGD"}

data_type_visit_dict = {}

for data_type, plot_dict_list in unique_data_type_dict.items():
    print(data_type, checkpoint_dict[data_type])
    """ ========================================================================
     >>>>>>>>>>>>>>>>>>>>>>>>>>>> [Classifier Model] <<<<<<<<<<<<<<<<<<<<<<<<<<<
    ======================================================================== """
    K.clear_session()
    tf.reset_default_graph()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    title = f"Aux_{opt.data}_ds{opt.data_std}_sd{opt.seed}_{opt.layers}-{opt.neurons}_it{opt.iteration}_bs{opt.batch_size}_lr{opt.learning_rate}"

    root_dir = 'Aux_Models'
    os.makedirs(root_dir, exist_ok=True)

    model_save_folder = os.path.join(root_dir, title + time.strftime("_%Y%m%d_%H%M%S"))
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    """
    >>> Data
    """
    rng = np.random.RandomState(seed=opt.seed)
    data_generator = MOG_2D(data_type, rng, std=opt.data_std, sample_per_mode=1000)
    n_modes = data_generator.n

    """
    >>> Model
    """
    def Classifier(x, n_hidden=opt.neurons, n_layer=opt.layers, reuse=False,
                   initializer=(tf.glorot_normal_initializer(seed=opt.seed) if opt.init == 'xavier'
                   else tf.initializers.orthogonal(gain=1.0))):
        with tf.variable_scope(f"classifier", reuse=reuse):
            for i in range(n_layer):
                x = tf.layers.dense(x, n_hidden, activation=activation_fn(opt.act), kernel_initializer=initializer)
            x = tf.layers.dense(x, n_modes + 1, activation=None, kernel_initializer=initializer)
        return x

    """
    >>> Pipeline
    """
    all_samples_ = tf.placeholder(tf.float32, [None, opt.x_dim])
    all_labels_onehot_ = tf.placeholder(tf.int32, [None, data_generator.n + 1])

    classifier_logits_ = Classifier(all_samples_)
    loss_classification_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=all_labels_onehot_, logits=classifier_logits_))

    classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, f"classifier")
    optimizer = tf.train.AdamOptimizer(opt.learning_rate)
    """
    >>> Saver
    """
    saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=8, write_version=tf.train.SaverDef.V1)
    """
    >>> Graph
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
    # sess = tf.Session(config=config)
        """
        >>> Training
        """
        feed_dict = {all_samples_: data_generator.all_samples, all_labels_onehot_: data_generator.all_labels_onehot}
        sess.run(tf.global_variables_initializer())

        def Calculate_prob(sess, data_type, x_in):
            if data_type in checkpoint_dict:
                latest_ckpt = tf.train.latest_checkpoint(os.path.join(root_dir, checkpoint_dict[data_type]))
                # print(data_type, checkpoint_dict[data_type])
                saver.restore(sess, latest_ckpt)
                classifier_probs = sess.run(tf.nn.softmax(classifier_logits_), feed_dict={all_samples_: x_in})
                classifier_classes = np.argmax(classifier_probs, axis=1)
                classifier_classes_onehot = np.zeros_like(classifier_probs)
                classifier_classes_onehot[np.arange(len(classifier_probs)), classifier_classes] = 1
                return classifier_probs, classifier_classes_onehot
            else:
                raise NotImplementedError

        """ ========================================================================
         >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> [Plot] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ======================================================================== """
        # x_out_prob = sess.run(tf.nn.softmax(classifier_logits_), feed_dict={all_samples_: idd["x_out"]})
        # plot_dict["x_out_prob_list"].append(x_out_prob)
        if data_type not in real_data_dict:
            data_generator = MOG_2D(data_type, rng, std=opt.data_std, num_samples=test_data_num)
            real_data_dict[data_type] = data_generator.sample_points
            classifier_probs, classifier_classes_onehot = Calculate_prob(sess, data_type, real_data_dict[data_type])
            real_data_prob_dict[data_type] = np.mean(classifier_probs, axis=0)
            print("classifier_probs\n", classifier_probs[:10, :])
            print("classifier_classes_onehot\n", classifier_classes_onehot[:10])
            classifier_classes_onehot_count = np.sum(classifier_classes_onehot, axis=0)
            real_data_mode_num_dict[data_type] = np.sum(classifier_classes_onehot_count[:-1] > 0)

        # print("x_out shape", idd["x_out"].shape)
        print("real_data_prob.shape", real_data_prob_dict[data_type].shape)
        print("real_data_prob", real_data_prob_dict[data_type])
        # print("x_out_prob.shape", x_out_prob.shape)
        # print("x_out_prob", x_out_prob)

        figsize = 3
        n_rows = 7
        n_cols = 6
        fig = plt.figure(figsize=(n_cols * figsize, n_rows * figsize))
        ax_dict = {}
        class Indexer:
            def __init__(self):
                self.v = -1

            def Index(self):
                self.v += 1
                return self.v

        indexer_1 = Indexer()
        indexer_2 = Indexer()

        ax_dict["wall_time"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["grad_norm"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["grad_norm_y"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["l2_dist"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["KL"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["mode_num"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["corr_norm_x"] = plt.subplot2grid((n_rows, n_cols), (indexer_1.Index(), 0), colspan=3)
        ax_dict["corr_rel_norm_x"] = ax_dict["corr_norm_x"].twinx()
        ax_dict["wall_time2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["grad_norm2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["grad_norm_y2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["l2_dist2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["KL2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["mode_num2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["corr_norm_x2"] = plt.subplot2grid((n_rows, n_cols), (indexer_2.Index(), 3), colspan=3)
        ax_dict["corr_rel_norm_x2"] = ax_dict["corr_norm_x2"].twinx()


        grad_norm_symbol = r"$||\nabla_\theta f||_2$"
        grad_norm_y_symbol = r"$||\nabla_y f||_2$"
        corr_norm_x_symbol = r"$||H_{xy}H_{yy}^{-1}\nabla_y f||_2$"
        corr_rel_norm_x_symbol = r"$||H_{xy}H_{yy}^{-1}\nabla_y f||_2 / |\nabla_x f||_2$"
        l2_dist_symbol = r"$||\theta_T - \theta_t||_2$"
        ax_dict["wall_time"].set_xlabel("Wall Time (s)")
        ax_dict["wall_time"].set_ylabel("Iteration")

        ax_dict["grad_norm"].set_xlabel("Wall Time (s)")
        ax_dict["grad_norm"].set_ylabel(grad_norm_symbol)

        ax_dict["grad_norm_y"].set_xlabel("Wall Time (s)")
        ax_dict["grad_norm_y"].set_ylabel(grad_norm_y_symbol)

        ax_dict["corr_norm_x"].set_xlabel("Wall Time (s)")
        ax_dict["corr_norm_x"].set_ylabel(corr_norm_x_symbol)
        ax_dict["corr_rel_norm_x"].set_ylabel(corr_rel_norm_x_symbol)

        ax_dict["l2_dist"].set_xlabel("Wall Time (s)")
        ax_dict["l2_dist"].set_ylabel(l2_dist_symbol)

        ax_dict["KL"].set_xlabel("Wall Time (s)")
        ax_dict["KL"].set_ylabel("KL Divergence")

        ax_dict["mode_num"].set_xlabel("Wall Time (s)")
        ax_dict["mode_num"].set_ylabel("# of Modes Covered")

        ax_dict["wall_time2"].set_xlabel("Iteration")
        ax_dict["wall_time2"].set_ylabel("Wall Time (s)")

        ax_dict["grad_norm2"].set_xlabel("Iteration")
        ax_dict["grad_norm2"].set_ylabel(grad_norm_symbol)

        ax_dict["grad_norm_y2"].set_xlabel("Iteration")
        ax_dict["grad_norm_y2"].set_ylabel(grad_norm_y_symbol)

        ax_dict["corr_norm_x2"].set_xlabel("Iteration")
        ax_dict["corr_norm_x2"].set_ylabel(corr_norm_x_symbol)
        ax_dict["corr_rel_norm_x2"].set_ylabel(corr_rel_norm_x_symbol)

        ax_dict["l2_dist2"].set_xlabel("Iteration")
        ax_dict["l2_dist2"].set_ylabel(l2_dist_symbol)

        ax_dict["KL2"].set_xlabel("Iteration")
        ax_dict["KL2"].set_ylabel("KL Divergence")

        ax_dict["mode_num2"].set_xlabel("Iteration")
        ax_dict["mode_num2"].set_ylabel("# of Modes Covered")

        for plot_dict in plot_dict_list:
            print(plot_dict)

            plot_dict["x_out_prob_list"] = []
            plot_dict["mode_num_list"] = []
            for x_out in plot_dict["x_out_list"]:
                classifier_probs, classifier_classes_onehot = Calculate_prob(sess, data_type, x_out)
                plot_dict["x_out_prob_list"].append(np.mean(classifier_probs, axis=0))
                classifier_classes_onehot_count = np.sum(classifier_classes_onehot, axis=0)
                mode_num = np.sum(classifier_classes_onehot_count[:-1] > 0)
                plot_dict["mode_num_list"].append(mode_num)

            for x_out_prob in plot_dict["x_out_prob_list"]:
                print("x_out_prob.shape", x_out_prob.shape)
                print("x_out_prob", x_out_prob)

            plot_dict["KL_list"] = [np.sum(real_data_prob_dict[data_type] * np.log(real_data_prob_dict[data_type] / (x_out_prob + epsilon))) for x_out_prob in plot_dict["x_out_prob_list"]]
            print("KL_list", plot_dict["KL_list"])


            if plot_dict["method"] == "fr3" and plot_dict["adapt"]:
                plot_dict["method"] = "fr4"
            if plot_dict["method"] == "fr3" and plot_dict["adapt2"]:
                plot_dict["method"] = "fr5"

            print(len(plot_dict["iter_list"]))
            print(len(plot_dict["wall_time_list"]))
            print(len(plot_dict["grad_norm_list"]))
            print(len(plot_dict["params_l2_dist_list"]))
            if len(plot_dict["params_l2_dist_list"]) < len(plot_dict["iter_list"]):
                if len(plot_dict["params_l2_dist_list"]) == 0:
                    plot_dict["params_l2_dist_list"].append(0.)
                else:
                    plot_dict["params_l2_dist_list"].append(plot_dict["params_l2_dist_list"][-1])

            spanning_text = ""
            if plot_dict["spanning"]:
                spanning_text = " span"
                wall_time_line_type = "--"
                grad_norm_line_type = "--"
                l2_dist_line_type = ":"
            else:
                wall_time_line_type = "-"
                grad_norm_line_type = "-"
                l2_dist_line_type = "-."

            if plot_dict["stop_span_iter"] > 0:
                plot_dict["iter_list"] = [it - plot_dict["stop_span_iter"] for it in plot_dict["iter_list"]]
                print("acutal iter", plot_dict["iter_list"])

            if plot_dict["stop_span_time"] > 0:
                plot_dict["wall_time_list"] = [wt - plot_dict["stop_span_time"] for wt in plot_dict["wall_time_list"]]
                print("acutal wall time", plot_dict["wall_time_list"])

            alpha = 0.75
            label = f'{method_name_dict[plot_dict["method"]]}{spanning_text}'
            ax_dict["wall_time"].plot(plot_dict["wall_time_list"], plot_dict["iter_list"], wall_time_line_type, label=f"{label}", color=line_color_dict[plot_dict["method"]], alpha=alpha)
            ax_dict["grad_norm"].semilogy(plot_dict["wall_time_list"], plot_dict["grad_norm_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["grad_norm_y"].semilogy(plot_dict["wall_time_list"], plot_dict["grad_norm_y_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["corr_norm_x"].semilogy(plot_dict["wall_time_list"], plot_dict["corr_norm_x_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label} {corr_norm_x_symbol}", linewidth=3, alpha=alpha)
            ax_dict["corr_rel_norm_x"].semilogy(plot_dict["wall_time_list"], plot_dict["corr_rel_norm_x_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label} {corr_rel_norm_x_symbol}", linewidth=2, alpha=alpha)
            ax_dict["l2_dist"].semilogy(plot_dict["wall_time_list"], plot_dict["params_l2_dist_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["KL"].semilogy(plot_dict["wall_time_list"], plot_dict["KL_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["mode_num"].plot(plot_dict["wall_time_list"], plot_dict["mode_num_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["wall_time2"].plot(plot_dict["iter_list"], plot_dict["wall_time_list"], wall_time_line_type, label=f"{label}", color=line_color_dict[plot_dict["method"]], alpha=alpha)
            ax_dict["grad_norm2"].semilogy(plot_dict["iter_list"], plot_dict["grad_norm_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["grad_norm_y2"].semilogy(plot_dict["iter_list"], plot_dict["grad_norm_y_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["l2_dist2"].semilogy(plot_dict["iter_list"], plot_dict["params_l2_dist_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["KL2"].semilogy(plot_dict["iter_list"], plot_dict["KL_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label} KL", linewidth=2, alpha=alpha)
            ax_dict["mode_num2"].plot(plot_dict["iter_list"], plot_dict["mode_num_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label}", linewidth=2, alpha=alpha)
            ax_dict["corr_norm_x2"].semilogy(plot_dict["iter_list"], plot_dict["corr_norm_x_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label} {corr_norm_x_symbol}", linewidth=3, alpha=alpha)
            ax_dict["corr_rel_norm_x2"].semilogy(plot_dict["iter_list"], plot_dict["corr_rel_norm_x_list"], grad_norm_line_type, color=line_color_dict[plot_dict["method"]], label=f"{label} {corr_rel_norm_x_symbol}", linewidth=2, alpha=alpha)

            x_grad_text = r"$\nabla_x f$"
            x_grad_corr_text = r"$c_x$"
            if plot_dict["method"] in ["fr4", "fr5"]:
                if plot_dict["stop_x_corr_time"] >= 0:
                    ax_dict["wall_time"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}",
                                                 color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["grad_norm"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["grad_norm_y"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}",
                                                 color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["corr_norm_x"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]],
                                               linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["corr_norm_x"].axhline(1e-2, label=f"{corr_norm_x_symbol} threshold", color=line_color_dict[plot_dict["method"]],
                                                   linestyle="-", linewidth=1, alpha=alpha)
                    ax_dict["corr_rel_norm_x"].axhline(1e-4, label=f"{corr_rel_norm_x_symbol} threshold", color=line_color_dict[plot_dict["method"]],
                                                   linestyle=":", linewidth=1, alpha=alpha)
                    ax_dict["l2_dist"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["KL"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}",
                                               color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["mode_num"].axvline(plot_dict["stop_x_corr_time"], label=f"Turn off {x_grad_corr_text} {spanning_text}",
                                                 color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                if plot_dict["stop_x_corr_iter"] >= 0:
                    ax_dict["wall_time2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["grad_norm2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["grad_norm_y2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}",
                                                  color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["corr_norm_x2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]],
                                                linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["corr_norm_x2"].axhline(1e-2, label=f"{corr_norm_x_symbol} threshold", color=line_color_dict[plot_dict["method"]],
                                                   linestyle="-", linewidth=1, alpha=alpha)
                    ax_dict["corr_rel_norm_x2"].axhline(1e-4, label=f"{corr_rel_norm_x_symbol} threshold", color=line_color_dict[plot_dict["method"]], linestyle=":", linewidth=1, alpha=alpha)
                    ax_dict["l2_dist2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["KL2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)
                    ax_dict["mode_num2"].axvline(plot_dict["stop_x_corr_iter"], label=f"Turn off {x_grad_corr_text} {spanning_text}", color=line_color_dict[plot_dict["method"]], linestyle=wall_time_line_type, linewidth=1, alpha=alpha)

            if data_type not in data_type_visit_dict:
                ax_dict["mode_num"].axhline(real_data_mode_num_dict[data_type], label=f"Ground truth", color="#000000", linestyle="-", linewidth=1, alpha=alpha)
                ax_dict["mode_num2"].axhline(real_data_mode_num_dict[data_type], label=f"Ground truth", color="#000000", linestyle="-", linewidth=1, alpha=alpha)
                data_type_visit_dict[data_type] = True

            ax_dict["wall_time"].legend(loc="lower right", fontsize="xx-small")
            ax_dict["grad_norm"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["grad_norm_y"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["corr_norm_x"].legend(loc="upper left", fontsize="xx-small")
            ax_dict["corr_rel_norm_x"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["l2_dist"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["KL"].legend(loc="upper right", fontsize="xx-small")
            ax_dict["mode_num"].legend(loc="upper right", fontsize="xx-small")
            ax_dict["wall_time2"].legend(loc="lower right", fontsize="xx-small")
            ax_dict["grad_norm2"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["grad_norm_y2"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["l2_dist2"].legend(loc="lower left", fontsize="xx-small")
            ax_dict["KL2"].legend(loc="upper right", fontsize="xx-small")
            ax_dict["mode_num2"].legend(loc="upper right", fontsize="xx-small")
            ax_dict["corr_norm_x2"].legend(loc="upper left", fontsize="xx-small")
            ax_dict["corr_rel_norm_x2"].legend(loc="lower left", fontsize="xx-small")

            ax_dict["wall_time"].set_xlim(left=0)
            ax_dict["wall_time"].set_ylim(bottom=0)
            ax_dict["grad_norm"].set_xlim(left=0)
            ax_dict["grad_norm_y"].set_xlim(left=0)
            ax_dict["corr_norm_x"].set_xlim(left=0)
            ax_dict["corr_rel_norm_x"].set_xlim(left=0)
            ax_dict["l2_dist"].set_xlim(left=0)
            ax_dict["KL"].set_xlim(left=0)
            ax_dict["mode_num"].set_xlim(left=0)
            ax_dict["wall_time2"].set_xlim(left=0)
            ax_dict["wall_time2"].set_ylim(bottom=0)
            ax_dict["grad_norm2"].set_xlim(left=0)
            ax_dict["grad_norm_y2"].set_xlim(left=0)
            ax_dict["l2_dist2"].set_xlim(left=0)
            ax_dict["KL2"].set_xlim(left=0)
            ax_dict["mode_num2"].set_xlim(left=0)
            ax_dict["corr_norm_x2"].set_xlim(left=0)
            ax_dict["corr_rel_norm_x2"].set_xlim(left=0)


        plot_dict = plot_dict_list[0]
        title = f'{plot_dict["data"]}_{plot_dict["divergence"]}_{time.strftime("_%Y%m%d_%H%M%S")}'
        plt.subplots_adjust(top=.9)
        plt.tight_layout()
        fig.suptitle(title)
        plt.savefig(os.path.join(summary_dir, task_dir + f"_{title}.svg"))


        # sess.close()
        # K.clear_session()
        # tf.reset_default_graph()
