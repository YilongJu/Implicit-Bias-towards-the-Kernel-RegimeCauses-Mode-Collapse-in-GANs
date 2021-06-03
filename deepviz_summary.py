from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from deepviz import *
from Run_Aux_Training import aux_model_folder
from Run_Aux_Training import Get_classification_logit_prob_class
from Run_Aux_Training import Load_aux_classifier
from Synthetic_Dataset import Synthetic_Dataset
from models import MLP

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

summary_dir = "Summaries"



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


class DeepVisuals_Summary(DeepVisuals_2D):
    def __init__(self, task_dir, dataset_name, cumul=False):
        super(DeepVisuals_Summary, self).__init__(name=task_dir, handle=-1)
        exp_data_path_list = glob.glob(os.path.join(summary_dir, task_dir, f"{dataset_name}*"))
        self.dataset_name = dataset_name
        self.cumul = cumul
        if self.cumul:
            self.attr['name'] += "_cumul"
        self.deepVisuals_2D_name_list = [exp_data_path.split("\\")[2][:-7] for exp_data_path in exp_data_path_list]
        self.deepVisuals_2D_list = []
        self.deepVisuals_classified_name_dict = {}
        self.attr["max_t"] = 0
        max_t_list = []
        counter = 0
        for exp_data_name in self.deepVisuals_2D_name_list:
            counter += 1
            deepVisuals_2D = DeepVisuals_2D()
            deepVisuals_2D.Load_data(exp_data_name, data_folder=os.path.join(summary_dir, task_dir))
            deepVisuals_2D.Calculate_max_t()
            max_t_list.append(deepVisuals_2D.attr["max_t"])
            self.deepVisuals_2D_list.append(deepVisuals_2D)
            print(f"Loaded exp: {exp_data_name}")
            exp_data_name_classified = Classify_experiments_name_by_keyword(exp_data_name, "opt_type")[:-13] # Get rid of timestamp


            opt_type = deepVisuals_2D.attr['args'].opt_type
            if counter == 2:
                opt_type = "rmsprop"
            print(f"exp_data_name_classified: {exp_data_name_classified}")
            print(f"opt_type: {opt_type}")

            dataset_config = f"{deepVisuals_2D.attr['args'].data}-{deepVisuals_2D.attr['args'].mog_scale}-{deepVisuals_2D.attr['args'].mog_std}"

            if exp_data_name_classified not in self.deepVisuals_classified_name_dict:
                if self.dataset_name not in ["mnist", "cifar"]:
                    aux_classifier_loaded, real_data_prob, real_mode_num = Load_aux_classifier(dataset_config)
                    print("real_data_prob", real_data_prob)
                else:
                    aux_classifier_loaded = None
                    real_data_prob = None
                    real_mode_num = None
                self.deepVisuals_classified_name_dict[exp_data_name_classified] = {opt_type: {"deepviz": deepVisuals_2D, "aux_classifier_loaded": aux_classifier_loaded, "real_data_prob": real_data_prob, "real_mode_num": real_mode_num}}
            else:
                if self.dataset_name not in ["mnist", "cifar"]:
                    aux_classifier_loaded, real_data_prob, real_mode_num = Load_aux_classifier(dataset_config)
                    print("real_data_prob", real_data_prob)
                    print("Found matching experiment!")
                else:
                    aux_classifier_loaded = None
                    real_data_prob = None
                    real_mode_num = None

                self.deepVisuals_classified_name_dict[exp_data_name_classified][opt_type] = {"deepviz": deepVisuals_2D, "aux_classifier_loaded": aux_classifier_loaded, "real_data_prob": real_data_prob, "real_mode_num": real_mode_num}

        self.attr["max_t"] = np.max(max_t_list).astype(int)

        """ Only keep matching experiments """
        self.deepVisuals_matching_name_dict = {}
        self.opt_type_list = ["sgd", "rmsprop"]

        for exp_data_name_classified in self.deepVisuals_classified_name_dict:
            if len(self.deepVisuals_classified_name_dict[exp_data_name_classified]) != len(self.opt_type_list):
                print(f"Dropped incomplete exp comparison: {exp_data_name_classified}")
            else:
                self.deepVisuals_matching_name_dict[exp_data_name_classified] = self.deepVisuals_classified_name_dict[exp_data_name_classified]

        self.figure_nrow = len(self.deepVisuals_matching_name_dict)
        self.figure_nsubcol = 4
        self.figure_ncol = len(self.opt_type_list) * self.figure_nsubcol


    def Init_figure(self):
        self.ims = []
        self.ax_dict = {}
        fig_size = 7
        grid_span = 6
        span_figure_r = 3
        span_figure_c = 2
        width_to_height = 1

        self.fig = pl.figure(figsize=(width_to_height * self.figure_ncol * fig_size, self.figure_nrow * fig_size))
        for i, exp_data_name_classified in enumerate(self.deepVisuals_matching_name_dict):
            for j, opt_type in enumerate(self.opt_type_list):
                title = f"{opt_type}_{exp_data_name_classified}"
                print(f"i={i} / {self.figure_nrow}, j={j} / {self.figure_ncol}, {title}")
                self.ax_dict[title] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, j * self.figure_nsubcol * grid_span), rowspan=grid_span // 2, colspan=2 * grid_span)
                if j == 0:
                    self.ax_dict[title].set_ylabel("Counts")
                self.ax_dict[title].set_xlabel("Parameter updates")
                self.ax_dict[title].set_xlim(0, 1)
                self.ax_dict[title].xaxis.set_major_locator(LinearLocator(numticks=21))
                self.ax_dict[title].set_ylim(0.9, 300)
                # self.ax_dict[title].set_title(title)

                self.ax_dict[f"{title}_zoom_in"] = zoomed_inset_axes(self.ax_dict[f"{title}"], 15, loc="center", bbox_to_anchor=[0.2, 0.4, 0.3, 0.6], bbox_transform=self.ax_dict[f"{title}"].transAxes)
                self.ax_dict[f"{title}_zoom_in"].set_xlim(0, 0.02)
                self.ax_dict[f"{title}_zoom_in"].xaxis.set_major_locator(LinearLocator(numticks=5))
                self.ax_dict[f"{title}_zoom_in"].set_ylim(0.9, 300)
                mark_inset(self.ax_dict[f"{title}"], self.ax_dict[f"{title}_zoom_in"], loc1=1, loc2=4, fc="none", ec="0.5")
                self.ax_dict[f"{title}_zoom_in"].set_aspect(0.003)


                self.ax_dict[f"BP_directions_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + grid_span // 2, j * self.figure_nsubcol * grid_span), rowspan=grid_span // 2, colspan=1 * grid_span)
                if j == 0:
                    self.ax_dict[f"BP_directions_G_{title}"].set_ylabel("Counts")
                self.ax_dict[f"BP_directions_G_{title}"].set_xlim(0, 180)
                self.ax_dict[f"BP_directions_G_{title}"].set_xlabel(r"BP direction update $^\circ$")
                self.ax_dict[f"BP_directions_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=13))
                self.ax_dict[f"BP_directions_G_{title}"].set_ylim(0.9, 130)

                self.ax_dict[f"BP_directions_G_{title}_zoom_in"] = zoomed_inset_axes(self.ax_dict[f"BP_directions_G_{title}"], 18, loc="center", bbox_to_anchor=[0.4, 0.4, 0.6, 0.6], bbox_transform=self.ax_dict[f"BP_directions_G_{title}"].transAxes)
                self.ax_dict[f"BP_directions_G_{title}_zoom_in"].set_xlim(0, 5)
                self.ax_dict[f"BP_directions_G_{title}_zoom_in"].xaxis.set_major_locator(LinearLocator(numticks=6))
                self.ax_dict[f"BP_directions_G_{title}_zoom_in"].set_ylim(0.9, 130)
                mark_inset(self.ax_dict[f"BP_directions_G_{title}"], self.ax_dict[f"BP_directions_G_{title}_zoom_in"], loc1=1, loc2=4, fc="none", ec="0.5")
                self.ax_dict[f"BP_directions_G_{title}_zoom_in"].set_aspect(1)


                self.ax_dict[f"BP_signed_distances_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + grid_span // 2, (j * self.figure_nsubcol + 1) * grid_span), rowspan=grid_span // 2, colspan=1 * grid_span)
                self.ax_dict[f"BP_signed_distances_G_{title}"].set_xlim(-10, 60)
                self.ax_dict[f"BP_signed_distances_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=15))
                self.ax_dict[f"BP_signed_distances_G_{title}"].set_ylim(0.9, 130)

                self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"] = zoomed_inset_axes(self.ax_dict[f"BP_signed_distances_G_{title}"], 17.5, loc="center", bbox_to_anchor=[0.4, 0.4, 0.6, 0.6], bbox_transform=self.ax_dict[f"BP_signed_distances_G_{title}"].transAxes)
                self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"].set_xlim(-1, 1)
                self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"].xaxis.set_major_locator(LinearLocator(numticks=5))
                self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"].set_ylim(0.9, 130)
                mark_inset(self.ax_dict[f"BP_signed_distances_G_{title}"], self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"], loc1=1, loc2=4, fc="none", ec="0.5")
                self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"].set_aspect(0.43)



                self.ax_dict[f"BP_delta_slopes_1_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 2) * grid_span), rowspan=grid_span // 2, colspan=1 * grid_span)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_xlim(-2, 2)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=11))
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_ylim(0.9, 130)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_title(title, loc="right")


                self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"] = zoomed_inset_axes(self.ax_dict[f"BP_delta_slopes_1_G_{title}"], 3, loc="center", bbox_to_anchor=[0.6, 0.6, 0.4, 0.4], bbox_transform=self.ax_dict[f"BP_delta_slopes_1_G_{title}"].transAxes)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"].set_xlim(-0.2, 0.2)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"].xaxis.set_major_locator(LinearLocator(numticks=5))
                self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"].set_ylim(0.9, 130)
                mark_inset(self.ax_dict[f"BP_delta_slopes_1_G_{title}"], self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"], loc1=1, loc2=4, fc="none", ec="0.5")
                self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"].set_aspect(0.1)


                self.ax_dict[f"BP_delta_slopes_2_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + grid_span // 2, (j * self.figure_nsubcol + 2) * grid_span), rowspan=grid_span // 2, colspan=1 * grid_span)

                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].set_xlim(-2, 2)
                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=11))
                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].set_ylim(0.9, 130)

                self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"] = zoomed_inset_axes(self.ax_dict[f"BP_delta_slopes_2_G_{title}"], 3, loc="center", bbox_to_anchor=[0.6, 0.6, 0.4, 0.4],
                                                                                     bbox_transform=self.ax_dict[f"BP_delta_slopes_2_G_{title}"].transAxes)
                self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"].set_xlim(-0.2, 0.2)
                self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"].xaxis.set_major_locator(LinearLocator(numticks=5))
                self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"].set_ylim(0.9, 130)
                mark_inset(self.ax_dict[f"BP_delta_slopes_2_G_{title}"], self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"], loc1=1, loc2=4, fc="none", ec="0.5")
                self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"].set_aspect(0.1)

                if self.dataset_name not in ["mnist", "cifar"]:
                    self.ax_dict[f"data_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 3) * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
                    deepviz = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["deepviz"]
                    self.ax_dict[f"data_{title}"].scatter(deepviz.attr["x_real"][:, 0], deepviz.attr["x_real"][:, 1], color="#000000", linewidth=2.0, alpha=0.7)
                    self.ax_dict[f"data_{title}"].axis(deepviz.attr["bbox_x"])  # set axis range by [xmin, xmax, ymin, ymax]
                    self.ax_dict[f"data_{title}"].set_aspect(abs(deepviz.attr["bbox_x"][1] - deepviz.attr["bbox_x"][0]) / abs(
                        deepviz.attr["bbox_x"][3] - deepviz.attr["bbox_x"][2]))  # set axis value ratio manually to get equal length
                    self.ax_dict[f"data_{title}"].set_xlabel(r"$x_1$, $G(z)_1$")
                    self.ax_dict[f"data_{title}"].set_ylabel(r"$x_2$, $G(z)_2$")


        self.fig.set_tight_layout(True)
        plt.subplots_adjust(hspace=1, top=0.9)
        print("Figure intialized.")

        # plt.savefig(os.path.join(summary_dir, f"{task_dir}_updates.png"), dpi=400)

    def Plot_step(self, t, loading=False):
        toc = time.time()
        line_animated = True
        imgs = []

        for i, exp_data_name_classified in enumerate(self.deepVisuals_matching_name_dict):
            for j, opt_type in enumerate(self.opt_type_list):
                title = f"{opt_type}_{exp_data_name_classified}"
                # print(f"i={i} / {self.figure_nrow}, j={j} / {self.figure_ncol}, {title}")
                deepviz = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["deepviz"]
                data_container_generator = deepviz.data_container_generator
                idd = next(data_container_generator)
                state_dict_G = idd["state_dict_G"]
                # print(state_dict_G["hidden_layer.bias"])
                BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"],
                                                                                          state_dict_G["hidden_layer.bias"],
                                                                                          state_dict_G["output_layer.weight"])


                """     Calculate KL """
                if self.dataset_name not in ["mnist", "cifar"]:
                    aux_classifier_loaded = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["aux_classifier_loaded"]
                    real_data_prob = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["real_data_prob"]
                    with torch.no_grad():
                        pred_data_prob, covered_mode_num = Get_classification_logit_prob_class(aux_classifier_loaded, idd["x_out"])
                        KL = np.sum(real_data_prob * np.log(real_data_prob / (pred_data_prob + EPS_BUFFER)))

                    performance_text = f"KL: {KL:.4f}\n# of modes: {covered_mode_num}\n"
                else:
                    pm_symbol = r"$\pm$"
                    performance_text = f"IS: {idd['inception_score'][0]:.4f} {pm_symbol} {idd['inception_score'][1]:.4f}\n"

                """     Plot text for iteration and KL """
                info_text = self.ax_dict[title].text(0.6, 0.5, f"Iter: {idd['iter']}\n{performance_text}total time: {idd['total_time']:.3f}\nloss_G: {idd['loss_G_tot']:.4f}\nloss_D: {idd['loss_D_tot']:.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict[title].transAxes, fontsize=13)
                imgs.append(info_text)

                """     Plot x_fake and x_real """
                if self.dataset_name not in ["mnist", "cifar"]:
                    try:
                        """ Compute density for G(z) """
                        kernel = stats.gaussian_kde(idd['x_out'].T)
                        xx_x, yy_x = np.mgrid[deepviz.attr["bbox_x"][0]:deepviz.attr["bbox_x"][1]:50j, deepviz.attr["bbox_x"][2]:deepviz.attr["bbox_x"][3]:50j]
                        positions_z = np.vstack([xx_x.ravel(), yy_x.ravel()])
                        G_z_density_surf = np.reshape(kernel(positions_z).T, xx_x.shape)

                        """ Contour for G density @ output"""
                        cfset = self.ax_dict[f"data_{title}"].contourf(xx_x, yy_x, G_z_density_surf, cmap=self.cmap, alpha=0.8)
                        imgs.extend(cfset.collections)
                    except:
                        print("KDE error")

                    x_out_scatter = self.ax_dict[f"data_{title}"].scatter(idd['x_out'][:, 0], idd['x_out'][:, 1], linewidth=1.5, alpha=0.5, c="#00FFFF")
                    imgs.append(x_out_scatter)


                if "state_dict_G_prev" not in self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]:
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"] = state_dict_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"] = BP_directions_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"] = BP_signed_distances_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"] = BP_delta_slopes_G
                else:
                    """ Calculate updates """
                    state_dict_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"]
                    BP_directions_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"]
                    BP_signed_distances_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"]
                    BP_delta_slopes_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"]
                    update_G_dict = {}
                    for parameter in state_dict_G:
                        update_G_dict[parameter] = state_dict_G[parameter] - state_dict_G_prev[parameter]
                    # print(update_G_dict)
                    # print(update_G_dict["hidden_layer.bias"])
                    """ Plot updates """
                    """     Plot param_G updates """
                    alpha = 0.75
                    bin_seq = np.linspace(0, 1, 51)
                    _, _, update_G_dict_bias_hist = self.ax_dict[title].hist(update_G_dict["hidden_layer.bias"].cpu().abs().numpy().ravel(), color="#173F5F", alpha=alpha, label="hidden_layer.bias", bins=bin_seq, log=True, edgecolor='k', rwidth=0.8)
                    imgs.extend(update_G_dict_bias_hist)

                    _, _, update_G_dict_weight_hist = self.ax_dict[title].hist(update_G_dict["hidden_layer.weight"].cpu().abs().numpy().ravel(), color="#DD4949", alpha=alpha, label="hidden_layer.weight", bins=bin_seq, log=True, edgecolor='k', rwidth=0.6)
                    imgs.extend(update_G_dict_weight_hist)

                    _, _, update_G_dict_vweight_hist = self.ax_dict[title].hist(update_G_dict["output_layer.weight"].cpu().abs().numpy().ravel(), color="#00BB00", alpha=alpha, label="output_layer.weight", bins=bin_seq, log=True, edgecolor='k', rwidth=0.4)
                    imgs.extend(update_G_dict_vweight_hist)


                    bin_seq_finer = np.linspace(0, 0.02, 21)
                    _, _, update_G_dict_bias_hist = self.ax_dict[f"{title}_zoom_in"].hist(update_G_dict["hidden_layer.bias"].cpu().abs().numpy().ravel(), color="#173F5F", alpha=alpha, label="hidden_layer.bias", bins=bin_seq_finer, log=True, edgecolor='k', rwidth=0.8)
                    imgs.extend(update_G_dict_bias_hist)

                    _, _, update_G_dict_weight_hist = self.ax_dict[f"{title}_zoom_in"].hist(update_G_dict["hidden_layer.weight"].cpu().abs().numpy().ravel(), color="#DD4949", alpha=alpha, label="hidden_layer.weight", bins=bin_seq_finer, log=True, edgecolor='k', rwidth=0.6)
                    imgs.extend(update_G_dict_weight_hist)

                    _, _, update_G_dict_vweight_hist = self.ax_dict[f"{title}_zoom_in"].hist(update_G_dict["output_layer.weight"].cpu().abs().numpy().ravel(), color="#00BB00", alpha=alpha, label="output_layer.weight", bins=bin_seq_finer, log=True, edgecolor='k', rwidth=0.4)
                    imgs.extend(update_G_dict_vweight_hist)

                    """     Plot BP_directions_G updates """
                    bin_seq_angles = np.linspace(0, 180, 37)
                    update_angle_BP_directions_G = Rad_to_Deg(np.arccos(Cossim(BP_directions_G, BP_directions_G_prev, output_vec=True)))
                    _, _, update_angle_BP_directions_G_hist = self.ax_dict[f"BP_directions_G_{title}"].hist(update_angle_BP_directions_G, color="#6B3074", alpha=alpha, label="BP_directions_G", bins=bin_seq_angles, log=True, edgecolor='k')
                    imgs.extend(update_angle_BP_directions_G_hist)

                    bin_seq_angles_finer = np.linspace(0, 5, 26)
                    _, _, update_angle_BP_directions_G_hist = self.ax_dict[f"BP_directions_G_{title}_zoom_in"].hist(update_angle_BP_directions_G, color="#6B3074", alpha=alpha, label="BP_directions_G", bins=bin_seq_angles_finer, log=True, edgecolor='k')
                    imgs.extend(update_angle_BP_directions_G_hist)

                    """     Plot BP_signed_distances_G updates """
                    bin_seq = np.linspace(-10, 60, 41)
                    _, _, update_BP_distances_G_hist = self.ax_dict[f"BP_signed_distances_G_{title}"].hist(np.abs(BP_signed_distances_G) - np.abs(BP_signed_distances_G_prev), color="#00FFFF", alpha=alpha, label="BP_signed_distances_G", bins=bin_seq, log=True, edgecolor='k')
                    imgs.extend(update_BP_distances_G_hist)

                    bin_seq_finer = np.linspace(-1, 1, 21)
                    _, _, update_BP_distances_G_hist = self.ax_dict[f"BP_signed_distances_G_{title}_zoom_in"].hist(np.abs(BP_signed_distances_G) - np.abs(BP_signed_distances_G_prev), color="#00FFFF", alpha=alpha, label="BP_signed_distances_G", bins=bin_seq_finer, log=True, edgecolor='k')
                    imgs.extend(update_BP_distances_G_hist)

                    """     Plot BP_delta_slopes_G updates """
                    bin_seq = np.linspace(-2, 2, 41)
                    # print("np.isfinite(BP_delta_slopes_G[0, :])", np.isfinite(BP_delta_slopes_G[0, :]))
                    update_BP_delta_slopes_G_1 = BP_delta_slopes_G[0, :] - BP_delta_slopes_G_prev[0, :]
                    update_BP_delta_slopes_G_2 = BP_delta_slopes_G[1, :] - BP_delta_slopes_G_prev[1, :]

                    if np.sum(np.isfinite(update_BP_delta_slopes_G_1)) > 0:
                        _, _, update_BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_1_G_{title}"].hist(update_BP_delta_slopes_G_1[np.isfinite(update_BP_delta_slopes_G_1)], color="#FF5733", alpha=alpha, label="BP_delta_slopes_G_1", bins=bin_seq, log=True, edgecolor='k')
                        imgs.extend(update_BP_delta_slopes_G_hist)
                    if np.sum(np.isfinite(update_BP_delta_slopes_G_2)) > 0:
                        _, _, update_BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_2_G_{title}"].hist(update_BP_delta_slopes_G_2[np.isfinite(update_BP_delta_slopes_G_2)], color="#FF5733", alpha=alpha, label="BP_delta_slopes_G_2", bins=bin_seq, log=True, edgecolor='k', hatch="o")
                        imgs.extend(update_BP_delta_slopes_G_hist)

                    bin_seq_finer = np.linspace(-0.2, 0.2, 21)
                    if np.sum(np.isfinite(update_BP_delta_slopes_G_1)) > 0:
                        _, _, update_BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_1_G_{title}_zoom_in"].hist(update_BP_delta_slopes_G_1[np.isfinite(update_BP_delta_slopes_G_1)], color="#FF5733", alpha=alpha, label="BP_delta_slopes_G_1", bins=bin_seq_finer, log=True, edgecolor='k')
                        imgs.extend(update_BP_delta_slopes_G_hist)
                    if np.sum(np.isfinite(update_BP_delta_slopes_G_2)) > 0:
                        _, _, update_BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_2_G_{title}_zoom_in"].hist(update_BP_delta_slopes_G_2[np.isfinite(update_BP_delta_slopes_G_2)], color="#FF5733", alpha=alpha, label="BP_delta_slopes_G_2", bins=bin_seq_finer, log=True, edgecolor='k', hatch="o")
                        imgs.extend(update_BP_delta_slopes_G_hist)

                    if not self.legend_drawn:
                        handles_list = []
                        labels_list = []
                        for ax_name in [title, f"BP_directions_G_{title}", f"BP_signed_distances_G_{title}", f"BP_delta_slopes_1_G_{title}", f"BP_delta_slopes_2_G_{title}"]:
                            handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                            handles_list.extend(handles)
                            labels_list.extend(labels)

                        self.ax_dict[title].legend(handles_list, labels_list, loc="upper right")
                        # handles, labels = self.ax_dict[title].get_legend_handles_labels()
                        # self.fig.legend(handles, labels, loc='upper center')

                    if not self.cumul:
                        self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"] = state_dict_G
                        self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"] = BP_directions_G
                        self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"] = BP_signed_distances_G
                        self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"] = BP_delta_slopes_G


        self.ims.append(tuple(imgs))
        if (not self.legend_drawn) and t > 0:
            self.legend_drawn = True

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
            if t >= start_pos and t < end_pos:
                if t % skip_frame != 0:
                    continue

                if t % (self.attr["max_t"] // 5) == 0:
                    print(f't {t}, max_t {self.attr["max_t"]}')

                self.Plot_step(t, loading=True)
                self.total_frame += 1
            else:
                try:
                    for i, exp_data_name_classified in enumerate(self.deepVisuals_matching_name_dict):
                        for j, opt_type in enumerate(self.opt_type_list):
                            data_container_generator = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["deepviz"].data_container_generator
                            idd = next(data_container_generator)
                            state_dict_G = idd["state_dict_G"]
                            # print(state_dict_G["hidden_layer.bias"])
                            BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"], state_dict_G["hidden_layer.bias"], state_dict_G["output_layer.weight"])
                            if t == 0 or (not self.cumul):
                                self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"] = state_dict_G
                                self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"] = BP_directions_G
                                self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"] = BP_signed_distances_G
                                self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"] = BP_delta_slopes_G

                except:
                    print("file end")
                    return

        print(f"Video production time: {time.time() - generating_start_time}")



if __name__ == "__main__":
    timer = Timer()
    task_dir = "random9-6_1_simgd_fr_freeze_w_seed0"
    task_dir = "x_scale_freeze_w2"
    task_dir = "grid5_simgd_fr_alpha_seed0"

    dataset_name = "random9-6_1"
    dataset_name = "grid5"
    deepVisuals_Summary = DeepVisuals_Summary(task_dir, dataset_name, cumul=True)
    print(deepVisuals_Summary.figure_nrow)
    print(deepVisuals_Summary.deepVisuals_classified_name_dict)
    print(len(deepVisuals_Summary.deepVisuals_classified_name_dict))
    print(deepVisuals_Summary.deepVisuals_matching_name_dict)
    print(len(deepVisuals_Summary.deepVisuals_matching_name_dict))
    deepVisuals_Summary.Init_figure()
    max_t = deepVisuals_Summary.attr["max_t"]
    max_t = 3
    # max_t = 15
    for t in range(max_t):
        print(f"Plotting frame {t + 1} / {max_t}")
        deepVisuals_Summary.Plot_step(t, loading=True)
        deepVisuals_Summary.total_frame += 1

    deepVisuals_Summary.Save_plot(figures_folder=summary_dir)
    timer.Print()
