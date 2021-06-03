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


class DeepVisuals_Summary_II(DeepVisuals_2D):
    def __init__(self, task_dir, dataset_name, cumul=False):
        super(DeepVisuals_Summary_II, self).__init__(name=task_dir, handle=-1)
        exp_data_path_list = glob.glob(os.path.join(summary_dir, task_dir, f"{dataset_name}*"))
        self.dataset_name = dataset_name
        self.cumul = cumul
        if self.cumul:
            self.attr['name'] += "_cumul"
        self.deepVisuals_2D_name_list = [exp_data_path.split("\\")[2][:-7] for exp_data_path in exp_data_path_list]
        self.deepVisuals_2D_list = []
        self.deepVisuals_classified_name_dict = {}
        self.attr["max_t"] = 0
        self.attr_seq_name_list = ["iter", "KL", "BP_G_entropy", "BP_D_entropy"]
        self.attr_seq = {}
        for attr_seq_name in self.attr_seq_name_list:
            self.attr_seq[attr_seq_name] = {}
        max_t_list = []
        for exp_data_name in self.deepVisuals_2D_name_list:
            deepVisuals_2D = DeepVisuals_2D()
            deepVisuals_2D.Load_data(exp_data_name, data_folder=os.path.join(summary_dir, task_dir))
            deepVisuals_2D.Calculate_max_t()
            max_t_list.append(deepVisuals_2D.attr["max_t"])
            self.deepVisuals_2D_list.append(deepVisuals_2D)
            print(f"Loaded exp: {exp_data_name}")
            exp_data_name_classified = Classify_experiments_name_by_keyword(exp_data_name, "opt_type")[:-13] # Get rid of timestamp
            opt_type = deepVisuals_2D.attr['args'].opt_type
            print(f"exp_data_name_classified: {exp_data_name_classified}")
            print(f"opt_type: {opt_type}")

            dataset_config = f"{deepVisuals_2D.attr['args'].data}-{deepVisuals_2D.attr['args'].mog_scale}-{deepVisuals_2D.attr['args'].mog_std}"

            if exp_data_name_classified not in self.deepVisuals_classified_name_dict:

                aux_classifier_loaded, real_data_prob, real_mode_num = Load_aux_classifier(dataset_config)
                print("real_data_prob", real_data_prob)
                self.deepVisuals_classified_name_dict[exp_data_name_classified] = {opt_type: {"deepviz": deepVisuals_2D, "aux_classifier_loaded": aux_classifier_loaded, "real_data_prob": real_data_prob, "real_mode_num": real_mode_num}}
            else:
                aux_classifier_loaded, real_data_prob, real_mode_num = Load_aux_classifier(dataset_config)
                print("real_data_prob", real_data_prob)
                print("Found matching experiment!")
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
        self.figure_nsubcol = 6
        self.figure_ncol = len(self.opt_type_list) * self.figure_nsubcol


    def Init_figure(self):
        self.ims = []
        self.ax_dict = {}
        fig_size = 5
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
                self.ax_dict[title].set_xlim(-15, 15)
                self.ax_dict[title].xaxis.set_major_locator(LinearLocator(numticks=21))
                self.ax_dict[title].set_ylim(0.9, 300)
                # self.ax_dict[title].set_title(title)

                """     BP_delta_slopes_1_G """
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 3) * grid_span), rowspan=grid_span // 3, colspan=1 * grid_span)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_xlim(-2, 2)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=11))
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_ylim(0.9, 130)
                self.ax_dict[f"BP_delta_slopes_1_G_{title}"].set_title(title, loc="right")

                """     BP_delta_slopes_2_G """
                self.ax_dict[f"BP_delta_slopes_2_G_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + grid_span // 3, (j * self.figure_nsubcol + 3) * grid_span), rowspan=grid_span // 3, colspan=1 * grid_span)

                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].set_xlim(-2, 2)
                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].xaxis.set_major_locator(LinearLocator(numticks=11))
                self.ax_dict[f"BP_delta_slopes_2_G_{title}"].set_ylim(0.9, 130)

                """     BP_delta_slopes_D """
                self.ax_dict[f"BP_delta_slopes_D_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + 2 * grid_span // 3, (j * self.figure_nsubcol + 3) * grid_span), rowspan=grid_span // 3, colspan=1 * grid_span)
                self.ax_dict[f"BP_delta_slopes_D_{title}"].set_xlim(-10, 10)
                self.ax_dict[f"BP_delta_slopes_D_{title}"].xaxis.set_major_locator(LinearLocator(numticks=11))
                self.ax_dict[f"BP_delta_slopes_D_{title}"].set_ylim(0.9, 130)


                self.ax_dict[f"data_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 4) * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
                deepviz = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["deepviz"]
                self.ax_dict[f"data_{title}"].scatter(deepviz.attr["x_real"][:, 0], deepviz.attr["x_real"][:, 1], color="#000000", linewidth=2.0, alpha=0.7)
                self.ax_dict[f"data_{title}"].axis(deepviz.attr["bbox_x"])  # set axis range by [xmin, xmax, ymin, ymax]
                self.ax_dict[f"data_{title}"].set_aspect(abs(deepviz.attr["bbox_x"][1] - deepviz.attr["bbox_x"][0]) / abs(
                    deepviz.attr["bbox_x"][3] - deepviz.attr["bbox_x"][2]))  # set axis value ratio manually to get equal length
                self.ax_dict[f"data_{title}"].set_xlabel(r"$x_1$, $G(z)_1$")
                self.ax_dict[f"data_{title}"].set_ylabel(r"$x_2$, $G(z)_2$")

                self.ax_dict[f"BP_G_density_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 2) * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
                self.ax_dict[f"BP_G_density_{title}"].axis(deepviz.attr["bbox_z"])  # set axis range by [xmin, xmax, ymin, ymax]
                self.ax_dict[f"BP_G_density_{title}"].set_aspect(abs(deepviz.attr["bbox_z"][1] - deepviz.attr["bbox_z"][0]) / abs(deepviz.attr["bbox_z"][3] - deepviz.attr["bbox_z"][2]))  # set axis value ratio manually to get equal length
                self.ax_dict[f"BP_G_density_{title}"].set_xlabel(r"$z_1$")
                self.ax_dict[f"BP_G_density_{title}"].set_ylabel(r"$z_2$")

                self.ax_dict[f"BP_G_density_{title}"].add_artist(Circle((0, 0), 3 * deepviz.attr["args"].z_std, color="#00FF00", fill=False, label=r"$3\simga_z$"))

                self.ax_dict[f"BP_D_density_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span, (j * self.figure_nsubcol + 5) * grid_span), rowspan=1 * grid_span, colspan=1 * grid_span)
                self.ax_dict[f"BP_D_density_{title}"].axis(deepviz.attr["bbox_x"])  # set axis range by [xmin, xmax, ymin, ymax]
                self.ax_dict[f"BP_D_density_{title}"].set_aspect(abs(deepviz.attr["bbox_x"][1] - deepviz.attr["bbox_x"][0]) / abs(deepviz.attr["bbox_x"][3] - deepviz.attr["bbox_x"][2]))  # set axisloading value ratio manually to get equal length
                self.ax_dict[f"BP_D_density_{title}"].set_xlabel(r"$x_1$")
                self.ax_dict[f"BP_D_density_{title}"].set_ylabel(r"$x_2$")


                self.ax_dict[f"KL_{title}"] = plt.subplot2grid((self.figure_nrow * grid_span, self.figure_ncol * grid_span), (i * grid_span + grid_span // 2, j * self.figure_nsubcol * grid_span), rowspan=grid_span // 2, colspan=2 * grid_span)
                if j == 0:
                    self.ax_dict[f"KL_{title}"].set_ylabel("KL divergence")
                # self.ax_dict[f"KL_{title}"].set_xlim(0, 180)
                self.ax_dict[f"KL_{title}"].set_xlabel(r"Iteration")
                # self.ax_dict[f"KL_{title}"].xaxis.set_major_locator(LinearLocator(numticks=13))
                self.ax_dict[f"KL_{title}"].set_ylim(0.09, 30)

                self.ax_dict[f"BP_entropy_{title}"] = self.ax_dict[f"KL_{title}"].twinx()
                if j == len(self.opt_type_list) - 1:
                    self.ax_dict[f"BP_entropy_{title}"].set_ylabel("BP entropy")


        for ax_name in self.ax_dict:
            self.ax_dict[ax_name].tick_params(direction="in", zorder=10)

        self.fig.set_tight_layout(True)
        plt.subplots_adjust(hspace=1, top=0.9)
        print("Figure intialized.")

        # plt.savefig(os.path.join(summary_dir, f"{task_dir}_updates.png"), dpi=400)

    def Plot_step(self, t, loading=False):
        """ Begin plotting """
        imgs = []

        for i, exp_data_name_classified in enumerate(self.deepVisuals_matching_name_dict):
            for j, opt_type in enumerate(self.opt_type_list):
                title = f"{opt_type}_{exp_data_name_classified}"
                # print(f"i={i} / {self.figure_nrow}, j={j} / {self.figure_ncol}, {title}")
                deepviz = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["deepviz"]
                data_container_generator = deepviz.data_container_generator
                idd = next(data_container_generator)
                state_dict_G = idd["state_dict_G"]
                state_dict_D = idd["state_dict_D"]
                # print(state_dict_G["hidden_layer.bias"])
                BP_directions_G, BP_signed_distances_G, BP_delta_slopes_G = Get_BP_params(state_dict_G["hidden_layer.weight"],
                                                                                          state_dict_G["hidden_layer.bias"],
                                                                                          state_dict_G["output_layer.weight"])
                BP_directions_D, BP_signed_distances_D, BP_delta_slopes_D = Get_BP_params(state_dict_D["hidden_layer.weight"],
                                                                                          state_dict_D["hidden_layer.bias"],
                                                                                          state_dict_D["output_layer.weight"])

                """     Calculate BP_G density """
                BP_G_radian_list = np.arctan2(BP_directions_G[:, 1], BP_directions_G[:, 0])
                BP_G_point_list = np.concatenate([[BP_signed_distances_G * np.cos(BP_G_radian_list)], [BP_signed_distances_G * np.sin(BP_G_radian_list)]]).T
                BP_G_kernel = stats.gaussian_kde(BP_G_point_list.T, bw_method=0.1)
                xx_z, yy_z = np.mgrid[deepviz.attr["bbox_z"][0]:deepviz.attr["bbox_z"][1]:50j, deepviz.attr["bbox_z"][2]:deepviz.attr["bbox_z"][3]:50j]
                positions_z = np.vstack([xx_z.ravel(), yy_z.ravel()])
                BP_G_density_surf = np.reshape(BP_G_kernel(positions_z).T, xx_z.shape)
                BP_G_density_list = Normalize(BP_G_kernel(BP_G_point_list.T).ravel())
                BP_G_entropy = Shannon_entropy(BP_G_density_list)
                # print("BP_G_entropy", BP_G_entropy)
                # print("BP_G_density_list\n", BP_G_density_list)

                """     Calculate BP_D density """
                BP_D_radian_list = np.arctan2(BP_directions_D[:, 1], BP_directions_D[:, 0])
                BP_D_point_list = np.concatenate([[BP_signed_distances_D * np.cos(BP_D_radian_list)], [BP_signed_distances_D * np.sin(BP_D_radian_list)]]).T
                BP_D_kernel = stats.gaussian_kde(BP_D_point_list.T, bw_method=0.2)
                xx_x, yy_x = np.mgrid[deepviz.attr["bbox_x"][0]:deepviz.attr["bbox_x"][1]:50j, deepviz.attr["bbox_x"][2]:deepviz.attr["bbox_x"][3]:50j]
                positions_x = np.vstack([xx_x.ravel(), yy_x.ravel()])
                BP_D_density_surf = np.reshape(BP_D_kernel(positions_x).T, xx_x.shape)
                BP_D_density_list = Normalize(BP_D_kernel(BP_D_point_list.T).ravel())
                BP_D_entropy = Shannon_entropy(BP_D_density_list)

                """     Calculate KL """
                if self.dataset_name not in ["mnist", "cifar"]:
                    aux_classifier_loaded = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["aux_classifier_loaded"]
                    real_data_prob = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["real_data_prob"]
                    with torch.no_grad():
                        pred_data_prob, covered_mode_num = Get_classification_logit_prob_class(aux_classifier_loaded, idd["x_out"])
                        KL = np.sum(real_data_prob * np.log(real_data_prob / (pred_data_prob + EPS_BUFFER)))
                    performance_text = f"KL: {KL:.4f}, # of modes: {covered_mode_num}\n"
                else:
                    print("Real dataset performance text not implemented")
                    raise NotImplementedError

                """ Storing attr_seq """
                if title not in self.attr_seq["iter"]:
                    self.attr_seq["iter"][title] = [idd["iter"]]
                    self.attr_seq["KL"][title] = [KL]
                    self.attr_seq["BP_G_entropy"][title] = [BP_G_entropy]
                    self.attr_seq["BP_D_entropy"][title] = [BP_D_entropy]
                else:
                    self.attr_seq["iter"][title].append(idd["iter"])
                    self.attr_seq["KL"][title].append(KL)
                    self.attr_seq["BP_G_entropy"][title].append(BP_G_entropy)
                    self.attr_seq["BP_D_entropy"][title].append(BP_D_entropy)

                if t == 0 or (not self.cumul):
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"] = state_dict_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"] = BP_directions_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"] = BP_signed_distances_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"] = BP_delta_slopes_G

                alpha = 0.75
                if loading:
                    """     Plot KL """
                    KL_plot, = self.ax_dict[f"KL_{title}"].semilogy(self.attr_seq["iter"][title], self.attr_seq["KL"][title], '-', linewidth=1.5, color="#000000", label=r"KL", alpha=0.7)
                    imgs.append(KL_plot)

                    """     Plot BP_G_entropy """
                    BP_G_entropy_plot, = self.ax_dict[f"BP_entropy_{title}"].semilogy(self.attr_seq["iter"][title], self.attr_seq["BP_G_entropy"][title], '-.', linewidth=1.5, color="#0000FF", label="BP_G_entropy", alpha=0.7)
                    imgs.append(BP_G_entropy_plot)

                    """     Plot BP_D_entropy """
                    BP_D_entropy_plot, = self.ax_dict[f"BP_entropy_{title}"].semilogy(self.attr_seq["iter"][title], self.attr_seq["BP_D_entropy"][title], '-.', linewidth=1.5, color="#FF00FF", label="BP_D_entropy", alpha=0.7)
                    imgs.append(BP_D_entropy_plot)

                    """     Plot text for iteration and KL """
                    info_text = self.ax_dict[title].text(0.3, 0.6, f"Iter: {idd['iter']}, {performance_text}total time: {idd['total_time']:.3f}, loss_G: {idd['loss_G_tot']:.4f}, loss_D: {idd['loss_D_tot']:.4f}", {"ha": "center", "va": "center"}, horizontalalignment="left", verticalalignment="top", transform=self.ax_dict[title].transAxes, fontsize=13)
                    imgs.append(info_text)

                    """     Plot x_fake and x_real """
                    try:
                        """ Compute density for G(z) """
                        kernel = stats.gaussian_kde(idd['x_out'].T)
                        # print("idd x_out\n", idd['x_out'])
                        xx_x, yy_x = np.mgrid[deepviz.attr["bbox_x"][0]:deepviz.attr["bbox_x"][1]:50j, deepviz.attr["bbox_x"][2]:deepviz.attr["bbox_x"][3]:50j]
                        positions_x = np.vstack([xx_x.ravel(), yy_x.ravel()])
                        G_z_density_surf = np.reshape(kernel(positions_x).T, xx_x.shape)

                        """ Contour for G density @ output"""
                        cfset = self.ax_dict[f"data_{title}"].contourf(xx_x, yy_x, G_z_density_surf, cmap=self.cmap, alpha=0.8)
                        imgs.extend(cfset.collections)
                    except:
                        print("KDE error")



                    D_prob_grid = 1 / (1 + np.exp(-idd['D_output_grid']))
                    cfset_D = self.ax_dict[f"data_{title}"].contourf(deepviz.attr['xx_D'], deepviz.attr['yy_D'], D_prob_grid, alpha=0.3,
                                                           levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                           colors=["#110000", "#440000", "#770000", "#AA0000", "#DD0000", "#00FF00", "#00DD00", "#00AA00", "#007700", "#004400", "#001100"])
                    imgs.extend(cfset_D.collections)

                    x_out_scatter = self.ax_dict[f"data_{title}"].scatter(idd['x_out'][:, 0], idd['x_out'][:, 1], linewidth=1.5, alpha=0.5, c="#00FFFF")
                    imgs.append(x_out_scatter)

                    """     Contour for BP_G density"""
                    cfset_G = self.ax_dict[f"BP_G_density_{title}"].contourf(xx_z, yy_z, BP_G_density_surf, cmap="viridis", alpha=0.8)
                    imgs.extend(cfset_G.collections)

                    BP_G_scatter = self.ax_dict[f"BP_G_density_{title}"].scatter(BP_G_point_list[:, 0], BP_G_point_list[:, 1], linewidth=1.5, alpha=0.5, c="#0000FF", label="BP_G")
                    imgs.append(BP_G_scatter)

                    """     Contour for BP_D density """
                    cfset_D = self.ax_dict[f"BP_D_density_{title}"].contourf(xx_x, yy_x, BP_D_density_surf, cmap="viridis", alpha=0.8)
                    imgs.extend(cfset_D.collections)

                    BP_D_scatter = self.ax_dict[f"BP_D_density_{title}"].scatter(BP_D_point_list[:, 0], BP_D_point_list[:, 1], linewidth=1.5, alpha=0.5, c="#FF00FF", label="BP_D")
                    imgs.append(BP_D_scatter)

                    """     Plot BP_delta_slopes_G """
                    bin_seq = np.linspace(-2, 2, 41)
                    if np.sum(np.isfinite(BP_delta_slopes_G[0, :])) > 0:
                        _, _, BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_1_G_{title}"].hist(
                            BP_delta_slopes_G[0, :][np.isfinite(BP_delta_slopes_G[0, :])], color="#3389FF", alpha=alpha, label="BP_delta_slopes_G_1",
                            bins=bin_seq, log=True, edgecolor='k')
                        imgs.extend(BP_delta_slopes_G_hist)
                    if np.sum(np.isfinite(BP_delta_slopes_G[1, :])) > 0:
                        _, _, BP_delta_slopes_G_hist = self.ax_dict[f"BP_delta_slopes_2_G_{title}"].hist(
                            BP_delta_slopes_G[1, :][np.isfinite(BP_delta_slopes_G[1, :])], color="#3389FF", alpha=alpha, label="BP_delta_slopes_G_2", bins=bin_seq, log=True, edgecolor='k', hatch="o")
                        imgs.extend(BP_delta_slopes_G_hist)

                    """     Plot BP_delta_slopes_D """
                    bin_seq = np.linspace(-10, 10, 41)
                    if np.sum(np.isfinite(BP_delta_slopes_D)) > 0:
                        _, _, BP_delta_slopes_D_hist = self.ax_dict[f"BP_delta_slopes_D_{title}"].hist(BP_delta_slopes_D[np.isfinite(BP_delta_slopes_D)], color="#FF57FF", alpha=alpha, label="BP_delta_slopes_D", bins=bin_seq, log=True, edgecolor='k')
                        imgs.extend(BP_delta_slopes_D_hist)


                    """ Plot updates """
                    if "state_dict_G_prev" in self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]:
                        """ Calculate updates """
                        state_dict_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"]
                        BP_directions_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"]
                        BP_signed_distances_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"]
                        BP_delta_slopes_G_prev = self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"]
                        update_G_dict = {}
                        for parameter in state_dict_G:
                            update_G_dict[parameter] = state_dict_G[parameter] - state_dict_G_prev[parameter]
                        """     Plot param_G updates """
                        bin_seq = np.linspace(-15, 15, 51)
                        _, _, state_dict_G_bias_hist = self.ax_dict[title].hist(state_dict_G["hidden_layer.bias"].cpu().numpy().ravel(), color="#173F5F", alpha=alpha, label="hidden_layer.bias", bins=bin_seq, log=True, edgecolor='k', rwidth=0.8)
                        imgs.extend(state_dict_G_bias_hist)

                        _, _, state_dict_G_weight_hist = self.ax_dict[title].hist(state_dict_G["hidden_layer.weight"].cpu().numpy().ravel(), color="#DD4949", alpha=alpha, label="hidden_layer.weight", bins=bin_seq, log=True, edgecolor='k', rwidth=0.6)
                        imgs.extend(state_dict_G_weight_hist)

                        _, _, state_dict_G_vweight_hist = self.ax_dict[title].hist(state_dict_G["output_layer.weight"].cpu().numpy().ravel(), color="#00BB00", alpha=alpha, label="output_layer.weight", bins=bin_seq, log=True, edgecolor='k', rwidth=0.4)
                        imgs.extend(state_dict_G_vweight_hist)

                        if (not self.legend_drawn) and t != 1:
                            handles_list = []
                            labels_list = []
                            for ax_name in [title, f"BP_delta_slopes_1_G_{title}", f"BP_delta_slopes_2_G_{title}", f"BP_delta_slopes_D_{title}"]:
                                handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                                handles_list.extend(handles)
                                labels_list.extend(labels)

                            self.ax_dict[title].legend(handles_list, labels_list, loc="upper right")

                            self.ax_dict[f"BP_G_density_{title}"].legend(loc="upper right")
                            self.ax_dict[f"BP_D_density_{title}"].legend(loc="upper right")

                            handles_list = []
                            labels_list = []
                            for ax_name in [f"KL_{title}", f"BP_entropy_{title}"]:
                                handles, labels = self.ax_dict[ax_name].get_legend_handles_labels()
                                handles_list.extend(handles)
                                labels_list.extend(labels)
                            self.ax_dict[f"KL_{title}"].legend(handles_list, labels_list, loc="upper right")

                if not self.cumul:
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["state_dict_G_prev"] = state_dict_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_directions_G_prev"] = BP_directions_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_signed_distances_G_prev"] = BP_signed_distances_G
                    self.deepVisuals_matching_name_dict[exp_data_name_classified][opt_type]["BP_delta_slopes_G_prev"] = BP_delta_slopes_G

        if loading:
            self.ims.append(tuple(imgs))
            # print(f"len(self.ims) {len(self.ims)}")
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



if __name__ == "__main__":
    timer = Timer()
    task_dir = "random9-6_1_simgd_fr_freeze_w_seed0"
    task_dir = "x_scale_freeze_w2"
    task_dir = "grid5_simgd_fr_alpha_seed0"
    task_dir = "grid5_simgd_alpha_seed0"

    dataset_name = "random9-6_1"
    dataset_name = "grid5"
    deepVisuals_Summary = DeepVisuals_Summary_II(task_dir, dataset_name, cumul=True)
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
