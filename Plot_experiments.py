import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.ticker import MaxNLocator, LogLocator
import os
import seaborn as sns
sns.set()

summary_dir = "Summaries"
group_by_name_list = ['iter', 'data', 'mog_scale', 'mog_std', 'plot_lim_x', 'method', 'opt_type', 'use_spectral_norm', 'z_dim',
                      'iteration', 'batch_size', 'alpha_mobility', 'alpha_mobility_D', 'g_layers', 'd_layers', 'g_hidden',
                      'd_hidden', 'g_lr', 'd_lr', 'gamma', 'zeta']

# grad_norm_var_list = ['Mode_2_to_3', 'Mode_3_to_4', 'Mode_2_to_5', 'Mode_4_to_5', 'prop_neg_samples']
BP_density_range_list = ["BP_density_range", "BP_density_range_dim_1", "BP_density_range_dim_2", "delta_slope_inner_prod"]
BP_entropy_list = ["BP_G_entropy", "BP_G_1_entropy", "BP_G_2_entropy", "BP_D_entropy", "affine_BP_G_prop"]
performance_metric_list = ["KL", "KL_mode", "covered_mode_num"]
grad_norm_metric_list = ['dG_dz_mean', 'dD_dx_mean', 'prop_neg_samples']
grad_norm_interpolated_metric_list = ['mean_of_dG_dz_interpolated_max', 'mean_of_dD_dx_interpolated_max', 'proportion_outliers_interpolated']
opt_metric_list = ["loss_G", "loss_D", "L_G", "L_D"]
params_metric_list = ["rel_act_diff_G", "rel_act_diff_D",
                      "ntk_G_1_reldiff", "ntk_G_2_reldiff", "ntk_D_reldiff",
                      'update_params_G_norm_all',
                      'update_params_G_tot_norm_mean',
                      'update_params_G_tot_norm_std',
                      'update_params_D_norm_all',
                      'update_params_D_tot_norm_mean',
                      'update_params_D_tot_norm_std',
                      'params_G_norm_init',
                      'params_G_norm_final',
                      'params_D_norm_init',
                      'params_D_norm_final']
fc_2_layer_NN_params_list = ["weights_G_norm_init", "biases_G_norm_init", "vweights_G_norm_init",
                             "weights_D_norm_init", "biases_D_norm_init", "vweights_D_norm_init",
                             "weights_G_norm_final", "biases_G_norm_final", "vweights_G_norm_final",
                             "weights_D_norm_final", "biases_D_norm_final", "vweights_D_norm_final"]

BP_metric_list = ['update_angle_BP_directions_G_tot_deg',
                  'update_BP_distances_G_tot_norm',
                  'update_BP_delta_slopes_G_tot_norm',
                  'update_angle_BP_directions_D_tot_deg',
                  'update_BP_distances_D_tot_norm',
                  'update_BP_delta_slopes_D_tot_norm',
                  'update_angle_BP_directions_G_tot_deg_mean',
                  'update_BP_distances_G_tot_norm_mean',
                  'update_BP_delta_slopes_G_tot_norm_mean',
                  'update_angle_BP_directions_D_tot_deg_mean',
                  'update_BP_distances_D_tot_norm_mean',
                  'update_BP_delta_slopes_D_tot_norm_mean',
                  'update_angle_BP_directions_G_tot_deg_std',
                  'update_BP_distances_G_tot_norm_std',
                  'update_BP_delta_slopes_G_tot_norm_std',
                  'update_angle_BP_directions_D_tot_deg_std',
                  'update_BP_distances_D_tot_norm_std',
                  'update_BP_delta_slopes_D_tot_norm_std',
                  'update_weights_G_tot_norm_mean',
                  'update_biases_G_tot_norm_mean',
                  'update_vweights_G_tot_norm_mean',
                  'update_weights_D_tot_norm_mean',
                  'update_biases_D_tot_norm_mean',
                  'update_vweights_D_tot_norm_mean',
                  'update_weights_G_tot_norm_std',
                  'update_biases_G_tot_norm_std',
                  'update_vweights_G_tot_norm_std',
                  'update_weights_D_tot_norm_std',
                  'update_biases_D_tot_norm_std',
                  'update_vweights_D_tot_norm_std',
                  'update_weights_G_norm_all',
                  'update_biases_G_norm_all',
                  'update_vweights_G_norm_all',
                  'update_weights_D_norm_all',
                  'update_biases_D_norm_all',
                  'update_vweights_D_norm_all',
                  'BP_distances_G_mean_init',
                  'BP_distances_D_mean_init',
                  'BP_distances_G_mean_final',
                  'BP_distances_D_mean_final']

BP_Lp_norm_list = np.linspace(1, 2, 11)


def Float_to_str(p):
    if p == 1.0 or p == 2.0:
        return str(int(p))
    else:
        return f"1_{np.round(10 * (p - 1), 0):.0f}"


BP_delta_slopes_G_Lp_norm_init_list = [f"BP_delta_slopes_G_L{p:.1f}_norm_init" for p in BP_Lp_norm_list]
BP_delta_slopes_G_Lp_norm_init_list_rename = [f"BP_delta_slopes_G_L{Float_to_str(p)}_norm_init" for p in BP_Lp_norm_list]
BP_delta_slopes_D_Lp_norm_init_list = [f"BP_delta_slopes_D_L{p:.1f}_norm_init" for p in BP_Lp_norm_list]
BP_delta_slopes_D_Lp_norm_init_list_rename = [f"BP_delta_slopes_D_L{Float_to_str(p)}_norm_init" for p in BP_Lp_norm_list]
BP_delta_slopes_G_Lp_norm_final_list = [f"BP_delta_slopes_G_L{p:.1f}_norm_final" for p in BP_Lp_norm_list]
BP_delta_slopes_G_Lp_norm_final_list_rename = [f"BP_delta_slopes_G_L{Float_to_str(p)}_norm_final" for p in BP_Lp_norm_list]
BP_delta_slopes_D_Lp_norm_final_list = [f"BP_delta_slopes_D_L{p:.1f}_norm_final" for p in BP_Lp_norm_list]
BP_delta_slopes_D_Lp_norm_final_list_rename = [f"BP_delta_slopes_D_L{Float_to_str(p)}_norm_final" for p in BP_Lp_norm_list]
BP_metric_list += BP_delta_slopes_G_Lp_norm_init_list + BP_delta_slopes_D_Lp_norm_init_list + \
                  BP_delta_slopes_G_Lp_norm_final_list + BP_delta_slopes_D_Lp_norm_final_list

metrics_all = performance_metric_list + BP_density_range_list + BP_entropy_list + grad_norm_metric_list + grad_norm_interpolated_metric_list + opt_metric_list + params_metric_list + fc_2_layer_NN_params_list + BP_metric_list
rename_dict = {"alpha_mobility": r"$\alpha_G$", \
               "alpha_mobility_D": r"$\alpha_D$", \
               "g_hidden": r"$H$", "d_hidden": r"$H_D$", \
               "opt_type": "Opt alg", \
               "gamma": r"$\gamma$", \
               "zeta": r"$\zeta$", \
               "g_lr": r"$\eta$"}
name_dict = {"sgd": "SGD", "rmsprop": "RMSprop", "rmsprop-x": r"RMSprop on $G$", "rmsprop-y": r"RMSprop on $D$", \
             "log_KL": r"$\log$ KL", "log_KL_mode": r"$\log$ KL_mode",
             "grid5": "Grid", "random9-6_2": "Random", "separated": "Separated", "mnist": "MNIST-01", \
             "covered_mode_num": "# covered mode", "loss_G": "loss_G", "loss_D": "loss_D", \
             "rel_act_diff_G": r"$\Delta G$ act. pattern", "rel_act_diff_D": r"$\Delta D$ act. pattern", "L_G": r"$L_G$", "L_D": r"$L_D$", \
             "update_angle_BP_directions_G_tot_deg": r"$\sum |\Delta \xi_G|$", "update_BP_distances_G_tot_norm": r"$\sum (\Delta \gamma_G)^2$",
             "update_BP_delta_slopes_G_tot_norm": r"$\sum ||\Delta \mu_G||_2^2$", \
             "update_angle_BP_directions_D_tot_deg": r"$\sum |\Delta \xi_D|$", "update_BP_distances_D_tot_norm": r"$\sum (\Delta \gamma_D)^2$",
             "update_BP_delta_slopes_D_tot_norm": r"$\sum ||\Delta \mu_D||_2^2$", \
             "update_angle_BP_directions_G_tot_deg_mean": r"$\mathrm{Avg}[|\Delta \xi_G|]$",
             "update_BP_distances_G_tot_norm_mean": r"$\mathrm{Avg}[|\Delta \gamma_G|]$",
             "update_BP_delta_slopes_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta \mu_G||_2]$", \
             "update_angle_BP_directions_G_tot_deg_std": r"$\mathrm{Var}[\Delta \xi_G]$",
             "update_BP_distances_G_tot_norm_std": r"$\mathrm{Var}[\Delta \gamma_G]$",
             "update_BP_delta_slopes_G_tot_norm_std": r"$\mathrm{Var} [||\Delta \mu_G||_2]$", \
             "update_weights_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta w_G||_2]$", "update_biases_G_tot_norm_mean": r"$\mathrm{Avg}[|\Delta b_G|]$",
             "update_vweights_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta v_G||_{2}]$", \
             "update_weights_G_tot_norm_std": r"$\mathrm{Var}[\Delta w_G]$", "update_biases_G_tot_norm_std": r"$\mathrm{Var}[\Delta b_G]$",
             "update_vweights_G_tot_norm_std": r"$\mathrm{Var} [\Delta v_G]$", \
             "update_angle_BP_directions_D_tot_deg_mean": r"$\mathrm{Avg}[|\Delta \xi_D|]$",
             "update_BP_distances_D_tot_norm_mean": r"$\mathrm{Avg}[|\Delta \gamma_D|]$",
             "update_BP_delta_slopes_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta \mu_D||_2]$", \
             "update_angle_BP_directions_D_tot_deg_std": r"$\mathrm{Var}[\Delta \xi_D]$",
             "update_BP_distances_D_tot_norm_std": r"$\mathrm{Var}[\Delta \gamma_D]$",
             "update_BP_delta_slopes_D_tot_norm_std": r"$\mathrm{Var} [||\Delta \mu_D||_2]$", \
             "update_weights_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta w_D||_2]$", "update_biases_D_tot_norm_mean": r"$\mathrm{Avg}[|\Delta b_D|]$",
             "update_vweights_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta v_D]||_{2}]$", \
             "update_weights_D_tot_norm_std": r"$\mathrm{Var}[\Delta w_D]$", "update_biases_D_tot_norm_std": r"$\mathrm{Var}[\Delta b_D]$",
             "update_vweights_D_tot_norm_std": r"$\mathrm{Var} [\Delta v_D]$", \
             "affine_BP_G_prop": "Prop. affine BP G", 'prop_neg_samples': "Prop. neg samples", \
             "Mode_pair_1": "Mode_pair_1", "Mode_pair_2": "Mode_pair_2", "Mode_pair_3": "Mode_pair_3", "Mode_pair_4": "Mode_pair_4", \
             "Mode_pair_1_prop_out": "Prop. out Mode_pair_1", "Mode_pair_2_prop_out": "Prop. out Mode_pair_2", "Mode_pair_3_prop_out": "Prop. out Mode_pair_3",
             "Mode_pair_4_prop_out": "Prop. out Mode_pair_4", \
             "BP_density_range_dim_1": "BP density range 1", "BP_density_range_dim_2": "BP density range 2",
             "delta_slope_inner_prod": r"$\langle \mu_{G, 1}, \mu_{G, 2} \rangle$"
             }

group_by_name_list_tuning = group_by_name_list.copy()
group_by_name_list_tuning.remove("iter")
group_by_name_list_tuning.remove("opt_type")
group_by_name_list_tuning.remove("zeta")

def Fix_float_error_for_df(data_frame, var_name, var_value_list, eps=1e-8, abs_eps=True):
    """ After this fix, it becomes possible to use == condition for float values. """
    data_frame_fixed = data_frame.copy()
    for var_value in var_value_list:
        if abs_eps:
            tol = eps
        else:
            tol = var_value * eps
        data_frame_fixed.loc[(data_frame_fixed[var_name] < var_value + tol)
                             & (data_frame_fixed[var_name] > var_value - tol), var_name] = var_value
    return data_frame_fixed


def Print_unique_values_in_df(data_frame, var_list=None):
    """ Print the unique values in the specified columns. """
    if var_list is None:
        var_list = data_frame.columns

    for column in data_frame.columns:
        if column in var_list:
            print("---", column, data_frame[column].unique())
            for val in data_frame[column].unique():
                print(val)


def Get_results_clean_and_melt(results_df, metrics_all=metrics_all):
    """ Clean data type. Melt the data frame (multiple metric column -> a single metric column). """
    results_df_cleaned = results_df.copy()
    results_df_cleaned["exp_name"] = ""
    mask = pd.to_numeric(results_df_cleaned['KL'], errors='coerce').isna()
    results_df_cleaned = results_df_cleaned[~mask]
    results_df_cleaned["KL"] = results_df_cleaned["KL"].astype(float)
    results_df_cleaned["KL_mode"] = results_df_cleaned["KL_mode"].astype(float)
    results_df_cleaned["alpha_mobility"] = results_df_cleaned["alpha_mobility"].astype(float)
    results_df_cleaned["alpha_mobility_D"] = results_df_cleaned["alpha_mobility_D"].astype(float)

    results_df_cleaned["iter"] = results_df_cleaned["iter"].astype(int)
    results_df_cleaned["iteration"] = results_df_cleaned["iteration"].astype(int)
    results_df_cleaned["covered_mode_num"] = results_df_cleaned["covered_mode_num"].astype(int)
    results_df_cleaned["seed"] = results_df_cleaned["seed"].astype(int)
    results_df_cleaned["log_KL"] = np.log(results_df_cleaned["KL"])
    results_df_cleaned["log_KL_mode"] = np.log(results_df_cleaned["KL_mode"])
    results_df_cleaned["IS"] = results_df_cleaned["KL"]
    results_df_cleaned.loc[results_df_cleaned["opt_type"] == "sgd", "gamma"] = 1

    results_df_cleaned["gamma"] = results_df_cleaned["gamma"].astype(float)
    results_df_cleaned["zeta"] = results_df_cleaned["zeta"].astype(float)
    results_df_cleaned["g_hidden"] = results_df_cleaned["g_hidden"].astype(int)
    results_df_cleaned["g_layers"] = results_df_cleaned["g_layers"].astype(int)
    results_df_cleaned["d_hidden"] = results_df_cleaned["d_hidden"].astype(int)
    results_df_cleaned["d_layers"] = results_df_cleaned["d_layers"].astype(int)
    results_df_cleaned["batch_size"] = results_df_cleaned["batch_size"].astype(int)
    results_df_cleaned["z_dim"] = results_df_cleaned["z_dim"].astype(int)

    results_df_cleaned["ntk_G_1_reldiff"] = results_df_cleaned["ntk_G_1_reldiff"].astype(float)
    results_df_cleaned["ntk_G_2_reldiff"] = results_df_cleaned["ntk_G_2_reldiff"].astype(float)
    results_df_cleaned["ntk_D_reldiff"] = results_df_cleaned["ntk_D_reldiff"].astype(float)
    """ num_plot_iter = 4000 for 2D GMM, 8000 for MNIST """
    """ Cumulative absolute value of angle differences in degrees per neuron, with angles measured per num_plot_iter iteration. """
    results_df_cleaned["update_angle_BP_directions_G_tot_deg"] = results_df_cleaned["update_angle_BP_directions_G_tot_deg"].astype(float) / results_df_cleaned[
        "g_hidden"].astype(float)
    """ Cumulative absolute value of differences of distances to the origin per neuron, with distances measured per num_plot_iter iteration. """
    results_df_cleaned["update_BP_distances_G_tot_norm"] = results_df_cleaned["update_BP_distances_G_tot_norm"].astype(float) / results_df_cleaned[
        "g_hidden"].astype(float)
    """ Cumulative absolute value of differences of delta-slopes per neuron, with delta-slopes measured per num_plot_iter iteration. """
    results_df_cleaned["update_BP_delta_slopes_G_tot_norm"] = results_df_cleaned["update_BP_delta_slopes_G_tot_norm"].astype(float) / results_df_cleaned[
        "g_hidden"].astype(float)
    results_df_cleaned["update_angle_BP_directions_D_tot_deg"] = results_df_cleaned["update_angle_BP_directions_D_tot_deg"].astype(float) / results_df_cleaned[
        "d_hidden"].astype(float)
    results_df_cleaned["update_BP_distances_D_tot_norm"] = results_df_cleaned["update_BP_distances_D_tot_norm"].astype(float) / results_df_cleaned[
        "d_hidden"].astype(float)
    results_df_cleaned["update_BP_delta_slopes_D_tot_norm"] = results_df_cleaned["update_BP_delta_slopes_D_tot_norm"].astype(float) / results_df_cleaned[
        "d_hidden"].astype(float)
    #     for var in grad_norm_var_list:
    #         results_df_cleaned[var] = results_df_cleaned[var].astype(float)

    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "g_lr", [0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "d_lr", [0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "gamma", [0.8, 0.9, 0.99, 0.999, 0.9999, 1])

    print(group_by_name_list)
    # print(results_df_cleaned.columns)
    results_df_melted = pd.melt(results_df_cleaned, id_vars=group_by_name_list + ["seed"], value_vars=["log_KL", "log_KL_mode"] + metrics_all)
    results_df_melted.loc[(~results_df_melted["value"].isna()) & (~results_df_melted["value"].isin(["None"])), "value"] = results_df_melted.loc[
        (~results_df_melted["value"].isna()) & (~results_df_melted["value"].isin(["None"])), "value"].astype(float)
    print(len(results_df_melted.columns))

    return results_df_cleaned, results_df_melted


def Rename_results_data(results_df_melted_opt_type, rename_dict=rename_dict, name_dict=name_dict):
    """ Rename the columns according to given rename dict. """
    results_df_melted_opt_type_renamed = results_df_melted_opt_type.rename(columns=rename_dict)
    if "opt_type" in results_df_melted_opt_type.columns:
        for opt_type in ["sgd", "rmsprop", "rmsprop-x", "rmsprop-y"]:
            results_df_melted_opt_type_renamed.loc[results_df_melted_opt_type_renamed["Opt alg"] == opt_type, "Opt alg"] = name_dict[opt_type]

    return results_df_melted_opt_type_renamed


def Get_max_performance_df(results_df_melted_opt_type, table_var_list, tuned_var,
                           tune_by_var="log_KL", dataset_name="grid5", alpha_G=1, alpha_D=1, max_lr=1.01e-3):
    """ Get a table that lists the best var for a given configuration, based on specified metric. """
    tune_by_var = "log_KL"
    results_plot = results_df_melted_opt_type[(results_df_melted_opt_type["variable"] == tune_by_var) \
                                              & (results_df_melted_opt_type["data"] == dataset_name) \
                                              & (results_df_melted_opt_type["alpha_mobility"] == alpha_G) \
                                              & (results_df_melted_opt_type["alpha_mobility_D"] == alpha_D) \
                                              & (results_df_melted_opt_type["g_lr"] <= max_lr) \
                                              & (results_df_melted_opt_type["d_lr"] <= max_lr)]
    results_plot = results_plot.drop(["opt_type", "zeta"], axis=1)
    results_plot_max_iter = results_plot[results_plot["iter"] == 400000]
    selected_var_list = table_var_list + [tuned_var] + ["value"]
    # print("selected_var_list", selected_var_list)
    results_plot_max_iter_clean = results_plot_max_iter[selected_var_list + ["seed"]]
    results_plot_max_iter_clean_mean = results_plot_max_iter_clean.groupby(table_var_list + [tuned_var]).agg({"value": np.mean}).reset_index()
    results_plot_max_iter_max_performance = results_plot_max_iter_clean_mean.loc[
        results_plot_max_iter_clean_mean.groupby(table_var_list)["value"].idxmin()]
    results_plot_max_iter_max_performance = results_plot_max_iter_max_performance.sort_values("value")

    return results_plot_max_iter_max_performance




def Get_tuned_df(results_df, best_lr_table):
    """ Given the best var for a certain configuration, select all experiments using this configuration with this var value. """
    results_df_tuned = results_df.copy()
    row_var = best_lr_table.index.name
    row_val_list = best_lr_table.index.values
    col_var = best_lr_table.columns.name
    col_val_list = best_lr_table.columns.values

    results_df_tuned_list = []
    for row_val in row_val_list:
        for col_val in col_val_list:
            results_df_tuned_tmp = results_df_tuned[(results_df_tuned[row_var] == row_val) &
                                                    (results_df_tuned[col_var] == col_val) &
                                                    (results_df_tuned[r"$\eta$"] == best_lr_table.loc[row_val, col_val])]
            results_df_tuned_list.append(results_df_tuned_tmp)

    results_df_tuned = pd.concat(results_df_tuned_list, axis=0)

    return results_df_tuned


def Get_result_df_from_result_csv_task_dir_list(result_csv_task_dir_list):
    result_df_list = [pd.read_csv(os.path.join(summary_dir, file_dir, f"results.csv")) for file_dir in result_csv_task_dir_list]
    results_df = pd.concat(result_df_list, axis=0)
    return results_df


def Get_result_cleaned_and_melted(result_csv_task_dir_list):
    results_df = Get_result_df_from_result_csv_task_dir_list(result_csv_task_dir_list)
    results_df_cleaned, results_df_melted = Get_results_clean_and_melt(results_df, metrics_all=metrics_all)
    return results_df_cleaned, results_df_melted




def Get_tuned_df_from_result_csv_task_dir_list(result_csv_task_dir_list, dataset_name, tuning=True):
    """ Integrated function of getting (un)tuned df from filepath. """
    results_df_cleaned, results_df_melted = Get_result_cleaned_and_melted(result_csv_task_dir_list)
    results_df_melted_renamed = Rename_results_data(results_df_melted, rename_dict=rename_dict, name_dict=name_dict)

    table_var_list = ["g_hidden", "gamma"]
    tuned_var = "g_lr"
    tune_by_var = "log_KL"
    pd.set_option("precision", 7)

    results_plot_max_iter_max_performance = Get_max_performance_df(results_df_melted, table_var_list, tuned_var,
                                                                   tune_by_var=tune_by_var, dataset_name=dataset_name)
    results_plot_max_iter_max_performance = Rename_results_data(results_plot_max_iter_max_performance,
                                                                rename_dict=rename_dict, name_dict=name_dict)
    results_plot_max_iter_max_performance
    best_lr_table = results_plot_max_iter_max_performance.pivot(index=rename_dict["g_hidden"],
                                                                columns=rename_dict["gamma"],
                                                                values=rename_dict["g_lr"])
    min_log_KL = results_plot_max_iter_max_performance.sort_values("value")["value"].values[0]
    best_H = results_plot_max_iter_max_performance.sort_values("value").head(1)[r"$H$"].values[0]
    best_gamma = results_plot_max_iter_max_performance.sort_values("value").head(1)[r"$\gamma$"].values[0]
    #     print(results_plot_max_iter_max_performance.sort_values("value").head())
    #     print("min_log_KL", min_log_KL)
    """ Filter data here """
    results_plot = results_df_melted_renamed[(results_df_melted_renamed["data"] == dataset_name) \
                                             & (results_df_melted_renamed[r"$\alpha_G$"] == 1) \
                                             & (results_df_melted_renamed[r"$\alpha_D$"] == 1) \
                                             & (results_df_melted_renamed[r"$\eta$"] <= 1.01e-3)]
    if tuning:
        results_plot_tuned = Get_tuned_df(results_plot, best_lr_table)
    else:
        results_plot_tuned = results_plot
    return results_plot_tuned




def Replace_grad_norm_mode_pair_name(result_tuned_df, grad_norm_rename_dict):
    """ Replace mode pair names for consistency. """
    result_tuned_df_tmp = result_tuned_df.copy()
    for key in grad_norm_rename_dict:
        result_tuned_df_tmp.loc[result_tuned_df_tmp["variable"] == key, "variable"] = grad_norm_rename_dict[key]

    return result_tuned_df_tmp


def Unstack_df(stacked_df):
    unstacked_df = stacked_df.set_index(["iter", 'data', 'method', 'Opt alg', 'use_spectral_norm', 'z_dim',
                                         'iteration', 'batch_size', r'$\alpha_G$', r'$\alpha_D$', 'g_layers',
                                         'd_layers', r'$H$', r'$H_D$', r'$\eta$', 'd_lr', r'$\gamma$', r'$\zeta$', "seed", "variable"]).unstack(
        level=-1).reset_index()

    new_column_name_list = []
    for val in unstacked_df.columns:
        if val[1] == "":
            new_column_name_list.append(val[0])
        if val[0] == "value":
            new_column_name_list.append(val[1])

    unstacked_df.columns = new_column_name_list
    return unstacked_df




def Get_result_tuned_clean_unmelt(result_tuned_df, final_iter=400000, filter_vars=True):
    """ Double-melt the df for plotting. """
    result_tuned_clean = result_tuned_df[result_tuned_df["iter"] == final_iter]
    if filter_vars:
        result_tuned_clean = result_tuned_clean[result_tuned_clean["variable"].isin(
            ['prop_neg_samples', 'log_KL', 'log_KL_mode', "loss_D", "covered_mode_num",
             'update_angle_BP_directions_G_tot_deg_mean', 'update_BP_distances_G_tot_norm_mean', 'update_BP_delta_slopes_G_tot_norm_mean',
             'update_angle_BP_directions_D_tot_deg_mean', 'update_BP_distances_D_tot_norm_mean', 'update_BP_delta_slopes_D_tot_norm_mean',
             'update_weights_G_tot_norm_mean', 'update_biases_G_tot_norm_mean', 'update_vweights_G_tot_norm_mean',
             'update_weights_D_tot_norm_mean', 'update_biases_D_tot_norm_mean', 'update_vweights_D_tot_norm_mean'])]

    result_tuned_clean_unmelt = result_tuned_clean.set_index(["iter", 'data', 'method', 'Opt alg', 'use_spectral_norm', 'z_dim',
                                                              'iteration', 'batch_size', r'$\alpha_G$', r'$\alpha_D$', 'g_layers',
                                                              'd_layers', r'$H$', r'$H_D$', r'$\eta$', 'd_lr', r'$\gamma$', r'$\zeta$', "seed",
                                                              "variable"]).unstack(level=-1).reset_index()

    new_column_name_list = []
    for val in result_tuned_clean_unmelt.columns:
        if val[1] == "":
            new_column_name_list.append(val[0])
        if val[0] == "value":
            new_column_name_list.append(val[1])

    result_tuned_clean_unmelt.columns = new_column_name_list
    return result_tuned_clean_unmelt


def Get_grad_norm_plot_ready_df(result_tuned_df, final_iter=400000):
    result_tuned_clean_unmelt = Get_result_tuned_clean_unmelt(result_tuned_df, final_iter=final_iter)
    print(result_tuned_clean_unmelt.shape)

    result_tuned_clean_melted = pd.melt(result_tuned_clean_unmelt,
                                        id_vars=["data", r"$H$", r"$\gamma$", r'$\eta$', 'prop_neg_samples', 'log_KL', 'log_KL_mode', "loss_D",
                                                 "covered_mode_num",
                                                 #                                                                             "Mode_pair_1_prop_out", "Mode_pair_2_prop_out", "Mode_pair_3_prop_out", "Mode_pair_4_prop_out",
                                                 'update_angle_BP_directions_G_tot_deg_mean', 'update_BP_distances_G_tot_norm_mean',
                                                 'update_BP_delta_slopes_G_tot_norm_mean',
                                                 'update_angle_BP_directions_D_tot_deg_mean', 'update_BP_distances_D_tot_norm_mean',
                                                 'update_BP_delta_slopes_D_tot_norm_mean',
                                                 'update_weights_G_tot_norm_mean', 'update_biases_G_tot_norm_mean', 'update_vweights_G_tot_norm_mean',
                                                 'update_weights_D_tot_norm_mean', 'update_biases_D_tot_norm_mean', 'update_vweights_D_tot_norm_mean'
                                                 ],
                                        value_vars=BP_entropy_list, var_name="z_loc", value_name=r"$||\nabla_z G||$")

    result_tuned_clean_melted[r"$||\nabla_z G||$"] = result_tuned_clean_melted[r"$||\nabla_z G||$"].astype(float)
    result_tuned_clean_melted["log_KL_"] = result_tuned_clean_melted["log_KL"]

    result_tuned_clean_melted = pd.melt(result_tuned_clean_melted, id_vars=["data", r"$H$", r"$\gamma$", r'$\eta$', "z_loc", r"$||\nabla_z G||$"],
                                        value_vars=['prop_neg_samples', 'log_KL', 'log_KL_mode', "loss_D", "covered_mode_num",
                                                    #                                                       "Mode_pair_1_prop_out", "Mode_pair_2_prop_out", "Mode_pair_3_prop_out", "Mode_pair_4_prop_out",
                                                    'update_angle_BP_directions_G_tot_deg_mean', 'update_BP_distances_G_tot_norm_mean',
                                                    'update_BP_delta_slopes_G_tot_norm_mean',
                                                    'update_angle_BP_directions_D_tot_deg_mean', 'update_BP_distances_D_tot_norm_mean',
                                                    'update_BP_delta_slopes_D_tot_norm_mean',
                                                    'update_weights_G_tot_norm_mean', 'update_biases_G_tot_norm_mean', 'update_vweights_G_tot_norm_mean',
                                                    'update_weights_D_tot_norm_mean', 'update_biases_D_tot_norm_mean', 'update_vweights_D_tot_norm_mean'
                                                    ])
    result_tuned_clean_melted["value"] = result_tuned_clean_melted["value"].astype(float)
    result_tuned_clean_melted[r"$\gamma^\dagger$"] = np.log(1 - result_tuned_clean_melted[r"$\gamma$"] + 1e-12)
    return result_tuned_clean_melted

def Get_nonnan_unique_values(unique_values):
    unique_values_nonnan = []
    for value in unique_values:
        if not isinstance(value, str):
            if not np.isnan(value):
                unique_values_nonnan.append(value)
        else:
            unique_values_nonnan.append(value)

    return unique_values_nonnan

def Adding_init_and_final_values_to_result_df(results_df):
    final_iter = results_df["iteration"].values[0]
    results_df = results_df.rename(columns=dict(zip(BP_delta_slopes_G_Lp_norm_init_list, BP_delta_slopes_G_Lp_norm_init_list_rename)))
    results_df = results_df.rename(columns=dict(zip(BP_delta_slopes_D_Lp_norm_init_list, BP_delta_slopes_D_Lp_norm_init_list_rename)))
    results_df = results_df.rename(columns=dict(zip(BP_delta_slopes_G_Lp_norm_final_list, BP_delta_slopes_G_Lp_norm_final_list_rename)))
    results_df = results_df.rename(columns=dict(zip(BP_delta_slopes_D_Lp_norm_final_list, BP_delta_slopes_D_Lp_norm_final_list_rename)))
    results_df_cleaned = results_df.copy()
    results_df_cleaned = results_df_cleaned[results_df_cleaned["iter"].isin([0, final_iter])]
    results_df_cleaned.loc[results_df_cleaned["opt_type"] == "sgd", "gamma"] = 1
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "g_lr",
                                                [0.0000001, 0.000000316, 0.000001, 0.00000316, 0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "d_lr",
                                                [0.0000001, 0.000000316, 0.000001, 0.00000316, 0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "alpha_mobility",
                                                [0.0000001, 0.000000316, 0.000001, 0.00000316, 0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01,
                                                 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "alpha_mobility_D",
                                                [0.0000001, 0.000000316, 0.000001, 0.00000316, 0.00001, 0.0000316, 0.0001, 0.000316, 0.001, 0.00316, 0.01,
                                                 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100])
    results_df_cleaned = Fix_float_error_for_df(results_df_cleaned, "gamma", [0.8, 0.9, 0.99, 0.999, 0.9999, 1])
    single_value_column_list = [column for column in results_df_cleaned.columns if len(Get_nonnan_unique_values(results_df_cleaned[column].unique())) <= 1] + [
        "opt_type"]

    results_df_cleaned = results_df_cleaned.drop(columns=single_value_column_list)
    results_df_cleaned["BP_G_density_range"] = (results_df_cleaned["BP_density_range_dim_1"] + results_df_cleaned["BP_density_range_dim_2"]) / 2
    results_df_cleaned = results_df_cleaned.drop(columns=["BP_density_range_dim_1", "BP_density_range_dim_2"])

    if "zeta" in results_df_cleaned.columns:
        results_df_cleaned = results_df_cleaned.drop(columns=["zeta"])
    if "BP_G_1_entropy" in results_df_cleaned.columns and "BP_G_2_entropy" in results_df_cleaned.columns:
        results_df_cleaned = results_df_cleaned.drop(columns=["BP_G_1_entropy", "BP_G_2_entropy"])

    var_with_init_or_final_list = [column for column in results_df_cleaned.columns if
                                   "init" in column or "final" in column or "update" in column or "diff" in column]
    exp_setting_var_list = ['seed', 'g_hidden', 'd_hidden', 'g_lr', 'd_lr', 'gamma']
    leave_out_var_list = ["KL", "KL_mode", "covered_mode_num", "prop_neg_samples",
                          "mean_of_dG_dz_interpolated_max", "mean_of_dD_dx_interpolated_max", "proportion_outliers_interpolated"]
    results_df_cleaned_leave_out_vars = results_df_cleaned[["iter"] + exp_setting_var_list + leave_out_var_list + var_with_init_or_final_list]
    results_df_cleaned_needs_init_vals = results_df_cleaned.drop(columns=leave_out_var_list + var_with_init_or_final_list)
    results_df_cleaned_needs_init_vals_init = results_df_cleaned_needs_init_vals[results_df_cleaned_needs_init_vals["iter"] == 0]
    results_df_cleaned_needs_init_vals_final = results_df_cleaned_needs_init_vals[results_df_cleaned_needs_init_vals["iter"] == final_iter]
    rename_column_list = [column for column in results_df_cleaned_needs_init_vals if column not in exp_setting_var_list and column != "iter"]
    rename_dict_init = dict(zip(rename_column_list, [f"{column}_init" for column in rename_column_list]))
    results_df_cleaned_needs_init_vals_init_renamed = results_df_cleaned_needs_init_vals_init.rename(columns=rename_dict_init).drop(columns="iter")
    rename_dict_final = dict(zip(rename_column_list, [f"{column}_final" for column in rename_column_list]))
    results_df_cleaned_needs_init_vals_final_renamed = results_df_cleaned_needs_init_vals_final.rename(columns=rename_dict_final).drop(columns="iter")
    results_df_cleaned_needs_init_vals_added = pd.merge(results_df_cleaned_needs_init_vals_init_renamed,
                                                        results_df_cleaned_needs_init_vals_final_renamed, on=exp_setting_var_list)
    results_df_cleaned = pd.merge(results_df_cleaned_leave_out_vars[results_df_cleaned_leave_out_vars["iter"] == final_iter],
                                  results_df_cleaned_needs_init_vals_added, on=exp_setting_var_list)
    results_fe_df = results_df_cleaned.rename(columns={"g_hidden": "H", "g_lr": "eta"})
    #     results_fe_df = results_fe_df.astype(float)
    results_fe_df["log_KL"] = np.log(results_fe_df["KL"])
    results_fe_df["log10_eta"] = np.log10(results_fe_df["eta"])
    results_fe_df["gamma_dagger"] = 1 - results_fe_df["gamma"] + 1e-12
    results_fe_df["log10_gamma_dagger"] = np.log10(results_fe_df["gamma_dagger"])
    results_fe_df["log2_H"] = np.log2(results_fe_df["H"])
    results_fe_df = results_fe_df.drop(columns=["iter", "d_hidden", "d_lr", "seed"])
    return results_df, results_df_cleaned, results_fe_df

def Get_results_fe_df_best_lr_renamed(result_csv_task_dir_list):
    results_df = Get_result_df_from_result_csv_task_dir_list(result_csv_task_dir_list)
    results_df, results_df_cleaned, results_fe_df = Adding_init_and_final_values_to_result_df(results_df)
    results_fe_df_groupby = results_fe_df.groupby(["H", "gamma", "eta"]).agg({"log_KL": "mean"}).reset_index()
    results_fe_df_groupby_tuned_H_gamma = results_fe_df_groupby.loc[results_fe_df_groupby.groupby(["H", "gamma"])["log_KL"].idxmin()]
    best_lr_table = results_fe_df_groupby_tuned_H_gamma.pivot(index="H", columns="gamma", values="eta")
    H_list = results_fe_df["H"].unique()
    gamma_list = results_fe_df["gamma"].unique()
    results_fe_df_best_lr_list = []
    for H in H_list:
        for gamma in gamma_list:
            results_fe_df_best_lr_list.append(results_fe_df[(results_fe_df["H"] == H) & (results_fe_df["gamma"] == gamma) & (results_fe_df["eta"] == best_lr_table[gamma][H])])
    results_fe_df_best_lr = pd.concat(results_fe_df_best_lr_list, axis=0)
    results_fe_df_best_lr["H"] = results_fe_df_best_lr["H"].astype(int)
    results_fe_df_best_lr_renamed = results_fe_df_best_lr.rename(columns={"H": r"$H$", "gamma": r"$\gamma$", "log_KL": r"$\log \mathrm{KL}$"})
    return results_fe_df_best_lr_renamed

if __name__ == "__main__":
    results_fe_df_best_lr_renamed_dict = {}
    results_fe_df_best_lr_renamed_dict["grid"] = Get_results_fe_df_best_lr_renamed(["Grid_simgd_gamma0.8-0.9-0.99-0.999-0.9999-1_widthGD-32-64-128-256-512-1024-2048", "Grid_simgd_gamma0.8-0.9-0.99-0.999-0.9999-1_widthGD-4096-8192-16384"])
    results_fe_df_best_lr_renamed_dict["random"] = Get_results_fe_df_best_lr_renamed(["Random_simgd_gamma0.8-0.9-0.99-0.999-0.9999-1_widthGD-32-64-128-256-512-1024-2048", "Random_simgd_gamma0.8-0.9-0.99-0.999-0.9999-1_widthGD-4096-8192-16384"])
    results_fe_df_best_lr_renamed_dict["grid"]["Dataset"] = "Grid"
    results_fe_df_best_lr_renamed_dict["random"]["Dataset"] = "Random"
    for key in results_fe_df_best_lr_renamed_dict:
        results_fe_df_best_lr_renamed_dict[key] = results_fe_df_best_lr_renamed_dict[key].rename(columns={"proportion_outliers_interpolated": "Prop. mixtures"})
        results_fe_df_best_lr_renamed_dict[key] = results_fe_df_best_lr_renamed_dict[key][results_fe_df_best_lr_renamed_dict[key][r"$\gamma$"].isin([0.999, 1])]

    results_fe_df_best_lr_renamed_dict_grid_random = pd.concat([results_fe_df_best_lr_renamed_dict["grid"], results_fe_df_best_lr_renamed_dict["random"]], axis=0)

    sns.set(font_scale=1.75)
    base_size = 20
    aspect_ratio = 5
    fig = plt.figure(figsize=(base_size, base_size / aspect_ratio))
    ax_dict = {}
    ax_dict["grid"] = plt.subplot2grid((1, 2), (0, 0))
    ax_dict["random"] = plt.subplot2grid((1, 2), (0, 1))
    g = sns.lineplot(data=results_fe_df_best_lr_renamed_dict_grid_random, x=r"$H$", y=r"$\log \mathrm{KL}$", hue="Dataset", style=r"$\gamma$", err_style="bars",
                     ax=ax_dict["grid"])
    h1, l1 = ax_dict["grid"].get_legend_handles_labels()
    ylim = [-2, 2]
    ylim = [-2.5, 3]
    # ax_dict["grid"].set_ylim(ylim)
    ax_dict["grid"].get_legend().remove()
    ax_dict["grid"].set_xscale("log")
    ax_dict["grid"].xaxis.set_major_locator(LogLocator(base=2, numticks=20))
    ax_dict["grid"].set_xticklabels([str(val) for val in 2 ** np.arange(3, 15)])

    # ax_dict["grid"].locator_params("x", {"nbins": 10, "steps": 2 ** np.arange(5, 15), "min_n_ticks": 10})
    g = sns.lineplot(data=results_fe_df_best_lr_renamed_dict_grid_random, x=r"$H$", y=r"Prop. mixtures", hue="Dataset", style=r"$\gamma$", err_style="bars",
                     ax=ax_dict["random"])
    h2, l2 = ax_dict["random"].get_legend_handles_labels()
    # ax_dict["random"].set_ylim(ylim)
    # ax_dict["random"].set_ylabel("")
    # ax_dict["random"].set_yticklabels([])
    ax_dict["random"].get_legend().remove()
    ax_dict["random"].set_xscale("log")
    ax_dict["random"].xaxis.set_major_locator(LogLocator(base=2, numticks=20))
    ax_dict["random"].set_xticklabels([str(val) for val in 2 ** np.arange(3, 15)])
    fig.legend(h1, l1, loc="center right")
    plt.tight_layout()
    fig.subplots_adjust(right=0.88)
    plt.show()