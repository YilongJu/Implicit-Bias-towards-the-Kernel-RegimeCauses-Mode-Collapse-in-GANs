#%%

# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from matplotlib.colors import LogNorm
import seaborn as sns
import scipy as sp
sns.set()
#%%



summary_dir = "Summaries"
group_by_name_list = ['iter', 'data', 'method', 'opt_type', 'use_spectral_norm', 'z_dim',
       'iteration', 'batch_size', 'alpha_mobility', 'alpha_mobility_D', 'g_layers', 'd_layers', 'g_hidden',
       'd_hidden', 'g_lr', 'd_lr', 'gamma', 'zeta']

grad_norm_var_list = ['proportion_outliers_interpolated', 'dG_dz_mean', 'dD_dx_mean', 'mean_of_dG_dz_interpolated_max', 'mean_of_dD_dx_interpolated_max']
# grad_norm_var_list = ['Mode_2_to_3', 'Mode_3_to_4', 'Mode_2_to_5', 'Mode_4_to_5', 'prop_neg_samples']
new_var_list = ["BP_density_range_dim_1", "BP_density_range_dim_2", "delta_slope_inner_prod"]

metrics_all = ["KL", "covered_mode_num", 'prop_neg_samples'] + grad_norm_var_list + new_var_list + ["loss_G", "loss_D", "L_G", "L_D", "rel_act_diff_G", "rel_act_diff_D", "ntk_G_1_reldiff", "ntk_G_2_reldiff", "ntk_D_reldiff", \
               "update_angle_BP_directions_G_tot_deg_mean", "update_BP_distances_G_tot_norm_mean", "update_BP_delta_slopes_G_tot_norm_mean", \
               "update_angle_BP_directions_D_tot_deg_mean", "update_BP_distances_D_tot_norm_mean", "update_BP_delta_slopes_D_tot_norm_mean", \
               "update_angle_BP_directions_G_tot_deg_std" , "update_BP_distances_G_tot_norm_std", "update_BP_delta_slopes_G_tot_norm_std", \
               "update_angle_BP_directions_D_tot_deg_std", "update_BP_distances_D_tot_norm_std", "update_BP_delta_slopes_D_tot_norm_std", \
               "update_weights_G_tot_norm_mean", "update_biases_G_tot_norm_mean", "update_vweights_G_tot_norm_mean", \
               "update_weights_D_tot_norm_mean", "update_biases_D_tot_norm_mean", "update_vweights_D_tot_norm_mean", \
               "update_weights_G_tot_norm_std", "update_biases_G_tot_norm_std", "update_vweights_G_tot_norm_std", \
               "update_weights_D_tot_norm_std", "update_biases_D_tot_norm_std", "update_vweights_D_tot_norm_std", \
               'update_angle_BP_directions_G_tot_deg', 'update_BP_distances_G_tot_norm', 'update_BP_delta_slopes_G_tot_norm', \
               'update_angle_BP_directions_D_tot_deg', 'update_BP_distances_D_tot_norm', 'update_BP_delta_slopes_D_tot_norm', \
                'BP_G_entropy', 'BP_D_entropy', "affine_BP_G_prop"]
other_metrics = ["IS", "log_KL", "covered_mode_num"]
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
                 "update_angle_BP_directions_G_tot_deg": r"$\sum |\Delta \xi_G|$", "update_BP_distances_G_tot_norm": r"$\sum (\Delta \gamma_G)^2$", "update_BP_delta_slopes_G_tot_norm": r"$\sum ||\Delta \mu_G||_2^2$", \
                 "update_angle_BP_directions_D_tot_deg": r"$\sum |\Delta \xi_D|$", "update_BP_distances_D_tot_norm": r"$\sum (\Delta \gamma_D)^2$", "update_BP_delta_slopes_D_tot_norm": r"$\sum ||\Delta \mu_D||_2^2$", \
                 "update_angle_BP_directions_G_tot_deg_mean": r"$\mathrm{Avg}[|\Delta \xi_G|]$", "update_BP_distances_G_tot_norm_mean": r"$\mathrm{Avg}[|\Delta \gamma_G|]$", "update_BP_delta_slopes_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta \mu_G||_2]$", \
                 "update_angle_BP_directions_G_tot_deg_std": r"$\mathrm{Var}[\Delta \xi_G]$", "update_BP_distances_G_tot_norm_std": r"$\mathrm{Var}[\Delta \gamma_G]$", "update_BP_delta_slopes_G_tot_norm_std": r"$\mathrm{Var} [||\Delta \mu_G||_2]$", \
                 "update_weights_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta w_G||_2]$", "update_biases_G_tot_norm_mean": r"$\mathrm{Avg}[|\Delta b_G|]$", "update_vweights_G_tot_norm_mean": r"$\mathrm{Avg}[||\Delta v_G||_{2}]$", \
                 "update_weights_G_tot_norm_std": r"$\mathrm{Var}[\Delta w_G]$", "update_biases_G_tot_norm_std": r"$\mathrm{Var}[\Delta b_G]$", "update_vweights_G_tot_norm_std": r"$\mathrm{Var} [\Delta v_G]$", \
                 "update_angle_BP_directions_D_tot_deg_mean": r"$\mathrm{Avg}[|\Delta \xi_D|]$", "update_BP_distances_D_tot_norm_mean": r"$\mathrm{Avg}[|\Delta \gamma_D|]$", "update_BP_delta_slopes_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta \mu_D||_2]$", \
                 "update_angle_BP_directions_D_tot_deg_std": r"$\mathrm{Var}[\Delta \xi_D]$", "update_BP_distances_D_tot_norm_std": r"$\mathrm{Var}[\Delta \gamma_D]$", "update_BP_delta_slopes_D_tot_norm_std": r"$\mathrm{Var} [||\Delta \mu_D||_2]$", \
                 "update_weights_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta w_D||_2]$", "update_biases_D_tot_norm_mean": r"$\mathrm{Avg}[|\Delta b_D|]$", "update_vweights_D_tot_norm_mean": r"$\mathrm{Avg}[||\Delta v_D]||_{2}]$", \
                 "update_weights_D_tot_norm_std": r"$\mathrm{Var}[\Delta w_D]$", "update_biases_D_tot_norm_std": r"$\mathrm{Var}[\Delta b_D]$", "update_vweights_D_tot_norm_std": r"$\mathrm{Var} [\Delta v_D]$", \
                 "affine_BP_G_prop": "Prop. affine BP G", 'prop_neg_samples': "Prop. neg samples", \
                 "Mode_pair_1": "Mode_pair_1", "Mode_pair_2": "Mode_pair_2", "Mode_pair_3": "Mode_pair_3", "Mode_pair_4": "Mode_pair_4", \
                 "Mode_pair_1_prop_out": "Prop. out Mode_pair_1", "Mode_pair_2_prop_out": "Prop. out Mode_pair_2", "Mode_pair_3_prop_out": "Prop. out Mode_pair_3", "Mode_pair_4_prop_out": "Prop. out Mode_pair_4", \
                 "BP_density_range_dim_1": "BP density range 1", "BP_density_range_dim_2": "BP density range 2", "delta_slope_inner_prod": r"$\langle \mu_{G, 1}, \mu_{G, 2} \rangle$"
                }

group_by_name_list_tuning = group_by_name_list.copy()
group_by_name_list_tuning.remove("iter")
group_by_name_list_tuning.remove("opt_type")
group_by_name_list_tuning.remove("zeta")
group_by_name_list_tuning

#%%

""" After this fix, it becomes possible to use == condition for float values. """


def Fix_float_error_for_df(data_frame, var_name, var_value_list, eps=1e-8, abs_eps=True):
    data_frame_fixed = data_frame.copy()
    for var_value in var_value_list:
        if abs_eps:
            tol = eps
        else:
            tol = var_value * eps
        data_frame_fixed.loc[(data_frame_fixed[var_name] < var_value + tol)
                             & (data_frame_fixed[var_name] > var_value - tol), var_name] = var_value
    return data_frame_fixed


""" Print the unique values in the specified columns. """


def Print_unique_values_in_df(data_frame, var_list=None):
    if var_list is None:
        var_list = data_frame.columns

    for column in data_frame.columns:
        if column in var_list:
            print("---", column, data_frame[column].unique())
            for val in data_frame[column].unique():
                print(val)


""" Clean data type. Melt the data frame (multiple metric column -> a single metric column). """


def Get_results_clean_and_melt(results_df, metrics_all=metrics_all):
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
    for var in grad_norm_var_list:
        results_df_cleaned[var] = results_df_cleaned[var].astype(float)

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


""" Rename the columns according to given rename dict. """


def Rename_results_data(results_df_melted_opt_type, rename_dict=rename_dict, name_dict=name_dict):
    results_df_melted_opt_type_renamed = results_df_melted_opt_type.rename(columns=rename_dict)
    if "opt_type" in results_df_melted_opt_type.columns:
        for opt_type in ["sgd", "rmsprop", "rmsprop-x", "rmsprop-y"]:
            results_df_melted_opt_type_renamed.loc[results_df_melted_opt_type_renamed["Opt alg"] == opt_type, "Opt alg"] = name_dict[opt_type]

    return results_df_melted_opt_type_renamed


""" Get a table that lists the best var for a given configuration, based on specified metric. """


def Get_max_performance_df(results_df_melted_opt_type, table_var_list, tuned_var,
                           tune_by_var="log_KL", dataset_name="grid5", alpha_G=1, alpha_D=1, max_lr=1.01e-3):
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


""" Given the best var for a certain configuration, select all experiments using this configuration with this var value. """


def Get_tuned_df(results_df, best_lr_table):
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


def Get_result_cleaned_and_melted(result_csv_task_dir_list):
    result_df_list = [pd.read_csv(os.path.join(summary_dir, file_dir, f"results.csv")) for file_dir in result_csv_task_dir_list]
    results_df = pd.concat(result_df_list, axis=0)
    results_df_cleaned, results_df_melted = Get_results_clean_and_melt(results_df, metrics_all=metrics_all)
    return results_df_cleaned, results_df_melted


""" Integrated function of getting (un)tuned df from filepath. """


def Get_tuned_df_from_result_csv_task_dir_list(result_csv_task_dir_list, dataset_name, tuning=True):
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


""" Replace mode pair names for consistency. """


def Replace_grad_norm_mode_pair_name(result_tuned_df, grad_norm_rename_dict):
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


""" Double-melt the df for plotting. """


def Get_result_tuned_clean_unmelt(result_tuned_df, final_iter=400000, filter_vars=True):
    result_tuned_clean = result_tuned_df[result_tuned_df["iter"] == final_iter]
    if filter_vars:
        result_tuned_clean = result_tuned_clean[result_tuned_clean["variable"].isin(
            ['prop_neg_samples', 'log_KL', 'log_KL_mode', "loss_D", "covered_mode_num",
             'update_angle_BP_directions_G_tot_deg_mean', 'update_BP_distances_G_tot_norm_mean', 'update_BP_delta_slopes_G_tot_norm_mean',
             'update_angle_BP_directions_D_tot_deg_mean', 'update_BP_distances_D_tot_norm_mean', 'update_BP_delta_slopes_D_tot_norm_mean',
             'update_weights_G_tot_norm_mean', 'update_biases_G_tot_norm_mean', 'update_vweights_G_tot_norm_mean',
             'update_weights_D_tot_norm_mean', 'update_biases_D_tot_norm_mean', 'update_vweights_D_tot_norm_mean'] + new_var_list)]

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
                                        value_vars=new_var_list, var_name="z_loc", value_name=r"$||\nabla_z G||$")

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


#%%

selected_var_list = ["log_KL", "log_KL_mode", "covered_mode_num", "dG_dz_mean", "dD_dx_mean", "mean_of_dD_dx_interpolated_max",
                     "mean_of_dG_dz_interpolated_max", "prop_neg_samples", "proportion_outliers_interpolated", "rel_act_diff_G", "rel_act_diff_D",
                     "update_weights_G_tot_norm_mean", "update_weights_D_tot_norm_mean"]

#%%

final_metrics_var_list = ['BP_D_entropy', 'BP_G_entropy', 'BP_density_range_dim_1',
                          'BP_density_range_dim_2', 'affine_BP_G_prop',
                          'covered_mode_num', 'dD_dx_mean', 'dG_dz_mean',
                          'delta_slope_inner_prod', 'log_KL', 'log_KL_mode', 'loss_D', 'loss_G',
                          'mean_of_dD_dx_interpolated_max', 'mean_of_dG_dz_interpolated_max',
                          'prop_neg_samples', 'proportion_outliers_interpolated',
                          'rel_act_diff_D', 'rel_act_diff_G',
                          'update_BP_delta_slopes_D_tot_norm',
                          'update_BP_delta_slopes_D_tot_norm_mean',
                          'update_BP_delta_slopes_D_tot_norm_std',
                          'update_BP_delta_slopes_G_tot_norm',
                          'update_BP_delta_slopes_G_tot_norm_mean',
                          'update_BP_delta_slopes_G_tot_norm_std',
                          'update_BP_distances_D_tot_norm', 'update_BP_distances_D_tot_norm_mean',
                          'update_BP_distances_D_tot_norm_std', 'update_BP_distances_G_tot_norm',
                          'update_BP_distances_G_tot_norm_mean',
                          'update_BP_distances_G_tot_norm_std',
                          'update_angle_BP_directions_D_tot_deg',
                          'update_angle_BP_directions_D_tot_deg_mean',
                          'update_angle_BP_directions_D_tot_deg_std',
                          'update_angle_BP_directions_G_tot_deg',
                          'update_angle_BP_directions_G_tot_deg_mean',
                          'update_angle_BP_directions_G_tot_deg_std',
                          'update_biases_D_tot_norm_mean', 'update_biases_D_tot_norm_std',
                          'update_biases_G_tot_norm_mean', 'update_biases_G_tot_norm_std',
                          'update_vweights_D_tot_norm_mean', 'update_vweights_D_tot_norm_std',
                          'update_vweights_G_tot_norm_mean', 'update_vweights_G_tot_norm_std',
                          'update_weights_D_tot_norm_mean', 'update_weights_D_tot_norm_std',
                          'update_weights_G_tot_norm_mean', 'update_weights_G_tot_norm_std']


#%%

def Get_name(name_dict, var):
    if var in name_dict:
        return name_dict[var]
    else:
        return var


#%%

def Get_melted_df_for_pair_plot(result_tuned_df, final_iter=400000):
    result_tuned_clean_unmelt = Get_result_tuned_clean_unmelt(result_tuned_df, final_iter=final_iter, filter_vars=False)
    group_by_name_list_renamed = [name_dict[name] if name in name_dict else name for name in group_by_name_list]
    group_by_name_list_renamed = [rename_dict[name] if name in rename_dict else name for name in group_by_name_list]
    result_tuned_clean_melt = pd.melt(result_tuned_clean_unmelt, id_vars=group_by_name_list_renamed + ["seed"],
                                      value_vars=final_metrics_var_list, var_name="var1", value_name="val1")
    result_tuned_clean_melt_aug = pd.merge(result_tuned_clean_melt, result_tuned_clean_unmelt, on=group_by_name_list_renamed + ["seed"])
    result_tuned_clean_melt_aug_melt = pd.melt(result_tuned_clean_melt_aug, id_vars=group_by_name_list_renamed + ["seed", "var1", "val1"],
                                               value_vars=final_metrics_var_list, var_name="var2", value_name="val2")
    result_tuned_clean_melt_aug_melt["val1"] = result_tuned_clean_melt_aug_melt["val1"].astype(float)
    result_tuned_clean_melt_aug_melt["val2"] = result_tuned_clean_melt_aug_melt["val2"].astype(float)
    return result_tuned_clean_melt_aug_melt


#%% md

# Grid

#%%

results_df_cleaned_grid, results_df_melted_grid = Get_result_cleaned_and_melted(
    ["Jan27_grid_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"])

#%%

result_tuned_grid = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_grid_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
    "grid5", tuning=True)

#%%

palette = sns.color_palette("icefire", 7)
sns.palplot(palette)

#%%

result_tuned_grid[r"$H$"].unique()

#%%

g = sns.relplot(data=result_tuned_grid,
                x="iter", y="value", \
                kind="line", \
                hue=r"$H$", \
                row=r"$\gamma$", col="variable", \
                palette=palette, \
                #                 style="sep", \
                legend="brief", \
                height=4, aspect=1.2, \
                facet_kws={"sharey": False, "sharex": True, "margin_titles": False, "gridspec_kws": {"hspace": 0.1, "wspace": 0.18}})
axes = g.axes
for i, axe_row in enumerate(axes):
    for j, axe in enumerate(axe_row):
        #         axe.set_ylim(2.0, 4.5)
        title = axe.get_title()
        #         print(title)
        row_text = title.split("|")[0]
        col_text = title.split("|")[1]
        row_var = row_text.split("=")[1].strip()
        col_var = col_text.split("=")[1].strip()
        if i == 0:
            grad_norm_text = r"$||dG/dz||_F$"
            if col_var in name_dict:
                axe.set_title(f"{name_dict[col_var]}")
            else:
                axe.set_title(f"{col_var}")
        #             axe.set_title(f"{col_text}")
        else:
            axe.set_title("")

        if j == 0:
            axe.set_ylabel(row_text)
        #             axe.set_ylabel(name_dict[row_var])
        #             axe.set_ylim([-1.5, 3])
        else:
            axe.set_ylabel("")

#         if i == 0:
#             axe.set_ylim(0, 1)
#         elif i == 1:
#             axe.set_ylim(-2.5, 2.5)
#         elif i == 2:
#             axe.set_ylim(-5, 45)
#         elif i == 3:
#             axe.set_ylim(-0.5, 3.5)
#         elif i == 4:
#             axe.set_ylim(-0.5, 3.5)

#         if j == 0:
#             axe.set_xlim(-5, 200)
#         elif j == 1:
#             axe.set_xlim(-5, 200)
#         elif j == 2:
#             axe.set_xlim(-2, 2)

#         axe.set_xlabel("")
#         axe.set_ylim(-1.5, 3.5)


#%%

palette = sns.color_palette("icefire", 6)
sns.palplot(palette)

#%%

g = sns.relplot(data=result_tuned_grid,
                x="iter", y="value", \
                kind="line", \
                hue=r"$\gamma$", \
                row=r"$H$", col="variable", \
                palette=palette, \
                #                 style="sep", \
                legend="brief", \
                height=4, aspect=1.2, \
                facet_kws={"sharey": False, "sharex": True, "margin_titles": False, "gridspec_kws": {"hspace": 0.1, "wspace": 0.18}})
axes = g.axes
for i, axe_row in enumerate(axes):
    for j, axe in enumerate(axe_row):
        #         axe.set_ylim(2.0, 4.5)
        title = axe.get_title()
        #         print(title)
        row_text = title.split("|")[0]
        col_text = title.split("|")[1]
        row_var = row_text.split("=")[1].strip()
        col_var = col_text.split("=")[1].strip()
        if i == 0:
            grad_norm_text = r"$||dG/dz||_F$"
            #             axe.set_title(f"{name_dict[col_var]}")
            axe.set_title(f"{col_text}")
        else:
            axe.set_title("")

        if j == 0:
            axe.set_ylabel(row_text)
        #             axe.set_ylabel(name_dict[row_var])
        #             axe.set_ylim([-1.5, 3])
        else:
            axe.set_ylabel("")

#         if i == 0:
#             axe.set_ylim(0, 1)
#         elif i == 1:
#             axe.set_ylim(-2.5, 2.5)
#         elif i == 2:
#             axe.set_ylim(-5, 45)
#         elif i == 3:
#             axe.set_ylim(-0.5, 3.5)
#         elif i == 4:
#             axe.set_ylim(-0.5, 3.5)

#         if j == 0:
#             axe.set_xlim(-5, 200)
#         elif j == 1:
#             axe.set_xlim(-5, 200)
#         elif j == 2:
#             axe.set_xlim(-2, 2)

#         axe.set_xlabel("")
#         axe.set_ylim(-1.5, 3.5)


#%% md

## Correlation plot

#%%


result_tuned_clean_melt_aug_melt = Get_melted_df_for_pair_plot(result_tuned_grid)

#%%

palette = sns.color_palette("icefire", 7)
sns.palplot(palette)

#%%

g = sns.relplot(data=result_tuned_clean_melt_aug_melt[result_tuned_clean_melt_aug_melt["var1"].isin(selected_var_list)
                                                      & result_tuned_clean_melt_aug_melt["var2"].isin(selected_var_list)
                                                      & result_tuned_clean_melt_aug_melt[r"$\gamma$"].isin([0.999])],
                x="val1", y="val2", hue=r"$H$", row="var1", col="var2", \
                palette=palette, \
                #                 style="data", \
                legend="brief", \
                height=4, aspect=1.2, \
                facet_kws={"sharey": False, "sharex": False, "margin_titles": False, "gridspec_kws": {"hspace": 0.1, "wspace": 0.18}})
axes = g.axes
for i, axe_row in enumerate(axes):
    for j, axe in enumerate(axe_row):
        title = axe.get_title()
        row_text = title.split("|")[0]
        col_text = title.split("|")[1]
        row_var = row_text.split("=")[1].strip()
        col_var = col_text.split("=")[1].strip()
        if i == 0:
            grad_norm_text = r"$||dG/dz||_F$"
            axe.set_title(f"{Get_name(name_dict, col_var)}")
        #             axe.set_title(f"{col_var}")
        else:
            axe.set_title("")

        if j == 0:
            #             axe.set_ylabel(row_text)
            axe.set_ylabel(Get_name(name_dict, row_var))
        #             axe.set_ylim([-1.5, 3])
        else:
            axe.set_ylabel("")

#%%

result_tuned_clean_unmelt = Get_result_tuned_clean_unmelt(result_tuned_grid, filter_vars=False)

cat_var = r"$H$"
result_pairplot = result_tuned_clean_unmelt.copy()
result_pairplot = result_pairplot[[cat_var] + selected_var_list]
result_pairplot

#%%

sns.pairplot(result_pairplot, \
             palette=palette, \
             hue=r"$H$", \
             height=4, aspect=1.2)

#%%

cat_var = r"$H$"
result_pairplot = result_tuned_clean_unmelt.copy()
result_pairplot = result_pairplot[result_pairplot[r"$\gamma$"].isin([0.999])]
result_pairplot = result_pairplot[[cat_var] + selected_var_list]
result_pairplot
sns.pairplot(result_pairplot, \
             palette=palette, \
             hue=r"$H$", \
             height=4, aspect=1.2)

#%% md

# Random

#%%

result_tuned_random = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_random_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
    "random9-6_2", tuning=True)

#%%

result_tuned_clean_unmelt_random = Get_result_tuned_clean_unmelt(result_tuned_random, filter_vars=False)
#%%

palette = sns.color_palette("icefire", 7)
sns.palplot(palette)
#%%
cat_var = r"$H$"
result_pairplot_random = result_tuned_clean_unmelt_random.copy()
result_pairplot_random = result_pairplot_random[[cat_var] + selected_var_list[:10]]
sns.pairplot(result_pairplot_random, \
             palette=palette, \
             hue=r"$H$", \
             height=4, aspect=1.2)
plt.show()
#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%%


#%% md

# Old Code

#%%

result_tuned_grid = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001",
     "Jan08_simgd_sgd_seed01234_width-both-G-D-32-64-128-256-512-1024-2048_lr0.01-0.00316-0.001-0.000316-0.0001-0.0000316-0.00001"], "grid5", tuning=True)

#%%

result_tuned_grid["variable"].unique()

#%%

result_tuned_grid_2 = Get_tuned_df_from_result_csv_task_dir_list([
                                                                     "Feb15_grid5-2-0.01_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
                                                                 "grid5", tuning=True)

#%%

result_tuned_separated = Get_tuned_df_from_result_csv_task_dir_list([
                                                                        "Feb11_separated_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
                                                                    "separated", tuning=True)

#%%

result_tuned_random = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001",
     "Jan08_simgd_sgd_seed01234_width-both-G-D-32-64-128-256-512-1024-2048_lr0.01-0.00316-0.001-0.000316-0.0001-0.0000316-0.00001"], "random9-6_2", tuning=True)

#%%

result_grid = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001",
     "Jan08_simgd_sgd_seed01234_width-both-G-D-32-64-128-256-512-1024-2048_lr0.01-0.00316-0.001-0.000316-0.0001-0.0000316-0.00001"], "grid5", tuning=False)

#%%

result_grid_2 = Get_tuned_df_from_result_csv_task_dir_list([
                                                               "Feb15_grid5-2-0.01_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
                                                           "grid5", tuning=False)

#%%

result_separated = Get_tuned_df_from_result_csv_task_dir_list([
                                                                  "Feb11_separated_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999-1_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001"],
                                                              "separated", tuning=False)

#%%

result_random = Get_tuned_df_from_result_csv_task_dir_list(
    ["Jan27_simgd_rmsprop-1-gamma0.8-0.9-0.99-0.999-0.9999_seed01234_widthGD-32-64-128-256-512-1024-2048_lr0.001-0.000316-0.0001-0.0000316-0.00001",
     "Jan08_simgd_sgd_seed01234_width-both-G-D-32-64-128-256-512-1024-2048_lr0.01-0.00316-0.001-0.000316-0.0001-0.0000316-0.00001"], "random9-6_2",
    tuning=False)

#%%

grad_norm_rename_dict = {'Mode_4_to_5': "Mode_pair_4", 'Mode_4_to_8': "Mode_pair_2", 'Mode_5_to_8': "Mode_pair_3",
                         'Mode_7_to_5': "Mode_pair_1",
                         'Mode_2_to_3': "Mode_pair_1", 'Mode_3_to_4': "Mode_pair_2", 'Mode_2_to_5': "Mode_pair_3",
                         'Mode_4_to_5_prop_out': "Mode_pair_4_prop_out", 'Mode_4_to_8_prop_out': "Mode_pair_2_prop_out",
                         'Mode_5_to_8_prop_out': "Mode_pair_3_prop_out",
                         'Mode_7_to_5_prop_out': "Mode_pair_1_prop_out",
                         'Mode_2_to_3_prop_out': "Mode_pair_1_prop_out", 'Mode_3_to_4_prop_out': "Mode_pair_2_prop_out",
                         'Mode_2_to_5_prop_out': "Mode_pair_3_prop_out"}

#%%

result_tuned_grid_2["data"] = "grid-2"
result_grid_2["data"] = "grid-2"

#%%

result_tuned_clean_unmelt_grid = Get_grad_norm_plot_ready_df(result_tuned_grid)
result_tuned_clean_unmelt_grid_2 = Get_grad_norm_plot_ready_df(result_tuned_grid_2)
result_tuned_clean_unmelt_random = Get_grad_norm_plot_ready_df(result_tuned_random)
result_tuned_clean_unmelt_separated = Get_grad_norm_plot_ready_df(result_tuned_separated)

result_tuned_clean_unmelt_all = pd.concat(
    [result_tuned_clean_unmelt_grid, result_tuned_clean_unmelt_grid_2, result_tuned_clean_unmelt_random, result_tuned_clean_unmelt_separated], axis=0)
result_tuned_clean_unmelt_all.shape

#%%

# result_tuned_all = pd.concat([result_tuned_grid, result_tuned_grid_2, result_tuned_random, result_tuned_separated], axis=0)
result_tuned_all = Unstack_df(pd.concat([Replace_grad_norm_mode_pair_name(result_tuned_grid, grad_norm_rename_dict),
                                         Replace_grad_norm_mode_pair_name(result_tuned_grid_2, grad_norm_rename_dict),
                                         Replace_grad_norm_mode_pair_name(result_tuned_random, grad_norm_rename_dict),
                                         Replace_grad_norm_mode_pair_name(result_tuned_separated, grad_norm_rename_dict)], axis=0))
result_tuned_all.shape

#%%

result_clean_unmelt_grid = Get_grad_norm_plot_ready_df(result_grid)
result_clean_unmelt_grid_2 = Get_grad_norm_plot_ready_df(result_grid_2)
result_clean_unmelt_random = Get_grad_norm_plot_ready_df(result_random)
result_clean_unmelt_separated = Get_grad_norm_plot_ready_df(result_separated)
result_clean_unmelt_all = pd.concat([result_clean_unmelt_grid, result_clean_unmelt_grid_2, result_clean_unmelt_random, result_clean_unmelt_separated], axis=0)
result_clean_unmelt_all.shape

#%%

palette = sns.color_palette("icefire", 6)
sns.palplot(palette)

#%%

result_tuned_clean_unmelt_all_tmp = result_tuned_clean_unmelt_all.copy()
result_tuned_clean_unmelt_all_tmp.loc[result_tuned_clean_unmelt_all_tmp["z_loc"] == "delta_slope_inner_prod", r"$||\nabla_z G||$"] = \
result_tuned_clean_unmelt_all_tmp[result_tuned_clean_unmelt_all_tmp["z_loc"] == "delta_slope_inner_prod"][r"$||\nabla_z G||$"] / \
result_tuned_clean_unmelt_all_tmp[result_tuned_clean_unmelt_all_tmp["z_loc"] == "delta_slope_inner_prod"][r"$H$"]

#%%

result_tuned_clean_unmelt_all[result_tuned_clean_unmelt_all["z_loc"] == "delta_slope_inner_prod"]

#%%

result_tuned_clean_unmelt_all_tmp[result_tuned_clean_unmelt_all_tmp["z_loc"] == "delta_slope_inner_prod"]
