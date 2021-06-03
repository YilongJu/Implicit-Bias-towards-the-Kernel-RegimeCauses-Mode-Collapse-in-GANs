import autograd.numpy as np
import autograd
import os
from autograd import grad
from autograd import jacobian
from mpl_toolkits.mplot3d import axes3d
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import pinv
import time
import random
import pickle
import argparse


from Toy_Experiments.Functions import Get_function
from Toy_Experiments.OptimizationAlgortihms import Get_trajectory
from ComputationalTools import *

from Toy_Experiments.Colorline import colorline
from mpl_toolkits.axes_grid1 import make_axes_locatable

Toy_experiment_folder = "Toy_Results"

epsilon = 1e-8
epsilon2 = 1e0
buffer = 1e-4
adapt_thres_rel_xy = 1e-3

line_type_dict = {"fr": "-", "nfr": "-", "anfr": "-", "fr2": "--", "fr3": "-.", "fr4": ":", "fr5": ":", "frm": ":", "simgd": "-", "jare1": "-", "jare.5": "--", "jare-.5": "--", "jare.2": "-.", "jare.1": ":", "cgd": "-", "gdn": "-", "cn": "-."}  # "#AA6622"
line_color_dict = {"fr": "#FF0000", "nfr": "#FF0000", "anfr": "#FF0000", "fr2": "#00FFFF", "fr3": "#FF00FF", "fr4": "#8800FF", "fr5": "#0000FF", "frm": "#FFAA00", "simgd": "#66CC22", "jare1": "#666622", "jare.5": "#996622", "jare-.5": "#226699", "jare.2": "#CC6622", "jare.1": "#FF6622", "cgd": "#00FFFF", "gdn": "#0022AA", "cn": "#22AA00", "grad_x": "#000000", "grad_x2": "#FF00FF", "grad_y": "#00FF00", "grad_y2": "#00FFFF"}
method_name_dict = {"fr": "FR", "nfr": "Newton FR", "anfr": "Absolute Newton FR", "fr2": "AltFR", "fr3": "BothFR", "fr4": "AdaptFR", "fr5": "NewFR", "frm": "FR-maximin", "simgd": "SimGD", "jare1": "JARE 1", "jare.5": "JARE 0.5", "jare-.5": "JARE -0.5", "jare.2": "JARE 0.2", "jare.1": "JARE 0.1", "cgd": "CGD", "gdn": "Gradient Descent Newton", "cn": "Complete Newton"}
# optimizer_dict = {"fr": follow, "fr2": follow_alt, "fr3": follow_both, "fr4": follow_adapt, "fr5": follow_new, "frm": follow_maximin, "simgd": gda, "jare1": jare, "jare.5": jare, "jare-.5": jare, "jare.2": jare, "jare.1": jare, "cgd": cgd}
gamma_dict = {"fr": 1., "nfr": 1., "anfr": 1., "fr2": 1., "fr3": 1., "fr4": 1., "fr5": 1., "frm": 1., "simgd": 1., "jare1": 1., "jare.5": 0.5, "jare-.5": -0.5, "jare.2": 0.2, "jare.1": 0.1, "cgd": 1., "gdn": 1., "cn": 1.}

offset_dir_dict = {"fr": [1, 0], "nfr": [1, -0.25], "anfr": [1, -0.125], "fr2": [1, -0.5], "fr3": [1, -1], "fr4": [0.5, -1], "fr5": [0, -1], "frm": [-0.5, -1], "simgd": [-1, -1], "jare1": [-1, 0], "jare.5": [-1, 0.5], "jare-.5": [1, -0.5], "jare.2": [-1, 1], "jare.1": [-0.5, 1], "cgd": [-1, -0.5], "gdn": [0.5, 1], "cn": [1, 1]}

for key, value in offset_dir_dict.items():
    value = np.array(value)
    offset_dir_dict[key] = value / np.linalg.norm(value)

grad_item_list = ["grad_x", "grad_x2", "grad_y", "grad_y2"]
optimizer_result_item_list = ["loc"] + grad_item_list

def Run_toy_example(func_id, method_list, lr=0.05, num_iter=None, iter_per_arrow=None, draw_arrows=False, plot_width=None, z_0=None, epsilon=epsilon, hessian_reg=epsilon2, theta=0, grad_noise=0, calculate_traj_stats=False, gamma=1):
    # Select target func_id
    target = Get_function(func_id, theta)
    root_dir = os.path.join(Toy_experiment_folder, f"f{func_id}")

    if func_id == 1: # (0,0) is local minimax and global minimax
        if z_0 is None:
            z_0 = [5., 7.]
        if num_iter is None:
            num_iter = 100
        if iter_per_arrow is None:
            iter_per_arrow = 2
        if plot_width is None:
            plot_width = 14          # Set range of the plot
    elif func_id == 2: # (0,0) is not local minimax and not global minimax
        if z_0 is None:
            z_0 = [6., 5.]
        if num_iter is None:
            num_iter = 100
        if iter_per_arrow is None:
            iter_per_arrow = 2
        if plot_width is None:
            plot_width = 14
    elif func_id == 3: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [7., 5.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = 20
        if plot_width is None:
            plot_width = 10
    elif func_id == 4: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [0.5, 0.5]
        if num_iter is None:
            num_iter = 100
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 20
        if plot_width is None:
            plot_width = 5
    elif func_id == 5: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [7., 5.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 20
        if plot_width is None:
            plot_width = 10
    elif func_id == 6: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [1., 1.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 20
        if plot_width is None:
            plot_width = 6
    elif func_id == 7: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [7., 5.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    elif func_id == 8: # (0,0) is local minimax
        if z_0 is None:
            z_0 = [3., 3.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    elif func_id in [9, 11]: # (0,0.5) is local minimax
        if z_0 is None:
            z_0 = [3., 3.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    elif func_id in [10, 14]: # (0,0.5) is local minimax
        if z_0 is None:
            z_0 = [3., 3.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    elif func_id in [12, 13]: # (0,0.5) is local minimax
        if z_0 is None:
            z_0 = [3., 3.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    elif func_id in [15]: # (0,0.5) is local minimax
        if z_0 is None:
            z_0 = [3., 3.]
        if num_iter is None:
            num_iter = 1000
        if iter_per_arrow is None:
            iter_per_arrow = num_iter / 5
        if plot_width is None:
            plot_width = 10
    else:
        NotImplementedError

    z_0 = np.array(z_0)

    result_dict_dict = {}
    result_dict_dict["lr"] = lr
    result_dict_dict["func_id"] = func_id
    result_dict_dict["z_0"] = z_0
    result_dict_dict["num_iter"] = num_iter
    result_dict_dict["iter_per_arrow"] = iter_per_arrow
    result_dict_dict["plot_width"] = plot_width
    result_dict_dict["root_dir"] = root_dir
    result_dict_dict["draw_arrows"] = draw_arrows
    result_dict_dict["target"] = target
    result_dict_dict["result"] = {}
    result_dict_dict["epsilon"] = epsilon
    result_dict_dict["hessian_reg"] = hessian_reg
    result_dict_dict["gamma"] = gamma
    result_dict_dict["theta"] = theta
    result_dict_dict["grad_noise"] = grad_noise

    for method in method_list:
        print("[Training] method", method)
        result_dict_dict["result"][method] = {}
        result_dict_dict["result"][method]["loc"], \
        result_dict_dict["result"][method]["grad_x"], \
        result_dict_dict["result"][method]["grad_x2"], \
        result_dict_dict["result"][method]["grad_y"], \
        result_dict_dict["result"][method]["grad_y2"], \
        result_dict_dict["result"][method]["adapt"], \
        result_dict_dict["result"][method]["off_itr"], \
        result_dict_dict["result"][method]["minimax_c1_list"], \
        result_dict_dict["result"][method]["minimax_c2_list"], \
        result_dict_dict["result"][method]["traj_angle_list"], \
        result_dict_dict["result"][method]["ref_angle_list"], \
        result_dict_dict["result"][method]["ref_dist_list"] \
            = Get_trajectory(method, z_0, target, lr=lr, num_iter=num_iter, epsilon=epsilon, hessian_reg=hessian_reg, gamma=gamma_dict[method], grad_noise=grad_noise, calculate_traj_stats=calculate_traj_stats)

    return result_dict_dict


def Plot_toy_example_list(result_dict_dict_list, iter_per_arrow=None, xlim=None, ylim=None, plot_width=None, iter_per_text=3, img_type="svg", dpi=400):
    result_dict_dict = result_dict_dict_list[0]
    func_id = result_dict_dict["func_id"]

    if iter_per_arrow is None:
        iter_per_arrow = result_dict_dict["iter_per_arrow"]
    num_iter = result_dict_dict["num_iter"]

    if plot_width is None:
        plot_width = result_dict_dict["plot_width"]

    lr = result_dict_dict["lr"]
    target = result_dict_dict["target"]
    epsilon = result_dict_dict["epsilon"]
    hessian_reg = result_dict_dict["hessian_reg"]
    gamma = result_dict_dict["gamma"]
    theta = result_dict_dict["theta"]
    grad_noise = result_dict_dict["grad_noise"]

    print("target", target)

    # Plot trajectory with contour
    figsize = 6
    plt.rcParams.update({'font.size': figsize * 1.2})
    def_colors = (plt.rcParams['axes.prop_cycle'].by_key()['color'])

    all_method_list = list(result_dict_dict["result"].keys())
    for method in all_method_list:
        print(f"Plotting method {method}")

        fig = plt.figure(figsize=(figsize, figsize))
        axes = plt.subplot2grid((1, 1), (0, 0), rowspan=3)
        if xlim is None:
            xlim = [-plot_width, plot_width]
        if ylim is None:
            ylim = [-plot_width, plot_width]

        axes.set_xlim(xlim[0], xlim[1])
        axes.set_ylim(ylim[0], ylim[1])
        len_scale = (xlim[1] + ylim[1] - xlim[0] - ylim[0]) / 8
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        x1 = np.arange(xlim[0], xlim[1], np.min([(xlim[1] - xlim[0]) / 200., 0.1]))
        y1 = np.arange(ylim[0], ylim[1], np.min([(ylim[1] - ylim[0]) / 200., 0.1]))

        X, Y = np.meshgrid(x1, y1)
        Z = np.zeros_like(X)
        for i in range(len(x1)):
            for j in range(len(y1)):
                z = np.array([x1[i], y1[j]])
                Z[j][i] = target(z)
        cset1 = axes.contourf(X, Y, Z, 30, cmap=plt.cm.gray)
        fig.colorbar(cset1, ax=axes)

        lw = figsize / 24
        text_size = figsize / 8
        offset = [0.1 * plot_width / 10.] * 5

        """ Init points """
        for i, result_dict_dict in enumerate(result_dict_dict_list):
            label = "Init points" if i == 0 else None
            z_0 = result_dict_dict["z_0"]
            axes.scatter(z_0[0], z_0[1], marker="D", color="#FFFF00", s=figsize * 5 / 6, linewidth=figsize / 4, label=label, zorder=4)
            axes.text(z_0[0], z_0[1], f"({z_0[0]:.1f},{z_0[1]:.1f})", fontsize=text_size)

        """ Training trajectory """
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        gradient_linspace = np.linspace(0, 1, num_iter)
        gradient = np.vstack((gradient_linspace, gradient_linspace))
        if func_id in [15]:
            color_line_color_map = plt.get_cmap("OrRd")
        else:
            color_line_color_map = plt.get_cmap("jet")
        cax.imshow(gradient, aspect='auto', cmap=color_line_color_map)
        cax.set_yticklabels([])
        axes.xaxis.tick_top()
        axes.xaxis.set_label_position('top')

        for i, result_dict_dict in enumerate(result_dict_dict_list):
            curr_result = result_dict_dict["result"][method]
            result_dict_dict["result"][method]["line"] \
                = colorline(curr_result["loc"][:, 0], \
                            curr_result["loc"][:, 1], \
                            gradient_linspace, \
                            cmap=color_line_color_map, \
                            linewidth=lw, alpha=0.3, ax=axes)

            """ Training iterations """
            prog_iter = len(curr_result[grad_item_list[0]]) // iter_per_text
            if prog_iter == 0:
                prog_iter += 1

            for i in range(len(curr_result[grad_item_list[0]])):
                if i % prog_iter == 0 or i % iter_per_arrow == 0:
                    current_loc = curr_result["loc"][i, :].copy()
                    text_loc_x = current_loc[0] + offset[0] * offset_dir_dict[method][0]
                    text_loc_y = current_loc[1] + offset[1] * offset_dir_dict[method][1]
                    if np.abs(current_loc[0]) < 1.1 * plot_width and np.abs(current_loc[1]) < 1.1 * plot_width:
                        axes.text(text_loc_x, text_loc_y, f"{i}", color="#000000", fontsize=text_size)
                        axes.scatter(current_loc[0], current_loc[1], marker="x", c="#000000", s=figsize / 6, alpha=0.5)

        """ End Point """
        black_label_added = False
        diverge_label_added = False
        converge_label_added = False
        for i, result_dict_dict in enumerate(result_dict_dict_list):
            diverge = False
            converge = False
            if np.linalg.norm(curr_result["loc"][-2, :] - curr_result["loc"][-1, :]) < epsilon:
                converge = True

            curr_result = result_dict_dict["result"][method]
            last_index = 1
            z_last = curr_result["loc"][-last_index, :]
            last_iter = len(curr_result["loc"]) - last_index - 1

            while ((not (z_last[0] > xlim[0] and z_last[0] < xlim[1] and z_last[1] > ylim[0] and z_last[1] < ylim[1])) or any(np.isnan(z_last))) and last_index < len(curr_result["loc"]):
                last_index += 1
                z_last = curr_result["loc"][-last_index, :]
                diverge = True

            label = None
            if diverge:
                end_point_color = "#FFFFFF"
                last_iter_symbol = ""
                if not diverge_label_added:
                    diverge_label_added = True
                    label = "Points before diverging"
            elif converge:
                end_point_color = "#00FF00"
                last_iter_symbol = "*"
                if not converge_label_added:
                    converge_label_added = True
                    label = "Converged points"
            else:
                end_point_color = "#000000"
                last_iter_symbol = ""
                if not black_label_added:
                    black_label_added = True
                    label = "Final points"

            # print(f"End point {z_last} plotted")
            axes.scatter(z_last[0], z_last[1], marker="X", color=end_point_color, s=figsize / 3, linewidth=figsize / 8, zorder=5, label=label, alpha=0.7)
            z_0 = result_dict_dict["z_0"]
            axes.scatter(z_0[0], z_0[1], marker="d", color=end_point_color, s=figsize / 2, linewidth=figsize / 8, zorder=5, label=None, alpha=1)

            axes.text(z_last[0] + len_scale * offset[0] * offset_dir_dict[method][0], \
                      z_last[1] + len_scale * offset[1] * offset_dir_dict[method][1], \
                      f"{last_iter + 1}{last_iter_symbol}({z_last[0]:.3f},{z_last[1]:.3f})",
                      color="#000000", fontsize=1.5 * text_size, zorder=6)

        gadgets = []
        gadgets_texts = []

        if func_id == 1:
            tr = theta * np.pi / 180
            ridge_y_linspace = (2 * np.cos(tr) + np.sin(tr)) * x1 / \
                               (np.cos(tr) - 2 * np.sin(tr))
            ridge_ref_line, = axes.plot(x1, ridge_y_linspace, color="#000000", label="ridge")
            gadgets.append(ridge_ref_line)
            gadgets_texts.append("ridge")

        axes.legend(loc="lower right")

        if func_id == 1:
            axes.scatter(0, 0, marker="x", color="#000000")
            function_text = r"$f(x, y) = -3x^{\dagger 2} -y^{\dagger 2} + 4x^\dagger y^\dagger$"
            x_text = r"$x^\dagger = x \cos \theta + y \sin \theta$"
            y_text = r"$y^\dagger = -x \sin \theta + y \cos \theta$"
            notes = "(0,0) is local minimax\nand global minimax"
            axes.text(3, 0, f"{function_text}\n{x_text}\n{y_text}\n{notes}\nlr = {lr}\nniter = {num_iter}\niter per arrow = {iter_per_arrow}")
        elif func_id == 2:
            axes.set_xlim(-12, 12)
            axes.set_ylim(-12, 12)
            function_text = r"$f = 3x^2 + y^2 + 4xy$"
            notes = "(0,0) is not local minimax\nand not global minimax"
            axes.scatter(0, 0, marker="x", color="#000000")
            axes.text(1, 2, f"{function_text}\n{notes}\nlr = {lr}\nniter = {num_iter}\niter per arrow = {iter_per_arrow}")
        elif func_id == 3:
            axes.set_xlim(-4, 8)
            axes.set_ylim(-4, 8)
            function_text = r"$f = (0.4x^2 - 0.1(y-3x + 0.05x^3)^2 - 0.01y^4)e^{-0.01(x^2 + y^2)}$"
            notes = "(0,0) is local minimax"
            axes.scatter(0, 0, marker="x", color="#000000")
            axes.text(-2, 4, f"{function_text}\n{notes}\nlr = {lr}\nniter = {num_iter}\niter per arrow = {iter_per_arrow}")
        elif func_id == 4:
            axes.set_xlim(-2, 3)
            axes.set_ylim(-2, 3)
            axes.scatter(0, 0, marker="x", color="#000000")
        elif func_id == 5:
            axes.set_xlim(-3, 3)
            axes.set_ylim(-3, 3)
            axes.scatter(0, 0, marker="x", color="#000000")
        elif func_id == 6:
            axes.set_xlim(-3, 3)
            axes.set_ylim(-3, 3)
            axes.scatter(0, 0, marker="x", color="#000000")
        elif func_id == 15:
            axes.scatter(0, np.pi / 2, marker="x", color="#00FF00")
            axes.text(0, np.pi / 2, f"local minimax")
            axes.scatter(0, -np.pi / 2, marker="x", color="#FF0000")
            axes.text(0, -np.pi / 2, f"local minimum")

        axes.add_artist(Circle((0, 0), epsilon, color="#000000", fill=False))
        axes.set_aspect('equal')
        theta_symbol = r"$\theta$"
        deg_symbol = r"$^{\circ}$"
        axes.set_title(f"{method_name_dict[method]}: lr={lr}, {theta_symbol}={theta}{deg_symbol}, hessian_reg={hessian_reg}, grad_noise={grad_noise}")
        # print("cax.get_xticklabels()", list(cax.get_xticklabels()))
        # cax.set_xticklabels([val + 1 for val in cax.get_xticklabels()])

        num_ticks = 10
        tick_loc_list = [np.round((i + 1) * num_iter / num_ticks, 0).astype(int) - 1 for i in range(0, num_ticks)]
        tick_loc_list = list(set(tick_loc_list))[:num_iter]
        tick_label_list = [tick_loc + 1 for tick_loc in tick_loc_list]

        cax.set_xticklabels(tick_label_list)
        cax.set_xticks(tick_loc_list)
        cax.yaxis.set_visible(False)
        cax.set_xlabel("Iteration")

        os.makedirs(Toy_experiment_folder, exist_ok=True)
        plt.tight_layout()
        if img_type == "svg":
            plt.savefig(os.path.join(Toy_experiment_folder, f"f{func_id}_{method}_lr{lr}_it{num_iter}_gamma{gamma}_theta{theta}_xlim{xlim}_ylim{ylim}_eps{epsilon}_hr{hessian_reg}_gn{grad_noise}_{time.strftime('%H%M%S')}.svg"))
        elif img_type == "png":
            plt.savefig(os.path.join(Toy_experiment_folder, f"f{func_id}_{method}_lr{lr}_it{num_iter}_gamma{gamma}_theta{theta}_xlim{xlim}_ylim{ylim}_eps{epsilon}_hr{hessian_reg}_gn{grad_noise}_{time.strftime('%H%M%S')}.png"), dpi=dpi)
        else:
            raise NotImplementedError

def Get_result_dict_dict_list(func_id, method_list, z_0_x_list, z_0_y_list, lr=0.05, grad_noise=0, num_iter=1000, theta_list=[0], skip_num=0, hessian_reg=0, critical_point=np.array([0., 0.]), gamma=0):
    X, Y = np.meshgrid(z_0_x_list, z_0_y_list)
    z_0_list = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    skip_counter = 1

    exp_timer = Timer()
    result_dict_dict_list = []
    for theta in theta_list:
        for z_0 in z_0_list:
            print(f"theta {theta}, z_0 {z_0}, skip_counter {skip_counter}, time {exp_timer.Print()}", end="")
            if skip_counter <= skip_num:
                skip_counter += 1
                print(" - skipped")
                continue
            skip_counter += 1
            print("\n", end="")
            result_dict_dict = Run_toy_example(func_id=func_id, \
                                               method_list=method_list, \
                                               lr=lr, num_iter=num_iter, z_0=z_0 + critical_point, \
                                               theta=theta, grad_noise=grad_noise, hessian_reg=hessian_reg, gamma=gamma)
            result_dict_dict_list.append(result_dict_dict)

    return result_dict_dict_list

def Run_toy_training_init_points(func_id, method_list, z_0_x_list, z_0_y_list, lr=0.05, grad_noise=0, num_iter=1000, theta_list=[0], skip_num=0, hessian_reg=0, xlim=[-5, 5], ylim=[-5, 5], plot_width=10, critical_point=np.array([0., 0.]), crit=False, gamma=0):

    if crit:
        if func_id in [9, 11]:
            critical_point = np.array([4.60128108e-02, 4.77188214e-01])
        elif func_id in [12, 13]:
            critical_point = np.array([4.77188214e-01, 4.60128108e-02])

    result_dict_dict_list = Get_result_dict_dict_list(func_id, method_list, z_0_x_list, z_0_y_list, lr=lr, grad_noise=grad_noise, num_iter=num_iter, theta_list=theta_list, skip_num=skip_num, hessian_reg=hessian_reg, critical_point=critical_point, gamma=gamma)
    Plot_toy_example_list(result_dict_dict_list, xlim=xlim + critical_point[0], ylim=ylim + critical_point[1], plot_width=plot_width, img_type="svg")
    Plot_toy_example_list(result_dict_dict_list, xlim=xlim + critical_point[0], ylim=ylim + critical_point[1], plot_width=plot_width, img_type="png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func_id", type=int, default=1, help="func_id")
    parser.add_argument("--method_list", type=str, default="simgd", help="List of methods")
    parser.add_argument("--z0_x_l", type=float, default=-1, help="Lower x of init")
    parser.add_argument("--z0_x_u", type=float, default=1, help="Upper x of init")
    parser.add_argument("--z0_y_l", type=float, default=-1, help="Lower y of init")
    parser.add_argument("--z0_y_u", type=float, default=1, help="Upper y of init")
    parser.add_argument("--num_x", type=int, default=5, help="# of points in x direction")
    parser.add_argument("--num_y", type=int, default=5, help="# of points in y direction")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=1, help="Gamma for JARE")
    parser.add_argument("--num_iter", type=int, default=1000, help="# of iterations")
    parser.add_argument("--skip_num", type=int, default=0, help="# of init skipped")
    parser.add_argument("--xlim_l", type=float, default=-2, help="Lower x lim")
    parser.add_argument("--xlim_u", type=float, default=2, help="Upper x lim")
    parser.add_argument("--ylim_l", type=float, default=-2, help="Lower y lim")
    parser.add_argument("--ylim_u", type=float, default=2, help="Upper y lim")
    parser.add_argument("--plot_width", type=float, default=2.2, help="plot_width")
    parser.add_argument("--hessian_reg", type=float, default=1, help="Regularization of Hessian")
    parser.add_argument("--grad_noise", type=float, default=0, help="Standard deviation of gradient noise")
    parser.add_argument("--crit", action='store_true', help="Whether to center around critical point")
    args = parser.parse_args()


    timer = Timer()

    method_list = args.method_list.split(",")
    print("method_list", method_list)
    Run_toy_training_init_points(func_id=args.func_id, method_list=method_list, z_0_x_list=np.linspace(args.z0_x_l, args.z0_x_u, args.num_x), z_0_y_list=np.linspace(args.z0_y_l, args.z0_y_u, args.num_y), lr=args.lr, num_iter=args.num_iter, theta_list=[0], skip_num=args.skip_num, xlim=np.array([-args.plot_width, args.plot_width]), ylim=np.array([-args.plot_width, args.plot_width]), plot_width=args.plot_width, hessian_reg=args.hessian_reg, grad_noise=args.grad_noise, crit=args.crit, gamma=args.gamma)
    timer.Print()