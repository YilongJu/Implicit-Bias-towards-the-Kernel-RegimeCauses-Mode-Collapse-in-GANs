import numpy as np
import torch
import matplotlib.cm as cm

from ComputationalTools import *

def Get_BP_direction(weights_torch):
    return weights_torch.cpu().detach().numpy() * (1. / torch.norm(weights_torch, p=2, dim=1).cpu().detach().numpy()[:, np.newaxis])


def Get_BP_signed_distance(weights_torch, biases_torch):
    return -biases_torch.cpu().detach().numpy() / torch.norm(weights_torch, p=2, dim=1).cpu().detach().numpy()


def Get_BP_delta_slopes(hidden_layer_weights_torch, output_layer_weights_torch):
    return output_layer_weights_torch.cpu().detach().numpy() * torch.norm(hidden_layer_weights_torch, p=2, dim=1).cpu().detach().numpy()


def Get_BP_params(hidden_layer_weights_torch, hidden_layer_biases_torch, output_layer_weights_torch):
    directions = Get_BP_direction(hidden_layer_weights_torch)
    signed_distances = Get_BP_signed_distance(hidden_layer_weights_torch, hidden_layer_biases_torch)
    delta_slopes = Get_BP_delta_slopes(hidden_layer_weights_torch, output_layer_weights_torch)
    return directions, signed_distances, delta_slopes

def Get_perp_vector(weight):
    perp_vec = np.ones(2)
    perp_vec[1] = -weight[0] / (weight[1] + EPS_BUFFER)
    return Normalize(perp_vec)

def Get_2D_line_points(w, b, point_num=5, radius=None, plot_lim=None):
    """ Line points for line: wx + b = 0 """
    w = w.ravel()
    assert w.shape == (2,)

    perp = Get_perp_vector(w)
    start_point = np.array([0, -b / (w[1] + EPS_BUFFER)])
    if radius is not None:
        t_linspace = np.array([-1, 0, 1])[:, np.newaxis] * radius
    elif plot_lim is not None:
        """ Find intersections with bounding box """
        t_l_x = (-plot_lim - start_point[0]) / (perp[0] + EPS_BUFFER)
        t_l_y = (-plot_lim - start_point[1]) / (perp[1] + EPS_BUFFER)
        t_u_x = (plot_lim - start_point[0]) / (perp[0] + EPS_BUFFER)
        t_u_y = (plot_lim - start_point[1]) / (perp[1] + EPS_BUFFER)

        critical_points = np.array([t_l_x, t_l_y, 0, t_u_x, t_u_y])
        critical_points.sort()

        t_linspace = critical_points[:, np.newaxis]
    else:
        t_linspace = np.linspace(-10, 10, point_num)[:, np.newaxis]

    line_points = t_linspace @ perp[np.newaxis, :] + start_point
    if plot_lim is not None:
        """ Crop lines at bounding box """
        line_points = line_points[np.linalg.norm(line_points, axis=1, ord=np.inf) <= plot_lim * 1.1]

    return line_points

def Get_diverging_color(value, cmap_pos="Blues", cmap_neg="Reds", color_scale=15):
    cmap_name = cmap_pos if value >= 0 else cmap_neg
    return cm.get_cmap(cmap_name)(np.abs(value) * color_scale)


if __name__ == "__main__":
    a = np.array([1, 2])
    b = 3
    perp = Get_perp_vector(a)
    start_point = np.array([0, -b / a[1]])
    radius = 12
    t_linspace = np.array([-1, 0, 1])[:, np.newaxis] * radius
    line = t_linspace @ Get_perp_vector(a)[np.newaxis, :] + start_point
    print(Get_perp_vector(a)[np.newaxis, :])
    print(np.dot(Get_perp_vector(a), a))
    print(t_linspace @ Get_perp_vector(a)[np.newaxis, :])
    print(line)
    print(line)
    print(line @ a[:, np.newaxis] + b)
    print(Get_2D_line_points(a, b))

    line_points = Get_2D_line_points(w=a, b=b, plot_lim=6)
    print(line_points)
    print(line_points @ a[:, np.newaxis] + b)
