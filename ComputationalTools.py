import matplotlib.pyplot as plt


import numpy as np
import scipy.sparse.linalg
import time
import datetime
import glob
import os
import platform
import random

from scipy.stats import norm
from scipy.optimize import fsolve
import scipy.stats as st

from nngeometry.layercollection import LayerCollection
from nngeometry.generator import Jacobian
from nngeometry.object import FMatDense
import torch
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset

np.set_printoptions(precision=2)

EPS_BUFFER = 1e-12

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def Get_all_relu_layer_output_as_vector(full_model, data_test):
#     layer_output_dict = {}
#
#     def Get_layer_output(layer_name):
#         def hook(model, input, output):
#             layer_output_dict[layer_name] = output.detach()
#
#         return hook
#
#     relu_layer_idx_list = [int(name) for name, layer in full_model.main.named_modules() if isinstance(layer, torch.nn.ReLU)]
#     for relu_layer_idx in relu_layer_idx_list:
#         full_model.main[relu_layer_idx].register_forward_hook(Get_layer_output(f"main_{relu_layer_idx}"))
#
#     output_test = full_model(data_test)
#     relu_output_vec = torch.cat([layer_output_dict[relu_layer].view(-1, 1) for relu_layer in layer_output_dict], dim=0)
#     return relu_output_vec

def Get_NTK_using_nngeometry(model, out_dim, test_data, centering=True):
#     model = G
#     out_dim = args.x_dim
#     test_data = z_test_torch_no_grad
#     centering = True
    batch_size = test_data.shape[0]
    dataset_test = TensorDataset(test_data, torch.ones(batch_size).to(device))
    dataLoader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)
    jacobian_generator = Jacobian(layer_collection=LayerCollection.from_model(model),
                             model=model,
                             n_output=out_dim,
                             centering=centering)
    ntk_MatDense = FMatDense(jacobian_generator, examples=dataLoader_test)
    ntk_torch_tensor = ntk_MatDense.get_dense_tensor()
    ntk_torch_mat = ntk_torch_tensor.reshape(ntk_torch_tensor.shape[0] * ntk_torch_tensor.shape[1], -1)
    return ntk_torch_mat


def Effective_rank_torch(kernel_mat_torch, eps=1e-12, top_k=None, sparse_eigs=True):
#     kernel_mat_torch = ntk_centered
#     sparse_eigs = True
    if sparse_eigs:
        if top_k is None:
            top_k = np.min([100, kernel_mat_torch.shape[0]])
        kernel_mat_eigvals, _ = scipy.sparse.linalg.eigs(kernel_mat_torch.detach().cpu().numpy(), top_k)
        kernel_mat_torch_eigvals_modulus = np.absolute(kernel_mat_eigvals)
    else:
        kernel_mat_torch_eigvals, _ = torch.eig(kernel_mat_torch)
        kernel_mat_torch_eigvals_modulus = np.linalg.norm(kernel_mat_torch_eigvals.detach().cpu().numpy(), axis=1, ord=2)

    kernel_mat_torch_eigvals_modulus_normalized = kernel_mat_torch_eigvals_modulus / np.sum(kernel_mat_torch_eigvals_modulus)
    kernel_mat_torch_eigvals_modulus_normalized_entropy = -np.sum(kernel_mat_torch_eigvals_modulus_normalized * np.log(kernel_mat_torch_eigvals_modulus_normalized + eps))
    kernel_mat_effective_rank = np.exp(kernel_mat_torch_eigvals_modulus_normalized_entropy)
    return kernel_mat_effective_rank, kernel_mat_torch_eigvals_modulus_normalized


def Get_all_relu_layer_output_as_vector(full_model, data_test, get_pre_act=False):
    layer_output_dict = {}

    def Get_layer_output(layer_name):
        def hook(model, input, output):
            layer_output_dict[layer_name] = output.detach()

        return hook

    relu_layer_idx_list = [int(name) for name, layer in full_model.main.named_modules() if
                           isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.LeakyReLU)]
    #     print(relu_layer_idx_list)
    for relu_layer_idx in relu_layer_idx_list:
        if get_pre_act:
            layer_idx = relu_layer_idx - 2
            full_model.main[relu_layer_idx].register_forward_hook(Get_layer_output(f"main_{relu_layer_idx}"))
        else:
            layer_idx = relu_layer_idx

        if layer_idx < 0:
            layer_output_dict[f"main_{layer_idx}"] = data_test
        else:
            full_model.main[layer_idx].register_forward_hook(Get_layer_output(f"main_{layer_idx}"))

    output_test = full_model(data_test)

    if get_pre_act:
        for relu_layer_idx in relu_layer_idx_list:
            layer_output_dict[f"main_{relu_layer_idx - 1}"] = full_model.main[relu_layer_idx - 1](layer_output_dict[f"main_{relu_layer_idx - 2}"])
        relu_output_vec = torch.cat([layer_output_dict[f"main_{relu_layer_idx - 1}"].view(-1, 1) for relu_layer_idx in relu_layer_idx_list], dim=0)
    else:
        relu_output_vec = torch.cat([layer_output_dict[f"main_{relu_layer_idx}"].view(-1, 1) for relu_layer_idx in relu_layer_idx_list], dim=0)

    return relu_output_vec


def Get_relative_activation_pattern_change_from_relu_outputs(relu_output_vec_init, relu_output_vec_final, from_pre_act=False):
    if from_pre_act:
        relu_output_is_0_init = relu_output_vec_init < 0
        relu_output_is_0_final = relu_output_vec_final < 0
    else:
        relu_output_is_0_init = relu_output_vec_init == 0
        relu_output_is_0_final = relu_output_vec_final == 0

    act_pattern_change = relu_output_is_0_init != relu_output_is_0_final
    relative_act_pattern_change = torch.sum(act_pattern_change).item() / np.prod(relu_output_vec_init.shape)
    return relative_act_pattern_change


# def Get_relative_activation_pattern_change_from_relu_outputs(relu_output_vec_init, relu_output_vec_final):
#     relu_output_is_0_init = relu_output_vec_init == 0
#     relu_output_is_0_final = relu_output_vec_final == 0
#     act_pattern_change = relu_output_is_0_init != relu_output_is_0_final
#     relative_act_pattern_change = torch.sum(act_pattern_change).item() / np.prod(relu_output_vec_init.shape)
#     return relative_act_pattern_change

def KDE(x_range, y_range, point_list, weight_list=None, bw_method=None, n=100j):
    xmin, xmax = x_range[0], x_range[1]
    ymin, ymax = y_range[0], y_range[1]
    xx, yy = np.mgrid[xmin:xmax:n, ymin:ymax:n] # grid
    positions = np.vstack([xx.ravel(), yy.ravel()])
    x = point_list[:, 0]; y = point_list[:, 1] # data pints
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values, weights=weight_list, bw_method=bw_method)
    density_KDE = np.reshape(kernel(positions).T, xx.shape) # density on grid
    return xx, yy, density_KDE


def Flatten_list(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return flat_list

def Get_output_by_batch(net, input, batch_size=4):
    total_data_number = input.shape[0]
    starting_data_idx = 0
    output_list = []
    while starting_data_idx <= total_data_number - 1:
        ending_data_idx = np.minimum(starting_data_idx + batch_size, total_data_number)
        input_batch = input[starting_data_idx:ending_data_idx, ...]
        output_batch = net(input_batch)
        output_list.append(output_batch)
        starting_data_idx += batch_size

    output = torch.cat(output_list, dim=0)
    return output


def _Get_perturbed_output(net, input_expanded_torch, std_grad_approx, device):
    perturb_torch = torch.normal(torch.zeros_like(input_expanded_torch), std_grad_approx * torch.ones_like(input_expanded_torch)).to(device)
    input_expanded_perturbed_torch = input_expanded_torch + perturb_torch

    output_expanded_torch = Get_output_by_batch(net, input_expanded_torch)
    output_expanded_perturbed_torch = Get_output_by_batch(net, input_expanded_perturbed_torch)

    output_diff_torch = output_expanded_perturbed_torch - output_expanded_torch
    output_diff_flattened_torch = output_diff_torch.view(output_diff_torch.shape[0], -1)

    difference_norm_squared_vec = (torch.norm(output_diff_flattened_torch, p=2, dim=1) ** 2 / std_grad_approx ** 2)
    return difference_norm_squared_vec


def Estimate_Jacobian_norm(net, input_torch, n_grad_approx=20, std_grad_approx=1e-4, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), using_loop=False, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if using_loop:
        input_expanded_torch = torch.repeat_interleave(input_torch, repeats=1, dim=0)
        difference_norm_squared_vec_list = []
        for i in range(n_grad_approx):
            difference_norm_squared_vec = _Get_perturbed_output(net, input_expanded_torch, std_grad_approx, device).unsqueeze(1)
            difference_norm_squared_vec_list.append(difference_norm_squared_vec)

        difference_norm_squared_mat = torch.cat(difference_norm_squared_vec_list, dim=1)
    else:
        input_expanded_torch = torch.repeat_interleave(input_torch, repeats=n_grad_approx, dim=0)
        difference_norm_squared_vec = _Get_perturbed_output(net, input_expanded_torch, std_grad_approx, device)
        difference_norm_squared_mat = difference_norm_squared_vec.view(-1, n_grad_approx)

    jacobian_norm_est_np = torch.pow(torch.mean(difference_norm_squared_mat, dim=1), 0.5).detach().cpu().numpy()

    return jacobian_norm_est_np

# def Estimate_Jacobian_norm(net, input_torch, n_grad_approx=20, std_grad_approx=1e-4, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
#     input_expanded_torch = torch.repeat_interleave(input_torch, repeats=n_grad_approx, dim=0)
#     perturb_torch = torch.normal(torch.zeros_like(input_expanded_torch), std_grad_approx * torch.ones_like(input_expanded_torch)).to(device)
#     input_expanded_perturbed_torch = input_expanded_torch + perturb_torch
#
#     output_expanded_torch = Get_output_by_batch(net, input_expanded_torch)
#     output_expanded_perturbed_torch = Get_output_by_batch(net, input_expanded_perturbed_torch)
#
#     output_diff_torch = output_expanded_perturbed_torch - output_expanded_torch
#     output_diff_flattened_torch = output_diff_torch.view(output_diff_torch.shape[0], -1)
#
#     difference_norm_squared_vec = torch.norm(output_diff_flattened_torch, p=2, dim=1) ** 2 / std_grad_approx ** 2
#     difference_norm_squared_mat = difference_norm_squared_vec.view(-1, n_grad_approx)
#     jacobian_norm_est_np = torch.pow(torch.mean(difference_norm_squared_mat, dim=1), 0.5).detach().cpu().numpy()
#
#     return jacobian_norm_est_np


def Get_feature_mat_for_a_batch_of_input_wrt_params(model, input_torch, out_dim):
    batch_size = input_torch.shape[0]
    model_params_list = list(params for params in model.parameters())
    num_params = np.sum([np.prod(params.shape) for params in model_params_list])
    feature_mat = torch.zeros(size=(batch_size * out_dim, num_params))

    for i in range(batch_size):
        input_i = input_torch[i, ...].unsqueeze(0)
        out_i = model(input_i).view(1, -1)
        for j in range(out_dim):
            input_i_out_dim_j_grad_list = torch.autograd.grad(outputs=out_i[:, j], inputs=model_params_list, create_graph=True, retain_graph=True)
            input_i_out_dim_j_grad_flat = torch.cat([grad.view(-1, 1) for grad in input_i_out_dim_j_grad_list])
            feature_mat[i + j * batch_size, :] = input_i_out_dim_j_grad_flat.squeeze(1)

    return feature_mat


def Get_list_of_Jacobian_for_a_batch_of_input(model, input_torch):
    out_jacobian_list = []
    batch_size = input_torch.shape[0]
    for i in range(batch_size):
        input_i = input_torch[i, ...].unsqueeze(0) # i-th data point
        out_i = model(input_i).view(1, -1) # flattened output for i-th data point
        out_i_grad_flat_list = []
        for j in range(out_i.shape[1]):
            out_i_grad_j = torch.autograd.grad(outputs=out_i[:, j], inputs=input_i, retain_graph=True)[0] #
            out_i_grad_j_flat = out_i_grad_j.view(1, -1)
            out_i_grad_flat_list.append(out_i_grad_j_flat)

        out_i_jacobian = torch.cat(out_i_grad_flat_list, 0)
        out_i_jacobian_flat = out_i_jacobian.view(1, -1)
        out_jacobian_list.append(out_i_jacobian_flat)

    return out_jacobian_list


def KL_divergence(p, q, eps=EPS_BUFFER):
    p = np.array(p)
    q = np.array(q)
    return np.sum(p * np.log(p / (q + eps) + eps))


def nan_to_num_torch(torch_tensor):
    if torch.isnan(torch_tensor).any():
        torch_tensor[torch_tensor != torch_tensor] = 0


def Get_latest_files(folder, latest_file_num=1, skip_num=0, delimiter="pickle"):
    file_list = glob.glob(os.path.join(folder, f"*.{delimiter}"))
    file_list.sort(key=os.path.getmtime)
    if platform.system() == "Darwin":
        print("Using MacOS.")
        path_delimiter = "/"
    elif platform.system() == "Linux":
        print("Using Linux.")
        path_delimiter = "/"
    else:
        print("Using Windows.")
        path_delimiter = "\\"

    task_list = [".".join(file.split(path_delimiter)[1].split(".")[:-1]) for file in file_list[-(latest_file_num + skip_num):]]
    task_list = task_list[:latest_file_num]

    return task_list


def Get_start_and_end_pos_for_worker(my_part, num_parts, base_start_pos, base_end_pos):
    base_num = (base_end_pos - base_start_pos) // num_parts
    remainder = (base_end_pos - base_start_pos) - base_num * num_parts
    start_pos = my_part * base_num + base_start_pos
    end_pos = (my_part + 1) * base_num + base_start_pos
    if my_part < remainder:
        start_pos += my_part
        end_pos += my_part + 1
    else:
        start_pos += remainder
        end_pos += remainder

    return start_pos, end_pos


def Rad_to_Deg(rad_var):
    if isinstance(rad_var, list):
        return [rad * 180. / np.pi for rad in rad_var]
    else:
        return rad_var * 180. / np.pi

def Now(time_format="%y%m%d-%H%M%S"):
    return datetime.datetime.now().strftime(time_format)

def Cossim(v1, v2, eps=EPS_BUFFER, output_vec=False):
    if output_vec:
        result = [np.dot(vv1, vv2) / (np.linalg.norm(vv1) * np.linalg.norm(vv2) + eps) for vv1, vv2 in zip(v1, v2)]
    else:
        v1 = v1.ravel()
        v2 = v2.ravel()
        result =  np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + eps)

    return np.clip(result, -1, 1)

def Normalize(vec, eps=EPS_BUFFER, ord=2):
    """ Normalize vector """
    vec = np.array(vec)
    return vec / (np.linalg.norm(vec, ord=ord) + eps)

def Shannon_entropy(prob_list, eps=EPS_BUFFER):
    return -np.sum(prob_list * np.log(prob_list + eps))

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def Print(self, print_time=True, msg=""):
        elasped_time = np.round(time.time() - self.start_time, 3)
        if print_time:
            print(f"elasped_time {elasped_time} {msg}")
        return elasped_time

def Normalize_range_pm1(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * 2. - 1.

class JacobianVectorProduct(scipy.sparse.linalg.LinearOperator):
    def __init__(self, grad, params, regularization=0):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)

        nparams = sum(p.numel() for p in params)
        self.shape = (nparams, self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params
        self.regularization = regularization

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        hv = autograd.grad(self.grad, self.params, v, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        if self.regularization != 0:
            hv = torch.cat(_hv) + self.regularization*v
        else:
            hv = torch.cat(_hv) 
        return hv.cpu()

    
class SchurComplement(scipy.sparse.linalg.LinearOperator):
    def __init__(self, A, B, C, D, tol_gmres=1e-6, precise=False):
        self.operator = [[A,B], [C,D]]
        self.shape = A.shape
        self.config = {'tol_gmres': tol_gmres}
        self.dtype = np.dtype('Float32')
        self.precise = precise
        
    def _matvec(self, v): 
        
        (A,B),(C,D) = self.operator

        u = C(v)
        
        if self.precise:
            w, status = scipy.sparse.linalg.gmres(D, u, tol=self.config['tol_gmres'], restart=D.shape[0])
            assert status == 0
        else:
            w, status = scipy.sparse.linalg.cg(D, u, maxiter=5)
        
        self.w = w

        p = A(v) - B(w)
        
        return p


class SchurComplement(scipy.sparse.linalg.LinearOperator):
    def __init__(self, A, B, C, D, tol_gmres=1e-6, precise=False, maxiter_cg=5):
        self.operator = [[A, B], [C, D]]
        self.shape = A.shape
        self.config = {'tol_gmres': tol_gmres}
        self.dtype = np.dtype('Float32')
        self.precise = precise
        self.maxiter_cg = maxiter_cg

    def _matvec(self, v):

        (A, B), (C, D) = self.operator

        u = C(v)

        if self.precise:
            w, status = scipy.sparse.linalg.gmres(D, u, tol=self.config['tol_gmres'], restart=D.shape[0])
            assert status == 0
        else:
            w, status = scipy.sparse.linalg.cg(D, u, maxiter=self.maxiter_cg, tol=1e-3)

        self.w = w

        p = A(v) - B(w)

        return p



#
# def Calculate_eig_vals(loss_list, param_list, regularization=0, tol_gmres=1e-6, k=3, precise=False):
#     G_loss, D_loss = loss_list
#     params_G, params_D = param_list
#     Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
#
#     # A = JacobianVectorProduct(f1, list(x1.parameters()))  # Hxx_f
#     # B = JacobianVectorProduct(f2, list(x1.parameters()))  # Hxy_g
#     # C = JacobianVectorProduct(f1, list(x2.parameters()))  # Hyx_f
#     # D = JacobianVectorProduct(f2, list(x2.parameters()))  # Hyy_g
#
#     AA, BB, CC, DD, JJ = build_game_jacobian([Dg, Dd], [G, D])
#     # DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)
#     DD_reg = JacobianVectorProduct(Dd, params_D, regularization)
#
#     calc_eigs = lambda F: np.hstack((scipy.sparse.linalg.eigs(F, k=k, which='SR')[0], scipy.sparse.linalg.eigs(F, k=k, which='LR')[0]))
#
#     A_eigs = calc_eigs(AA)
#     D_eigs = calc_eigs(DD)
#     D_reg_eigs = calc_eigs(DD_reg)
#     J_eigs = calc_eigs(JJ)
#
#     SC_reg = SchurComplement(AA, BB, CC, DD_reg, tol_gmres=tol_gmres, precise=precise)
#     SC_reg_eigs = calc_eigs(SC_reg)
#
#     return A_eigs, D_eigs, D_reg_eigs, J_eigs, SC_reg_eigs


class Mixture_of_Gaussian_Generator:
    def __init__(self, Pi, Mu, Sigma2, mu_g=0., sigma_g=1., seed=None):
        self.Pi = np.array(Pi)
        self.Mu = np.array(Mu)
        self.Sigma2 = np.array(Sigma2)
        self.mu_g = mu_g
        self.sigma_g = sigma_g

        self.seed = seed

    def Generate_numbers(self, n, Pi=None, Mu=None, Sigma2=None, seed=None):

        if Pi is None:
            Pi = self.Pi
        if Mu is None:
            Mu = self.Mu
        if Sigma2 is None:
            Sigma2 = self.Sigma2
        if seed is None:
            seed = self.seed

        if seed is not None:
            np.random.seed(seed)

        c_vec = np.random.uniform(size=n)
        Pi_cum = np.cumsum(Pi)[:-1]
        Pi_cum_aug = np.repeat(Pi_cum[:, np.newaxis], n, axis=1)
        c_idx_vec = np.sum(Pi_cum_aug < c_vec, axis=0)
        # print("c_idx_vec", c_idx_vec)
        return np.array([np.random.normal(loc=Mu[c_idx], scale=np.sqrt(Sigma2[c_idx])) for c_idx in c_idx_vec]), c_idx_vec


    def Mixed_Gaussian_PDF(self, x, Pi=None, Mu=None, Sigma2=None):
        x = np.array(x)

        if Pi is None:
            Pi = self.Pi
        if Mu is None:
            Mu = self.Mu
        if Sigma2 is None:
            Sigma2 = self.Sigma2

        MG_PDF_list = np.array([Pi[i] * norm.pdf(x, loc=Mu[i], scale=np.sqrt(Sigma2[i])) \
                                for i in range(len(Pi))])
        return np.sum(MG_PDF_list, axis=0)

    def Mixed_Gaussian_CDF(self, x, Pi=None, Mu=None, Sigma2=None):
        x = np.array(x)

        if Pi is None:
            Pi = self.Pi
        if Mu is None:
            Mu = self.Mu
        if Sigma2 is None:
            Sigma2 = self.Sigma2

        MG_CDF_list = np.array([Pi[i] * norm.cdf(x, loc=Mu[i], scale=np.sqrt(Sigma2[i])) \
                                for i in range(len(Pi))])
        return np.sum(MG_CDF_list, axis=0)

    def _Inverse_mixed_CDF(self, t, init_guess):
        func = lambda x: self.Mixed_Gaussian_CDF(x) - t
        if init_guess is None:
            x_init_guess = norm.ppf(t, loc=np.sum(self.Pi * self.Mu), \
                                    scale=np.sqrt(np.sum(self.Pi * self.Pi * self.Sigma2)))
        else:
            x_init_guess = init_guess
        x_solution = fsolve(func, x_init_guess)

        return x_solution

    def _Inverse_mixed_CDF_uniform(self, t, init_guess):
        func = lambda x: self.Mixed_Gaussian_CDF(x) - t
        if init_guess is None:
            x_init_guess = norm.ppf(t, loc=np.sum(self.Pi * self.Mu), \
                                    scale=np.sqrt(np.sum(self.Pi * self.Pi * self.Sigma2)))
        else:
            x_init_guess = init_guess
        x_solution = fsolve(func, x_init_guess)

        return x_solution

    def Solve_inverse_mixed_CDF(self, t, init_guess=None):
        if len(t) > 1:
            t_sol = np.zeros_like(t)
            for i in range(len(t)):
                t_sol[i] = self._Inverse_mixed_CDF(t[i], init_guess=init_guess)
        else:
            t_sol = self._Inverse_mixed_CDF(t, init_guess=init_guess)

        self.numerical_error = np.abs(self.Mixed_Gaussian_CDF(t_sol) - t)
        return t_sol, self.numerical_error

    def Solve_inverse_mixed_CDF_acc(self, t, precise_gt=True, notes="normal"):
        t = t.ravel()
        if precise_gt:
            # z_linspace = np.linspace(-3, 3, 101)
            num_inits = 31
            init_guess_linspace = np.linspace(np.min(t), np.max(t), num_inits)
            x_linspace_mat = np.zeros([num_inits, len(t)])
            num_error_mat = np.zeros([num_inits, len(t)])

            for i in range(num_inits):
                x_linspace_mat[i, :], num_error_mat[i, :] = \
                    self.Solve_inverse_mixed_CDF(t, init_guess_linspace[i])

            num_error_acc = np.min(num_error_mat, axis=0)
            x_linspace_acc = x_linspace_mat[np.argmin(num_error_mat, axis=0), \
                                            np.arange(0, len(t))]
        else:
            x_linspace_acc, num_error_acc = self.Solve_inverse_mixed_CDF(t)

        return x_linspace_acc, num_error_acc

    def Get_full_Str(self, check_length=False):
        Pi_config = np.array2string(self.Pi, precision=2, \
                                    separator='_', suppress_small=True)
        Mu_config = np.array2string(self.Mu, precision=2, \
                                    separator='_', suppress_small=True)
        Sigma2_config = np.array2string(self.Sigma2, precision=2, separator='_', suppress_small=True)

        if len(self.Pi) > 3 and check_length:
            return f"Pi[{self.Pi[0]}..{self.Pi[-1]}]-Mu[{self.Mu[0]}..{self.Mu[-1]}]-Sigma2[{self.Sigma2[0]}..{self.Sigma2[-1]}]"
        else:
            return f"Pi{Pi_config}-Mu{Mu_config}-Sigma2{Sigma2_config}"

    def __repr__(self):
        return self.Get_full_Str(check_length=True)

    def Get_bounding_box(self):
        ub_list = self.Mu + 3 * np.sqrt(self.Sigma2)
        lb_list = self.Mu - 3 * np.sqrt(self.Sigma2)
        return np.array([np.min(lb_list), np.max(ub_list)])

if __name__ == "__main__":

    Pi = [0.5, 0.5]
    Mu = [-7.5, 7.5]
    Sigma2 = [1., 1.]
    MG = Mixture_of_Gaussian_Generator(Pi, Mu, Sigma2, seed=1)
    data_samples, data_labels = MG.Generate_numbers(5)

    print(data_samples)
    print(data_labels)
    print(MG.Get_bounding_box())
    #
    # plt.hist(data_samples, bins=100)
    # plt.savefig("MG_hist.png", dpi=400)
    # plt.show()
