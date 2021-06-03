import time
import math
import torch
import torch.autograd as autograd

from .cgd_utils import zero_grad, general_conjugate_gradient, Hvp_vec
from ComputationalTools import *
import scipy.sparse.linalg as ssla
import time

from hessian import hessian
# git clone https://github.com/mariogeiger/hessian.git

class FR(object):
    def __init__(self, params_x, params_y, lr_x=1e-3, lr_y=1e-3, eps=1e-8, beta=0.999, tol=1e-12, atol=1e-20, device=torch.device('cpu'), follow_x=False, follow_y=False, collect_info=True, adapt_x=False, adapt_y=False, hessian_reg_x=0.0, hessian_reg_y=0.0, cg_maxiter=5, maximin=False, calculate_eig_vals=False, eig_vals_num=3, tol_gmres=1e-6, precise=False, use_momentum=False, momentum=0.5, gamma_x=0.99999, gamma_y=0.9999999, newton_y=False, cgd=False, solve_x=False, zeta=1, rmsprop_init_1=False):
        self.params_x = params_x
        self.params_y = params_y
        self.lr_x_0 = lr_x
        self.lr_y_0 = lr_y
        self.rmsprop_init_1 = rmsprop_init_1
        self.state = {'lr_x': lr_x, 'lr_y': lr_y,
                      'eps': eps, 'follow_x': follow_x, 'follow_y': follow_y, 'newton_y': newton_y,
                      'tol': tol, 'atol': atol, "solve_x": solve_x, "zeta": zeta,
                      'beta': beta, 'step': 0, 'momentum': momentum,
                      'gamma_x': gamma_x, 'gamma_y': gamma_y,
                      'sqrt_lr_x_CDx_g_old': None, 'sqrt_lr_y_CDy_g_old': None,  # start point of CG
                      'sq_exp_avg_x': None, 'sq_exp_avg_y': None,
                      "momentum_x": None, "momentum_y": None}  # save last update
        self.info = {'grad_raw_norm_x': None, 'grad_raw_norm_y': None,
                     'grad_corr_norm_x': None, 'grad_corr_norm_y': None,
                     'update_tot_norm_x': None, 'update_tot_norm_y': None,
                     'training_time': 0, 'gc_time': 0, 'iter_num': 0,
                     'lr_x_actual': None, 'lr_y_actual': None}
        # , 'eig_vals_Hxx_f': None, 'eig_vals_Hyy_g': None, 'eig_vals_Hxx_f_reg': None, 'eig_vals_Hyy_g_reg': None, 'eig_vals_J': None, 'eig_vals_Hxx_f_Schur': None, 'eig_vals_Hyy_g_Schur': None, 'eig_calculation_time': 0}

        self.device = device
        self.collect_info = collect_info
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y

        self.follow_x = follow_x
        self.follow_y = follow_y
        self.newton_y = newton_y

        self.hessian_reg_x = hessian_reg_x
        self.hessian_reg_y = hessian_reg_y

        self.cg_maxiter = cg_maxiter
        self.maximin = maximin

        self.culmulative_training_time = 0.0
        self.culmulative_grad_corr_time = 0.0
        self.eig_calculation_time = 0.0

        self.calculate_eig_vals = calculate_eig_vals
        self.eig_vals_num = eig_vals_num
        self.tol_gmres = tol_gmres
        self.precise = precise

        self.update_x = None
        self.update_y = None

        self.use_momentum = use_momentum
        self.cgd = cgd

    def zero_grad(self):
        zero_grad(self.params_x)
        zero_grad(self.params_y)

    def get_info(self):
        if self.info['training_time'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info['grad_raw_norm_x'], self.info['grad_raw_norm_y'], self.info['grad_corr_norm_x'], self.info['grad_corr_norm_y'], self.info['update_tot_norm_x'], self.info['update_tot_norm_y'], self.info['training_time'], self.info['gc_time'], self.info['iter_num'], self.info['lr_x_actual'], self.info['lr_y_actual']
            # , self.info['eig_vals_Hxx_f'], self.info['eig_vals_Hyy_g'], self.info['eig_vals_Hxx_f_reg'], self.info['eig_vals_Hyy_g_reg'], self.info['eig_vals_J'], self.info['eig_vals_Hxx_f_Schur'], self.info['eig_vals_Hyy_g_Schur'], self.info['eig_calculation_time']

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_x, lr_y):
        self.state.update({'lr_x': lr_x, 'lr_y': lr_y})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_x, lr_y))

    def backward(self, loss_G, loss_D, lr_x=None, lr_y=None):
        if lr_x is None:
            lr_x = self.state['lr_x']
        if lr_y is None:
            lr_y = self.state['lr_y']
        beta = self.state['beta']
        momentum = self.state['momentum']
        gamma_x = self.state['gamma_x']
        gamma_y = self.state['gamma_y']
        zeta = self.state['zeta']

        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        show_num = 6
        calculate_hessian = False
        verbose = False # True, False

        st = time.time()
        Dx_f = autograd.grad(loss_G, self.params_x, create_graph=True, retain_graph=True)
        Dy_g = autograd.grad(loss_D, self.params_y, create_graph=True, retain_graph=True)


        Dx_f_flat = torch.cat([_.flatten() for _ in Dx_f]).view(-1, 1)
        Dy_g_flat = torch.cat([_.flatten() for _ in Dy_g]).view(-1, 1)

        Dx_f_flat_dt = Dx_f_flat.clone().detach()
        Dy_g_flat_dt = Dy_g_flat.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_x']
        sq_avg_y = self.state['sq_exp_avg_y']
        momentum_x = self.state['momentum_x']
        momentum_y = self.state['momentum_y']

        lr_x *= gamma_x ** (time_step - 1)
        lr_y *= gamma_y ** (time_step - 1)

        # addcmul: out = input + value × tensor1 × tensor2
        # A.addcmul_(k, v1, v2)
        # A += k * v1 * v2
        # sq_avg <- beta * sq_avg + (1 - beta) * Dy_g_flat_dt * Dy_g_flat_dt
        if self.adapt_x and (not self.use_momentum):
            if self.rmsprop_init_1:
                sq_avg_x = torch.ones_like(Dx_f_flat_dt, requires_grad=False) if sq_avg_x is None else sq_avg_x
                # sq_avg_x = Dx_f_flat_dt ** 2 if sq_avg_x is None else sq_avg_x
                bias_correction = 1
            else:
                sq_avg_x = torch.zeros_like(Dx_f_flat_dt, requires_grad=False) if sq_avg_x is None else sq_avg_x
                bias_correction = 1 - beta ** time_step

            # print(f"[time_step {time_step}] bias_correction", bias_correction)
            # print("sq_avg_x", sq_avg_x[:10, :])
            sq_avg_x.mul_(beta).addcmul_(1 - beta, Dx_f_flat_dt, Dx_f_flat_dt)
            lr_x = math.sqrt(bias_correction) ** zeta * lr_x / (sq_avg_x.sqrt().add(eps)) ** zeta
            # print(f"[zeta = {zeta}] lr_x_0 = {self.lr_x_0}, lr_x = {lr_x[:10, :]}\nlr_x_1 = {lr_x_1[:10, :]}")
            # print("math.sqrt(bias_correction) ** zeta", math.sqrt(bias_correction) ** zeta)
            # print("(sq_avg_x.sqrt().add(eps)) ** zeta", (sq_avg_x.sqrt().add(eps)) ** zeta)

        if self.adapt_y and (not self.use_momentum):
            if self.rmsprop_init_1:
                sq_avg_y = torch.ones_like(Dy_g_flat_dt, requires_grad=False) if sq_avg_y is None else sq_avg_y
                # sq_avg_y = Dy_g_flat_dt ** 2 if sq_avg_y is None else sq_avg_y
                bias_correction = 1
            else:
                sq_avg_y = torch.zeros_like(Dy_g_flat_dt, requires_grad=False) if sq_avg_y is None else sq_avg_y
                bias_correction = 1 - beta ** time_step

            sq_avg_y.mul_(beta).addcmul_(1 - beta, Dy_g_flat_dt, Dy_g_flat_dt)
            lr_y = math.sqrt(bias_correction) ** zeta * lr_y / (sq_avg_y.sqrt().add(eps)) ** zeta

        # print("lr_x", lr_x)
        # print("lr_y", lr_y)
        # TODO: Add momentum to Dx_f_flat_dt
        # Dx_f_moment = beta2 * Dx_f_moment + (1 - beta2) * Dx_f_flat_dt
        # ...
        # Dx_f_moment = Dx_f_flat_dt
        # Dy_g_moment = Dy_g_flat_dt
        # update_x_raw = torch.mul(lr_x, Dx_f_flat_dt)
        # update_y_raw = torch.mul(lr_y, Dy_g_flat_dt)

        update_x = lr_x * Dx_f_flat # scaled_grad_x
        update_y = lr_y * Dy_g_flat # scaled_grad_y

        gc_st = time.time()
        update_corr_x = torch.zeros_like(Dx_f_flat)
        update_corr_y = torch.zeros_like(Dy_g_flat)

        if self.cgd:
            Dy_g_ravel = Dy_g_flat.view(-1)
            Dy_g_ravel_dt = Dy_g_ravel.clone().detach()
            Dx_f_ravel = Dx_f_flat.view(-1)
            Dx_f_ravel_dt = Dx_f_ravel.clone().detach()

            lr_y_Hxy_g_Dy_g = Hvp_vec(Dy_g_ravel, self.params_x, update_y.view(-1), retain_graph=True) # hvp_x_vec
            lr_x_Hyx_f_Dx_f = Hvp_vec(Dx_f_ravel, self.params_y, update_x.view(-1), retain_graph=True) # hvp_y_vec
            # print(f"Dy_g_ravel.shape {Dy_g_ravel.shape}, update_y {update_y.shape}")

            p_x = -Dx_f_ravel_dt - lr_y_Hxy_g_Dy_g # -(Dx_f + lr_y * Hxy_g_Dy_g)
            p_y = Dy_g_ravel_dt + lr_x_Hyx_f_Dx_f # Dy_g + lr_x * Hyx_g_Dx_g

            # print(f"lr_y_Hxy_g_Dy_g.shape {lr_y_Hxy_g_Dy_g.shape}, p_x {p_x.shape}")
            if isinstance(lr_x, float):
                lr_x_tmp = lr_x
            else:
                lr_x_tmp = lr_x.view(-1)
            if isinstance(lr_x, float):
                lr_y_tmp = lr_y
            else:
                lr_y_tmp = lr_y.view(-1)

            lr_x_torch = torch.ones_like(Dx_f_ravel_dt) * lr_x_tmp
            lr_y_torch = torch.ones_like(Dy_g_ravel_dt) * lr_y_tmp
            if self.state['solve_x']:
                p_y = torch.sqrt(lr_y_torch) * p_y
                # p_y.mul_(lr_y.sqrt())
                # print("solve_x")
                sqrt_lr_y_cgd_Dy_g, iter_num = general_conjugate_gradient(grad_x=Dy_g_ravel, grad_y=-Dx_f_ravel, x_params=self.params_y, y_params=self.params_x, b=p_y, x=self.state['sqrt_lr_y_CDy_g_old'], tol=tol, atol=atol, lr_x=lr_y_torch, lr_y=lr_x_torch, device=self.device) # inv(I + sqrt(lr_y) * Hyx_g * lr_x * Hxy_g * sqrt(lr_y)) * (Dy_g + lr_x * Hyx_g_Dx_g) * sqrt(lr_y)
                # CDy_g
                sqrt_lr_y_CDy_g_old = sqrt_lr_y_cgd_Dy_g.detach_()
                update_raw_y = update_y.clone()
                update_y = sqrt_lr_y_cgd_Dy_g.mul(torch.sqrt(lr_y_torch)).view(-1, 1) # lr_y * inv(I + sqrt(lr_y) * Hyx_g * lr_x * Hxy_g * sqrt(lr_y)) * (Dy_g + lr_x * Hyx_g_Dx_g)
                                                               # lr_y * CDy_g
                update_corr_y = update_y - update_raw_y
                lr_y_Hxy_g_CDy_g = Hvp_vec(Dy_g_ravel, self.params_x, update_y.view(-1), retain_graph=True).detach_() # Hxy_g * (lr_y * CDy_g)
                update_corr_x = lr_x * lr_y_Hxy_g_CDy_g.view(-1, 1)
                update_x = update_x + update_corr_x
                sqrt_lr_x_CDx_g_old = -(Dx_f_ravel_dt + lr_y_Hxy_g_CDy_g).mul(torch.sqrt(lr_x_torch)) # -(Dx_f + lr_y_Hxy_g_CDy_g) * sqrt(lr_x)
            else:
                # print(f"p_x.shape {p_x.shape}, lr_x_torch {lr_x_torch.shape}")
                p_x = torch.sqrt(lr_x_torch) * p_x # -(Dx_f + lr_y * Hxy_g_Dy_g) * sqrt(lr_x)
                # print(f"p_x.shape {p_x.shape}")
                # p_x.mul_(lr_x.sqrt())
                # print("solve_y")
                cg_x, iter_num = general_conjugate_gradient(grad_x=-Dx_f_ravel, grad_y=Dy_g_ravel, x_params=self.params_x, y_params=self.params_y, b=p_x,  x=self.state['sqrt_lr_x_CDx_g_old'], tol=tol, atol=atol, lr_x=lr_x_torch, lr_y=lr_y_torch, device=self.device) # -inv(I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) * (Dx_f + lr_y * Hxy_g_Dy_g) * sqrt(lr_x)
                                # -CDx_f
                sqrt_lr_x_CDx_g_old = cg_x.detach_()
                update_x_raw = update_x.clone()
                update_x = -cg_x.mul(torch.sqrt(lr_x_torch)).view(-1, 1) # lr_x * inv(I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) * (Dx_f - lr_y * Hxy_f_Dy_g)
                                                  # lr_x * CDx_f
                update_corr_x = update_x - update_x_raw
                lr_x_Hyx_f_CDx_f = Hvp_vec(Dx_f_ravel, self.params_y, update_x.view(-1), retain_graph=True).detach_() # Hyx_f * (lr_x * CDx_f)
                update_corr_y = lr_y * lr_x_Hyx_f_CDx_f.view(-1, 1)
                update_y = update_y + update_corr_y
                sqrt_lr_y_CDy_g_old = (Dy_g_ravel_dt + lr_x_Hyx_f_CDx_f).mul(torch.sqrt(lr_y_torch)) # (Dy_g + lr_x_Hyx_f_CDx_f) * sqrt(lr_y)
                # print("type(sqrt_lr_y_CDy_g_old)", type(sqrt_lr_y_CDy_g_old))

            self.state.update({'sqrt_lr_x_CDx_g_old': sqrt_lr_x_CDx_g_old, 'sqrt_lr_y_CDy_g_old': sqrt_lr_y_CDy_g_old})
            self.state['solve_x'] = False if self.state['solve_x'] else True

        if self.follow_x:
            """ Follow_x """
            # print("Follow x")
            Dy_f = autograd.grad(loss_G, self.params_y, create_graph=True, retain_graph=True)
            Dy_f_flat = torch.cat([_.flatten() for _ in Dy_f]).view(-1, 1)
            Hyy_g_reg_lo = JacobianVectorProduct(Dy_g, self.params_y, self.hessian_reg_y)
            Dy_f_flat_np = np.nan_to_num(Dy_f_flat.cpu().detach().numpy())
            Hyy_g_reg_inv_Dy_f_flat, status = ssla.cg(Hyy_g_reg_lo, Dy_f_flat_np, maxiter=self.cg_maxiter)

            Hxy_g_lo = JacobianVectorProduct(Dy_g, self.params_x)
            Hxy_g_Hyy_g_reg_inv_Dy_f_flat = torch.Tensor(Hxy_g_lo(Hyy_g_reg_inv_Dy_f_flat)).view(-1, 1).to(self.device)
            update_corr_x = -lr_x * Hxy_g_Hyy_g_reg_inv_Dy_f_flat
            nan_to_num_torch(update_corr_x)


            if calculate_hessian:
                # print("Hyy_g_reg_inv_Dy_f_flat.shape", Hyy_g_reg_inv_Dy_f_flat.shape)
                # print(f"Dx_f {Dx_f[:show_num]}")
                # print(f"Dy_g {Dy_g[:show_num]}")
                print(f"self.params_x.shape {[param.shape for param in self.params_x]}")
                h_st = time.time()
                Hxx_f_exact = hessian(loss_G, self.params_x)
                print(f"[Time {time.time() - h_st:.3f}] Hxx_f_exact.shape {Hxx_f_exact.shape}\n{Hxx_f_exact[:show_num]}")
                print(f"self.params_y.shape {[param.shape for param in self.params_y]}")
                Hyy_g_exact = hessian(loss_D, self.params_y)
                print(f"[Time {time.time() - h_st:.3f}] Hyy_g_exact.shape {Hyy_g_exact.shape}\n{Hyy_g_exact[:show_num]}")
                h_st = time.time()
                Hyy_g_reg_pinv_exact = scipy.linalg.pinv(Hyy_g_exact.cpu().detach().numpy() + self.hessian_reg_y * np.identity(Hyy_g_exact.shape[0]))
                Hyy_g_reg_pinv_Hyx_f_Dy_f_flat_exact = Hyy_g_reg_pinv_exact @ Dy_f_flat_np
                print(f"[Time {time.time() - gc_st:.3f}] Hyy_g_reg_inv_Dy_f_flat.shape {Hyy_g_reg_inv_Dy_f_flat.shape}\n{Hyy_g_reg_inv_Dy_f_flat[:show_num]}")
                print(f"[Time {time.time() - h_st:.3f}] Hyy_g_reg_pinv_Hyx_f_Dy_f_flat_exact.shape {Hyy_g_reg_pinv_Hyx_f_Dy_f_flat_exact.shape}\n{Hyy_g_reg_pinv_Hyx_f_Dy_f_flat_exact[:show_num]}")
                print("Cossim:", Cossim(Hyy_g_reg_inv_Dy_f_flat, Hyy_g_reg_pinv_Hyx_f_Dy_f_flat_exact))
                print("SANITY CHECK", Hyy_g_reg_pinv_exact @ (Hyy_g_exact.cpu().detach().numpy() + self.hessian_reg_y * np.identity(Hyy_g_exact.shape[0])))

        if self.follow_y:
            """ Follow_y """
            # print("Follow y")
            Hyx_f_lo = JacobianVectorProduct(Dx_f, self.params_y)
            lr_x_Dx_f_flat = lr_x * Dx_f_flat
            lr_x_Dx_f_flat_np = np.nan_to_num(lr_x_Dx_f_flat.view(-1).cpu().detach().numpy())
            lr_x_Hyx_f_Dx_f_flat = torch.Tensor(Hyx_f_lo(lr_x_Dx_f_flat_np)).view(-1, 1).to(self.device)
            
            if self.newton_y:
                update_corr_y = lr_x_Hyx_f_Dx_f_flat
            else:
                Hyy_g_reg_lo = JacobianVectorProduct(Dy_g, self.params_y, self.hessian_reg_y)
                Hyy_g_reg_inv_Hyx_f_Dx_f_flat, status = ssla.cg(Hyy_g_reg_lo, lr_x_Hyx_f_Dx_f_flat.cpu().detach().numpy(), maxiter=self.cg_maxiter)
                update_corr_y = torch.Tensor(Hyy_g_reg_inv_Hyx_f_Dx_f_flat).view(-1, 1).to(self.device)

            # if torch.isnan(update_corr_y).any():
            #     # raise ValueError('vector nan')
            #     update_corr_y[update_corr_y != update_corr_y] = 0
            nan_to_num_torch(update_corr_y)
            # print("update_corr_y\n", update_corr_y.cpu().detach().numpy().ravel()[:6])

            if calculate_hessian:
                # print(f"[{time_step}] lr_x", lr_x)
                # print("Dx_f_flat", Dx_f_flat)
                # print("lr_x_Dx_f_flat_np", lr_x_Dx_f_flat_np)
                # print(f"Dx_f {Dx_f[:show_num]}")
                # print(f"Dy_g {Dy_g[:show_num]}")
                print(f"self.params_x.shape {[param.shape for param in self.params_x]}")
                h_st = time.time()
                Hxx_f_exact = hessian(loss_G, self.params_x)
                print(f"[Time {time.time() - h_st:.3f}] Hxx_f_exact.shape {Hxx_f_exact.shape}\n{Hxx_f_exact[:show_num]}")
                print(f"self.params_y.shape {[param.shape for param in self.params_y]}")
                Hyy_g_exact = hessian(loss_D, self.params_y)
                print(f"[Time {time.time() - h_st:.3f}] Hyy_g_exact.shape {Hyy_g_exact.shape}\n{Hyy_g_exact[:show_num]}")
                h_st = time.time()
                Hyy_g_reg_pinv_exact = scipy.linalg.pinv(Hyy_g_exact.cpu().detach().numpy() + self.hessian_reg_y * np.identity(Hyy_g_exact.shape[0]))
                Hyy_g_reg_pinv_Hyx_f_Dx_f_flat_exact = Hyy_g_reg_pinv_exact @ lr_x_Hyx_f_Dx_f_flat.cpu().detach().numpy()
                print(f"[Time {time.time() - gc_st:.3f}] Hyy_g_reg_inv_Hyx_f_Dx_f_flat.shape {Hyy_g_reg_inv_Hyx_f_Dx_f_flat.shape}\n{Hyy_g_reg_inv_Hyx_f_Dx_f_flat[:show_num]}")
                print(f"[Time {time.time() - h_st:.3f}] Hyy_g_reg_pinv_Hyx_f_Dx_f_flat_exact.shape {Hyy_g_reg_pinv_Hyx_f_Dx_f_flat_exact.shape}\n{Hyy_g_reg_pinv_Hyx_f_Dx_f_flat_exact[:show_num]}")
                print("Cossim:", Cossim(Hyy_g_reg_inv_Hyx_f_Dx_f_flat.cpu().detach().numpy(), Hyy_g_reg_pinv_Hyx_f_Dx_f_flat_exact))
                print("SANITY CHECK", Hyy_g_reg_pinv_exact @ (Hyy_g_exact.cpu().detach().numpy() + self.hessian_reg_y * np.identity(Hyy_g_exact.shape[0])))

        self.culmulative_grad_corr_time += time.time() - gc_st

        if not self.cgd:
            update_x = update_x + update_corr_x
            update_y = update_y + update_corr_y

        if self.newton_y:
            Hyy_g_reg_lo = JacobianVectorProduct(Dy_g, self.params_y, self.hessian_reg_y)
            Hyy_g_reg_inv_update_y, status = ssla.cg(Hyy_g_reg_lo, update_y.cpu().detach().numpy(), maxiter=self.cg_maxiter)
            update_y = torch.Tensor(Hyy_g_reg_inv_update_y).view(-1, 1).to(self.device)

        if self.use_momentum:
            if verbose: print(f"{'=' * 40}\ntime_step: {time_step}")
            if verbose: print(f"update_x: {update_x[:show_num].cpu().detach().numpy().ravel()}")
            if verbose: print(f"update_y: {update_y[:show_num].cpu().detach().numpy().ravel()}")
            if verbose: print(f"lr_x: {lr_x}")
            if verbose: print(f"lr_y: {lr_y}")

            momentum_x = torch.zeros_like(update_x, requires_grad=False) if momentum_x is None else momentum_x
            momentum_y = torch.zeros_like(update_y, requires_grad=False) if momentum_y is None else momentum_y
            sq_avg_x = torch.zeros_like(update_x, requires_grad=False) if sq_avg_x is None else sq_avg_x
            sq_avg_y = torch.zeros_like(update_y, requires_grad=False) if sq_avg_y is None else sq_avg_y

            if verbose: print(f"prev momentum_x: {momentum_x[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"prev momentum_y: {momentum_y[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"prev sq_avg_x: {sq_avg_x[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"prev sq_avg_y: {sq_avg_y[:show_num].cpu().numpy().ravel()}")
            if verbose:
                update_x_np = update_x.detach().cpu().numpy().ravel()
                momentum_x_np = momentum_x.cpu().numpy().ravel()
                momentum_x_np_after = momentum * momentum_x_np + (1 - momentum) * (update_x_np / lr_x) ** 2

                sq_avg_x_np = sq_avg_x.cpu().numpy().ravel()
                sq_avg_x_np_after = beta * sq_avg_x_np + (1 - beta) * (update_x_np / lr_x) ** 2

                print(f"momentum_x_np_after: {momentum_x_np_after[:show_num]}")
                print(f"sq_avg_x_np_after: {sq_avg_x_np_after[:show_num]}")

            momentum_x.mul_(momentum).add_((1 - momentum) * update_x.detach() / lr_x)
            momentum_y.mul_(momentum).add_((1 - momentum) * update_y.detach() / lr_y)
            sq_avg_x.mul_(beta).addcmul_(1 - beta, update_x.detach() / lr_x, update_x.detach() / lr_x)
            sq_avg_y.mul_(beta).addcmul_(1 - beta, update_y.detach() / lr_y, update_y.detach() / lr_y)

            if verbose: print(f"next momentum_x: {momentum_x[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"next momentum_y: {momentum_y[:show_num].cpu().numpy().ravel()}")


            if verbose: print(f"next sq_avg_x: {sq_avg_x[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"next sq_avg_y: {sq_avg_y[:show_num].cpu().numpy().ravel()}")

            bias_corr_mo = 1 - momentum ** time_step
            bias_corr_sq = 1 - beta ** time_step
            if verbose: print(f"bias_corr_mo: {bias_corr_mo}")
            if verbose: print(f"bias_corr_sq: {bias_corr_sq}")


            if verbose: print(f"actual lr_x: {lr_x}")
            if verbose: print(f"actual lr_y: {lr_y}")
            update_x = lr_x * momentum_x * np.sqrt(bias_corr_sq) / sq_avg_x.sqrt().mul(bias_corr_mo).add(eps * bias_corr_sq)
            update_y = lr_y * momentum_y * np.sqrt(bias_corr_sq) / sq_avg_y.sqrt().mul(bias_corr_mo).add(eps * bias_corr_sq)

            if verbose: print(f"actual update_x: {update_x[:show_num].cpu().numpy().ravel()}")
            if verbose: print(f"actual update_y: {update_y[:show_num].cpu().numpy().ravel()}")


        self.culmulative_training_time += time.time() - st

        """ Update  """
        # TODO: Add conjugate gradient
        self.state.update({"Dx_f": Dx_f_flat_dt, "Dy_g": Dy_g_flat_dt, 'sq_exp_avg_x': sq_avg_x, 'sq_exp_avg_y': sq_avg_y, 'momentum_x': momentum_x, 'momentum_y': momentum_y})

        self.update_x = update_x
        self.update_y = update_y

        if self.collect_info:
            grad_raw_norm_x = torch.norm(Dx_f_flat).item()
            grad_raw_norm_y = torch.norm(Dy_g_flat).item()
            grad_corr_norm_x = torch.norm(update_corr_x).item()
            grad_corr_norm_y = torch.norm(update_corr_y).item()
            update_tot_norm_x = torch.norm(update_x).item()
            update_tot_norm_y = torch.norm(update_y).item()
            iter_num = self.info["iter_num"]

            self.info.update({'grad_raw_norm_x': grad_raw_norm_x, 'grad_raw_norm_y': grad_raw_norm_y, 'grad_corr_norm_x': grad_corr_norm_x,
                              'grad_corr_norm_y': grad_corr_norm_y, 'update_tot_norm_x': update_tot_norm_x, 'update_tot_norm_y': update_tot_norm_y,
                              'training_time': self.culmulative_training_time, 'gc_time': self.culmulative_grad_corr_time, 'iter_num': iter_num + 1,
                              "lr_x_actual": lr_x, "lr_y_actual": lr_y})

    def step(self):
        """ Perform actual parameter updates """
        index = 0
        for p in self.params_x:
            p.data.add_(-self.update_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == self.update_x.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.params_y:
            p.data.add_(-self.update_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == self.update_y.numel(), 'Minimizer CG size mismatch'

    def Calculate_eig_vals(self, loss_G, loss_D, eigs=True):
        if eigs:
            eig_st = time.time()
            Dx_f = autograd.grad(loss_G, self.params_x, create_graph=True, retain_graph=True)
            Dy_g = autograd.grad(loss_D, self.params_y, create_graph=True, retain_graph=True)
            Hxx_f_lo = JacobianVectorProduct(Dx_f, self.params_x)
            Hxy_g_lo = JacobianVectorProduct(Dy_g, self.params_x)
            Hyx_f_lo = JacobianVectorProduct(Dx_f, self.params_y)
            Hyy_g_lo = JacobianVectorProduct(Dy_g, self.params_y)
            Hxx_f_reg_lo = JacobianVectorProduct(Dx_f, self.params_x, self.hessian_reg_x)
            Hyy_g_reg_lo = JacobianVectorProduct(Dy_g, self.params_y, self.hessian_reg_y)

            Jacobian_lo = JacobianVectorProduct(Dx_f + Dy_g, self.params_x + self.params_y)
            def Get_sparse_eig_vals(lo):
                try:
                    return np.hstack((scipy.sparse.linalg.eigs(lo, k=self.eig_vals_num, which='LR')[0], scipy.sparse.linalg.eigs(lo, k=self.eig_vals_num, which='SR')[0]))
                except:
                    print("Eigvals nonconvergent")
                    return None

            # print("\n\n0\n")
            eig_vals_Hxx_f = Get_sparse_eig_vals(Hxx_f_lo)
            eig_vals_Hyy_g = Get_sparse_eig_vals(Hyy_g_lo)
            # print("\n\n1\n")
            eig_vals_Hxx_f_reg = Get_sparse_eig_vals(Hxx_f_reg_lo)
            eig_vals_Hyy_g_reg = Get_sparse_eig_vals(Hyy_g_reg_lo)
            # print("\n\n2\n")
            eig_vals_J = Get_sparse_eig_vals(Jacobian_lo)

            Hxx_f_Schur_lo = SchurComplement(Hxx_f_lo, Hxy_g_lo, Hyx_f_lo, Hyy_g_reg_lo, tol_gmres=self.tol_gmres, precise=self.precise, maxiter_cg=5)
            Hyy_g_Schur_lo = SchurComplement(Hyy_g_lo, Hyx_f_lo, Hxy_g_lo, Hxx_f_reg_lo, tol_gmres=self.tol_gmres, precise=self.precise, maxiter_cg=5)

            # print("\n\n3\n")
            eig_vals_Hxx_f_Schur = Get_sparse_eig_vals(Hxx_f_Schur_lo)
            eig_vals_Hyy_g_Schur = Get_sparse_eig_vals(Hyy_g_Schur_lo)
            self.eig_calculation_time += time.time() - eig_st

            return eig_vals_Hxx_f, eig_vals_Hyy_g, eig_vals_Hxx_f_reg, eig_vals_Hyy_g_reg, eig_vals_J, eig_vals_Hxx_f_Schur, eig_vals_Hyy_g_Schur, self.eig_calculation_time
        else:
            return None, None, None, None, None, None, None, self.eig_calculation_time





