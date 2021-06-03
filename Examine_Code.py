""" Stackelberg """

# Dg, Dd = build_game_gradient([G_loss, D_loss], [G, D])
Dg = autograd.grad(G_loss, G.parameters(), create_graph=True)
Dd = autograd.grad(D_loss, D.parameters(), create_graph=True)
Dd_g = autograd.grad(G_loss, D.parameters(), create_graph=True)  # Dy_f

DD_reg = JacobianVectorProduct(Dd, list(D.parameters()), regularization)
# leader_grad, q, x0 = compute_stackelberg_grad(G, Dg, Dd, Dd_g, DD_reg, x0, precise=precise, device=device)
Dg_vec = torch.cat([_.flatten() for _ in Dg]).view(-1, 1) # Dx_f_flat
Dd_g_vec = torch.cat([_.flatten() for _ in Dd_g]).view(-1, 1) # Dy_f_flat

w, status = scipy.sparse.linalg.cg(DD_reg, Dd_g_vec.cpu().detach().numpy(), maxiter=5)
q = torch.Tensor(JacobianVectorProduct(Dd, list(G.parameters()))(w)).view(-1, 1).to(device)

leader_grad = Dg_vec - q
""" Ours """
Dx_f = autograd.grad(loss_G, list(G.parameters()), create_graph=True, retain_graph=True)
Dy_g = autograd.grad(loss_D, list(D.parameters()), create_graph=True, retain_graph=True)
Dy_f = autograd.grad(loss_G, list(D.parameters()), create_graph=True, retain_graph=True)

Hyy_g_reg_lo = JacobianVectorProduct(Dy_g, list(D.parameters()), self.hessian_reg_y)

Dy_g_flat = torch.cat([_.flatten() for _ in Dy_g]).view(-1, 1)
Dy_f_flat = torch.cat([_.flatten() for _ in Dy_f]).view(-1, 1)
Dy_f_flat_np = np.nan_to_num(Dy_f_flat.cpu().detach().numpy())

Hyy_g_reg_inv_Dy_f_flat, status = ssla.cg(Hyy_g_reg_lo, Dy_f_flat_np, maxiter=self.cg_maxiter)
# print("Hyy_g_reg_inv_Dy_f_flat.shape", Hyy_g_reg_inv_Dy_f_flat.shape)

Hxy_g_lo = JacobianVectorProduct(Dy_g, list(G.parameters()))
Hxy_g_Hyy_g_reg_inv_Dy_f_flat = torch.Tensor(Hxy_g_lo(Hyy_g_reg_inv_Dy_f_flat)).view(-1, 1).to(self.device)
update_x = lr_x * (Dx_f_flat - Hxy_g_Hyy_g_reg_inv_Dy_f_flat)
