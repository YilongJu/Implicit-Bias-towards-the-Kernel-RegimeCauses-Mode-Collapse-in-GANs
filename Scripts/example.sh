python Run_GAN_Training.py --z_dim 2 --z_std 1 --test_data_num 1024 --plot_lim_z 7 --plot_lim_x 2 --mog_std 0.01 --mog_scale 1 --data grid5 --opt_type rmsprop --divergence JS --method simgd --g_lr 0.001 --d_lr 0.001 --d_penalty 0 --g_hessian_reg 1 --d_hessian_reg 1 --iteration 400000 --plot_iter 4000 --seed 0 --arch mlp --g_layers 1 --g_hidden 32 --d_layers 1 --d_hidden 32 --rmsprop_init_1 --gamma 0.8 --save_param

python Run_GAN_Training.py --z_dim 2 --z_std 1 --test_data_num 1024 --plot_lim_z 7 --plot_lim_x 2 --mog_std 0.01 --mog_scale 1 --data grid5 --opt_type rmsprop --divergence JS --method simgd --g_lr 0.001 --d_lr 0.001 --d_penalty 0 --g_hessian_reg 1 --d_hessian_reg 1 --iteration 400000 --plot_iter 4000 --seed 0 --arch mlp --g_layers 1 --g_hidden 256 --d_layers 1 --d_hidden 256 --rmsprop_init_1 --gamma 1 --save_param --lazy --alpha_mobility 0.01 --alpha_mobility_D 1

python Run_GAN_Training.py --z_dim 100 --batch_size 256 --data mnist --opt_type rmsprop --gamma 0.999 --divergence JS --method simgd --g_lr 0.001 --d_lr 0.001 --iteration 200000 --plot_iter 50000 --eval_iter 50000 --rmsprop_init_1 --seed 0 --G_base_filter_num 32 --D_base_filter_num 32 --alpha_mobility 0.01 --alpha_mobility_D 0.01 --lazy

python Run_GAN_Training.py --z_dim 100 --batch_size 256 --data mnist --opt_type rmsprop --gamma 0.999 --divergence JS --method simgd --g_lr 0.001 --d_lr 0.001 --iteration 200000 --plot_iter 50000 --eval_iter 50000 --rmsprop_init_1 --seed 0 --G_base_filter_num 32 --D_base_filter_num 32 --alpha_mobility 0.01 --alpha_mobility_D 0.01 --lazy