>📋  A template README.md for code accompanying a Machine Learning paper

# Implicit Bias towards the Kernel RegimeCauses Mode Collapse in GANs

This repository is the official implementation of Implicit Bias towards the Kernel RegimeCauses Mode Collapse in GANs.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
Synthetic datasets will be automatically generated. And MNIST will be automatically downloaded.

## Main Steps
There are 3 main steps of this work: 1) training GANs with different hyperparameters, 2) Compute metrics of interest from the results, and 3) create figures or perform causal analysis on the results. We will introduce accordingly. 

## Training

To train the model(s) in the paper, we provide example scripts `Scripts/example_training_GANs.sh` for training Shallow ReLU GANs on 2D mixture of Gaussian datasets Grid and Random (first 2 commands, more datasets avaiable in `Synthetic_Dataset.py`), and for training DCGANs on MNIST (last 2 commands). The option `--alpha_mobility` and `--alpha_mobility_D` modify `alpha` of the generator and discriminator, respectively. And `--lazy` implements the lazy training scheme in [[1]](#1). These 3 options should be used combined.

```train
python Run_GAN_Training.py --z_dim 2 --z_std 1 --test_data_num 1024 --plot_lim_z 7 --plot_lim_x 2 --mog_std 0.01 --mog_scale 1 --data grid5 --opt_type rmsprop --divergence JS --method simgd --g_lr 0.001 --d_lr 0.001 --d_penalty 0 --g_hessian_reg 1 --d_hessian_reg 1 --iteration 400000 --plot_iter 4000 --seed 0 --arch mlp --g_layers 1 --g_hidden 32 --d_layers 1 --d_hidden 32 --rmsprop_init_1 --gamma 0.8 --save_param
```

Use ``--save_param`` when you want to save the metrics of interest and NN parameters per `--plot_iter` iterations during training. Otherwise, only the initial and final values will be saved. The saved metrics of interest and NN parameters will be in folder `Data` if not specified by `--save_path`.

## Computing Metrics

To compute metrics, first move all results to be computed to a folder under `Summaries`, such as the provided folder `GAN_training_results_examples`. Then, refer to the command in `Scripts/example_computing_metrics.sh`. For example, run:

```compute
python Data_to_csv.py --task_type 2D --summary_dir Summaries --task_dir GAN_training_results_examples
```

Then, a file named `results.csv` with the initial and final values of metrics will appear in this task folder. For values of all available iterations, modify variable `only_last_t` in `Data_to_csv.py` to `False`.


## Plotting

To reproduce Fig.1a in paper, run

```plot
python Plot_experiments.py
```

## Causal Analysis
To perform causal analysis and reproduce the plots for the distribution of marginal treatment effect (MTE) of each treatment shown in paper Fig. 5, run
```causal
python Causal_analysis.py
```
For retraining the DeepIV model, modify variable `load_deepIvEst_3_2` to `False`. Depending on the initial seed, the results can be slightly different.

## References
<a id="1">[1]</a> 
Chizat, Lénaïc, Edouard Oyallon, and Francis Bach. "On Lazy Training in Differentiable Programming." Advances in Neural Information Processing Systems 32 (2019): 2937-2947.