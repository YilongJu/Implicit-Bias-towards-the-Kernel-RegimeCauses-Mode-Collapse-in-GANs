from Synthetic_Dataset import Synthetic_Dataset

import random
import numpy as np
import math

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.models.inception import inception_v3

import argparse
from Run_GAN_Training import *
from ComputationalTools import *
from models import MLP, ResNet18

aux_model_folder = "Aux_Models"
if not os.path.exists(aux_model_folder):
    os.makedirs(aux_model_folder)

def Get_classification_logit_prob_class(net, input, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    with torch.no_grad():
        pred_logits = net(torch.from_numpy(input).to(device).float())
        # print(pred_logits)
        pred_probs = F.softmax(pred_logits, dim=1).data.cpu().numpy()
        # print(pred_probs)
        mean_pred_probs = np.mean(pred_probs, axis=0)
        # print(mean_pred_probs)

        pred_classes = np.argmax(pred_probs, axis=1)
        # print(pred_classes)
        pred_classes_one_hot = np.zeros_like(pred_probs)
        pred_classes_one_hot[np.arange(len(pred_probs)), pred_classes] = 1
        # print(pred_classes_one_hot)
        pred_classes_count = np.sum(pred_classes_one_hot, axis=0)
        # print(pred_classes_count)
        mode_num = np.sum(pred_classes_count[:-1] > 0)
        # print(mode_num)

    return mean_pred_probs, mode_num, pred_classes_count

def Load_aux_classifier(dataset_config):
    print("dataset_config", dataset_config)
    aux_model_checkpoint_folder_list = glob.glob(os.path.join(aux_model_folder, f"{dataset_config}*"))
    print("aux_model_folder", aux_model_folder)
    print("dataset_config", dataset_config)
    print("aux_model_checkpoint_folder_list", aux_model_checkpoint_folder_list)
    aux_model_checkpoint_folder = aux_model_checkpoint_folder_list[0]
    aux_model_checkpoint_path_list = glob.glob(os.path.join(aux_model_checkpoint_folder, f"*.pth"))
    aux_model_checkpoint_path = aux_model_checkpoint_path_list[0]
    """ load_models """
    checkpoint = torch.load(aux_model_checkpoint_path)
    args = checkpoint["args"]
    rng = np.random.RandomState(seed=args.seed)
    dataset = Synthetic_Dataset(args.data, rng, std=args.mog_std, scale=args.mog_scale, sample_per_mode=1000, with_neg_samples=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    aux_classifier_loaded = MLP(n_hidden_layers=args.layers, n_hidden_neurons=args.hidden, input_dim=2, output_dim=dataset.n + 1, type="D", use_bias=True).to(device)
    aux_classifier_loaded.load_state_dict(checkpoint["aux_state"])
    real_data_prob, real_mode_num, pred_classes_count = Get_classification_logit_prob_class(aux_classifier_loaded, dataset.all_samples)
    real_data_prob[-1] = EPS_BUFFER
    real_data_prob = Normalize(real_data_prob, ord=1)
    return aux_classifier_loaded, real_data_prob, real_mode_num

def Train_aux(args):
    np.random.seed(args.seed)
    torch.manual_seed(seed=args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"{'=' * 40}\n[Using device: {device}]\n{'=' * 40}")
    if args.data == "mnist":
        if args.arch == "default":
            arch = "ResNet18"
        else:
            arch = args.arch

        additional_arch_text = ""
        if args.transfer_learning:
            additional_arch_text += "_tl"
        if args.fine_tune:
            additional_arch_text += "_ft"
        if args.use_KL_loss:
            additional_arch_text += f"_kll{args.lambda_KL_loss}"


        model_name = f"{args.data}_{arch}{additional_arch_text}_oln{args.test_data_num}_lr{args.lr}_seed{args.seed}"
    else:
        model_name = f"{args.data}-{args.mog_scale}-{args.mog_std}_MLP-{args.layers}-{args.hidden}"

    model_save_folder = os.path.join(aux_model_folder, model_name)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    if args.data == "mnist":
        output_shape = [1, 28, 28]
        mnist_dataset = MNIST(root=os.path.join(real_data_folder, args.data), train=True, transform=transform, download=True)
        dataset = MNIST(root=os.path.join(real_data_folder, args.data), train=True, transform=transform, download=True)

        """ Get an untrained Generator """
        noise_shape = (args.test_data_num, args.z_dim)
        fixed_noise = torch.randn(noise_shape, device=device) * args.z_std
        G, D = Get_models(args, output_shape, gpu_num=1)
        G = G.to(device)

        """ Generate outliers """
        n_batches = int(math.ceil(float(len(fixed_noise)) / float(args.batch_size)))
        n_out = 0
        outliers = []
        print("fixed_noise.shape", fixed_noise.shape)

        for i in range(n_batches):
            z = fixed_noise[(i * args.batch_size):min((i + 1) * args.batch_size, len(fixed_noise)), :].to(device)
            # print("z.shape", z.shape)
            outlier = G(z)
            # print("outlier.shape", outlier.shape)

            outliers.append(outlier.detach().cpu().numpy())
            n_out += outlier.shape[0]
            print(f"{n_out} / {args.test_data_num}", z.shape)

        outliers = np.concatenate(outliers, axis=0)
        outliers = np.squeeze(outliers, axis=1).astype(np.float32)
        print("outliers.shape", outliers.shape)
        plt.imshow(outliers[0, ...])
        plt.savefig("test.png")

        outliers_torch = torch.from_numpy(outliers).byte()


        """ Add outliers to dataset """
        print("mnist_dataset.data[0].shape", mnist_dataset.data[0].shape)
        print("mnist_dataset.data.type()", mnist_dataset.data.type())
        print("outliers_torch.type()", outliers_torch.type())
        print("mnist_dataset.targets[0].shape", mnist_dataset.targets[0].shape, mnist_dataset.targets[0])
        dataset.data = torch.cat([mnist_dataset.data, outliers_torch], dim=0)
        outlier_label = 10
        dataset.targets = torch.cat([mnist_dataset.targets, torch.ones(args.test_data_num).long() * outlier_label])

        plt.imshow(mnist_dataset.data[0, ...])
        plt.savefig("test_5.png")
    else:
        rng = np.random.RandomState(seed=args.seed)
        dataset = Synthetic_Dataset(args.data, rng, std=args.mog_std, scale=args.mog_scale, sample_per_mode=1000, with_neg_samples=True)
        output_shape = None

    # dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    if args.data == "mnist":
        aux_classifier = ResNet18(num_classes=11).to(device)

        pretrained_classifier_without_outlier_class = ResNet18().to(device)
        pretrained_classifier_without_outlier_class.load_state_dict(torch.load(args.model_dir))

        if args.transfer_learning:
            for params_pre, params_new in zip(pretrained_classifier_without_outlier_class.parameters(), aux_classifier.parameters()):
                print(params_pre.shape, params_new.shape, end=", ")
                if params_pre.shape == params_new.shape:
                    print("assigned")
                    params_new = params_pre
                    if args.fine_tune:
                        params_new.requires_grad = True
                    else:
                        params_new.requires_grad = False
                else:
                    print("\n", end="")

    else:
        aux_classifier = MLP(n_hidden_layers=args.layers, n_hidden_neurons=args.hidden, input_dim=2, output_dim=dataset.n + 1, type="D", use_bias=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=aux_classifier.parameters(), lr=args.lr)
    criterion_KL = nn.KLDivLoss()
    Softmax = nn.Softmax(dim=1)

    training_timer = Timer()
    """ Training """
    if args.data == "mnist":
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        epoch_num = args.epoch
        if epoch_num <= 0:
            epoch_num = int(1e8)

        iteration = 0
        for e in range(epoch_num):
            for batch_id, data_batch in enumerate(dataloader):

                images, labels = data_batch
                # print("len(images)", len(images))
                # print("images.shape", images.shape)
                mask = labels != outlier_label
                images = images.to(device)
                labels = labels.to(device)
                # print("images_torch.shape", images_torch.shape)

                pred_logits = aux_classifier(images) # (N, 11)
                pretrained_logits = pretrained_classifier_without_outlier_class(images[mask, ...])  # (N - n_o, 10)
                pred_logits_without_outliers = pred_logits[mask, ...] # (N - n_o, 11)
                labels_without_outliers = labels[mask, ...] # (N - n_o)

                loss = criterion(pred_logits, labels)
                vanilla_loss = loss.item()
                loss_pretrained = criterion(pretrained_logits, labels_without_outliers)

                pred_probs_without_outliers = Softmax(pred_logits_without_outliers)  # (N - n_o, 11)
                pretrained_probs_without_outliers = Softmax(pretrained_logits) # (N - n_o, 10)
                pretrained_probs_without_outliers_padded = torch.cat([pretrained_probs_without_outliers, torch.zeros(size=[pretrained_probs_without_outliers.shape[0], 1]).to(device)], dim=1) # (N - n_o, 11)

                loss_KL = criterion_KL(pred_probs_without_outliers, pretrained_probs_without_outliers_padded)

                pred_labels = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
                pred_labels_without_outliers = pred_labels[mask, ...]
                pred_labels_outliers = pred_labels[~mask, ...]

                # print(pred_labels_without_outliers)
                # print(labels_without_outliers)
                correct_prediction_without_outliers = np.sum(pred_labels_without_outliers == labels_without_outliers.cpu().numpy())
                correct_prediction_outliers = np.sum(pred_labels_outliers == outlier_label)

                accuracy_without_outliers = correct_prediction_without_outliers / len(pred_labels_without_outliers)
                accuracy_outliers = correct_prediction_outliers / len(pred_labels_outliers)

                if args.use_KL_loss:
                    loss += args.lambda_KL_loss * loss_KL

                if iteration == 0 or iteration % args.plot_iter == 0:
                    print(f"Iter {iteration} / {args.iteration}, Loss {vanilla_loss:.6f}, Loss (Pre) {loss_pretrained.item():.6f}, Loss (KL) {loss_KL.item():.6f}, Loss (Total) {loss.item():.6f}, Time: {training_timer.Print(print_time=False)}")

                    print(f"accuracy_without_outliers = {correct_prediction_without_outliers} / {len(pred_labels_without_outliers)} = {accuracy_without_outliers:.4f}")
                    print(f"accuracy_outliers = {correct_prediction_outliers} / {len(pred_labels_outliers)} = {accuracy_outliers:.4f}")

                if iteration % args.save_iter == 0:
                    torch.save({"args": args, "aux_state": aux_classifier.state_dict()}, os.path.join(model_save_folder, f"iter_{iteration}-{args.iteration}.pth"))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iteration += 1

            if args.iteration > 0 and iteration > args.iteration:
                break


    else:
        for i in range(args.iteration + 1):
            features = torch.from_numpy(dataset.all_samples).to(device).float()
            labels = torch.from_numpy(dataset.all_labels).to(device).long()
            pred_logits = aux_classifier(features)
            loss = criterion(pred_logits, labels)

            if i % args.plot_iter == 0:
                print(f"Iter {i} / {args.iteration}, Loss {loss.item():.4f}, Time: {training_timer.Print(print_time=False)}")

                pred_all_labels = np.argmax(pred_logits.detach().cpu().numpy(), axis=1)
                plt.figure(figsize=(14, 14))
                plt.scatter(dataset.all_samples[:, 0], dataset.all_samples[:, 1], \
                            c=pred_all_labels, cmap=dataset.cm)
                plt.axes().set_aspect('equal')
                plt.savefig(os.path.join(model_save_folder, f"iter_{i}-{args.iteration}.png"), dpi=300)
                plt.close()

            if i % args.save_iter == 0:
                torch.save({"args": args, "aux_state": aux_classifier.state_dict()}, os.path.join(model_save_folder, f"iter_{i}-{args.iteration}.pth"))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=-1, help="number of iterations of training")
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--test_data_num", type=int, default=256, help="number of test data")
    parser.add_argument("--lr", type=float, default=0.002, help="generator learning rate")
    parser.add_argument("--lr_warmup", type=float, default=0.00000001, help="generator warmup learning rate")
    parser.add_argument("--lr_decay", type=float, default=0, help="generator learning rate decay")
    parser.add_argument("--warmup_iter", type=int, default=1, help="number of warmup iterations")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0, help="weight decay factor")
    parser.add_argument("--gp", type=float, default=0, help="gradient penalty")
    parser.add_argument("--z_dim", type=int, default=-1, help="dimension of latent noise")
    parser.add_argument("--z_std", type=float, default=1, help="std of latent noise")
    parser.add_argument("--mog_std", type=float, default=-1, help="std of mog")
    parser.add_argument("--mog_scale", type=float, default=-1, help="scale of mog")
    parser.add_argument("--hidden", type=int, default=32, help="dimension of hidden units")
    parser.add_argument("--layers", type=int, default=2, help="num of hidden layer")
    parser.add_argument("--x_dim", type=int, default=2, help="data dimension")
    parser.add_argument("--init", type=str, default="xavier", help="which initialization scheme")
    parser.add_argument("--arch", type=str, default="default", help="which gan architecture")
    parser.add_argument("--data", type=str, default="cifar", help="which dataset")
    parser.add_argument("--data_std", type=float, default=0.1, help="std for each mode")
    parser.add_argument("--act", type=str, default="relu", help="which activation function for gen")  # elu, relu, tanh
    parser.add_argument("--divergence", type=str, default="JS", help="which activation function for disc")  # NS, JS, indicator, wgan
    parser.add_argument("--method", type=str, default="fr", help="which optimization regularization method to use")
    parser.add_argument("--opt_type", type=str, default="sgd", help="which optimization method to use")
    parser.add_argument("--reg_param", type=float, default=5.0, help="reg param for JARE")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum coefficient for the whole system")
    parser.add_argument("--gamma", type=float, default=0.999, help="gamma for adaptive learning rate")
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument("--inner_iter", type=int, default=5, help="conjugate gradient or gradient descent steps")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--damping", type=float, default=1.0, help="damping term for CG")
    parser.add_argument("--adapt_damping", action='store_false', help="whether to adapt damping")
    parser.add_argument("--real_time_video", action='store_false', help="whether to play video in real time")
    parser.add_argument("--plot_iter", type=int, default=150, help="number of iter to plot")
    parser.add_argument("--save_iter", type=int, default=10000, help="number of iter to save")
    parser.add_argument("--eval_iter", type=int, default=-1, help="number of iter to eval")
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to use")
    parser.add_argument("--spanning_init", action='store_true', help="whether to spanning_init")
    parser.add_argument("--si_lr", type=float, default=0.01, help="spanning_init learning rate")
    parser.add_argument("--degen_z", action='store_true', help="whether to reduce dim_z")
    parser.add_argument("--penalty", type=float, default=0, help="l2 norm penalty of G params")
    parser.add_argument("--hessian_reg", type=float, default=0, help="reg of G hessian")
    parser.add_argument("--plot_lim_z", type=float, default=7, help="plot_lim_z")
    parser.add_argument("--plot_lim_x", type=float, default=7, help="plot_lim_x")
    parser.add_argument("--verbose", action='store_true', help="whether to print debug info")
    parser.add_argument("--base_filter_num", type=int, default=64, help="number of base filter number for G")
    parser.add_argument("--G_base_filter_num", type=int, default=64, help="number of base filter number for G")
    parser.add_argument("--D_base_filter_num", type=int, default=64, help="number of base filter number for D")
    parser.add_argument('--freeze_w', action='store_true', help="freeze_w")
    parser.add_argument('--model_dir', default="Saved_models/mnist_model_10.ckpt")
    parser.add_argument("--fine_tune", action='store_true', help="whether to fine tune new network")
    parser.add_argument("--transfer_learning", action='store_true', help="whether to transfer weights from pretrained network")
    parser.add_argument("--use_KL_loss", action='store_true', help="whether to add KL loss (compared to the pretrained classfier)")
    parser.add_argument("--lambda_KL_loss", type=float, default=0, help="weight for KL loss")

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    Train_aux(args)
