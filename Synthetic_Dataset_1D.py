import torch
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from matplotlib import colors
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ComputationalTools import *


class Synthetic_Dataset_1D(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, rng=None, std=-1, scale=-1, num_samples=5000, sample_per_mode=None, seed=2020, with_neg_samples=False, range_x=[-1, 1], scale_y=1., noise_std=0, num_periods=1, offset_x=0, offset_y=0, Pi=None, Mu=None, Sigma2=None, args=None):
        self.args = args
        if self.args is not None:
            self.seed = self.args.seed
            self.Pi = [float(ele) for ele in self.args.Pi.split(",")]
            self.Mu = [float(ele) for ele in self.args.Mu.split(",")]
            self.Sigma2 = [float(ele) for ele in self.args.Sigma2.split(",")]
        else:
            self.seed = seed
            self.Pi = Pi
            self.Mu = Mu
            self.Sigma2 = Sigma2

        self.Pi = self.Pi / np.sum(self.Pi)

        'Initialization'
        self.with_neg_samples = with_neg_samples
        np.random.seed(self.seed)
        random.seed(self.seed)

        if rng is None:
            self.rng = np.random.RandomState(seed=self.seed)
        else:
            self.rng = rng

        self.data = data
        self.range_x = range_x
        self.span_x = range_x[1] - range_x[0]
        self.scale_y = scale_y
        self.num_samples = num_samples
        self.num_periods = num_periods
        self.noise_std = noise_std
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.MG = None

        if self.data == "sine":
            self.x = self.rng.uniform(low=range_x[0], high=range_x[1], size=self.num_samples)
            self.x_linspace = np.linspace(-1.1, 1.1, 1001)
            self.xlim = np.array([(range_x[0] + range_x[1]) / 2 - 1.25 * self.span_x / 2, (range_x[0] + range_x[1]) / 2 + 1.25 * self.span_x / 2])
            self.y = np.sin((self.x - range_x[0]) * 2 * np.pi / (self.span_x / self.num_periods)) * self.scale_y + self.rng.normal(scale=self.noise_std, size=self.num_samples)
            self.y_linspace = np.sin((self.x_linspace - range_x[0]) * 2 * np.pi / (self.span_x / self.num_periods)) * self.scale_y
            self.ylim = np.array([-1.25, 1.25]) * self.scale_y
        elif self.data == "step":
            self.x = self.rng.uniform(low=range_x[0], high=range_x[1], size=self.num_samples - self.num_samples // 3)
            self.x_linspace = np.linspace(-1.1, 1.1, 1001)
            self.xlim = np.array([(range_x[0] + range_x[1]) / 2 - 1.25 * self.span_x / 2, (range_x[0] + range_x[1]) / 2 + 1.25 * self.span_x / 2])
            x_0 = self.rng.uniform(low=-0.01, high=0.01, size=self.num_samples // 3)
            self.x = np.sort(np.concatenate([x_0, self.x]))
            y_1 = (-self.x[self.x < -0.01] ** 2 - 1) * self.scale_y
            y_2 = 100 * self.x[np.logical_and(self.x < 0.01, self.x >= -0.01)] * self.scale_y
            y_3 = (-self.x[self.x >= 0.01] ** 2 + 1) * self.scale_y
            self.y = np.concatenate([y_1, y_2, y_3])
            self.ylim = np.array([-2.5, 2.5]) * self.scale_y
            y_1 = (-self.x_linspace[self.x_linspace < -0.01] ** 2 - 1) * self.scale_y
            y_2 = 100 * self.x_linspace[np.logical_and(self.x_linspace < 0.01, self.x_linspace >= -0.01)] * self.scale_y
            y_3 = (-self.x_linspace[self.x_linspace >= 0.01] ** 2 + 1) * self.scale_y
            self.y_linspace = np.concatenate([y_1, y_2, y_3])
        elif self.data == "mog":
            self.MG = Mixture_of_Gaussian_Generator(self.Pi, self.Mu, self.Sigma2, seed=self.seed)
            self.all_samples, self.all_labels = self.MG.Generate_numbers(self.num_samples)
            self.all_samples = self.all_samples[:, np.newaxis]
            self.all_labels = self.all_labels[:, np.newaxis]

            self.xlim = self.MG.Get_bounding_box()
        else:
            raise NotImplementedError

        if self.data not in ["mog"]:
            self.x = self.x + self.offset_x
            self.y = self.y + self.offset_y

            self.x_linspace = self.x_linspace + self.offset_x
            self.y_linspace = self.y_linspace + self.offset_y

            self.xlim = self.xlim + self.offset_x
            self.ylim = self.ylim + self.offset_x

            self.all_samples = np.concatenate([self.x[:, np.newaxis], self.y[:, np.newaxis]], axis=1)
            self.all_labels = np.zeros(shape=[len(self.all_samples), 1])


    def sample(self, N):
        index = self.rng.permutation(self.num_samples)
        return self.all_samples[index[:N]], self.all_labels[index[:N]]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_samples)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.all_samples[index], self.all_labels[index]


if __name__ == "__main__":
    # rng = np.random.RandomState(seed=2021)
    # syn_dataset = Synthetic_Dataset_1D("step", rng, num_samples=20, noise_std=0, range_x=[-1.1, 1.1])
    # print(len(syn_dataset))
    # print(syn_dataset.all_samples[:5])
    # print(syn_dataset[:5])
    # print(syn_dataset.sample(5))
    #
    # plt.scatter(syn_dataset.all_samples[:, 0], syn_dataset.all_samples[:, 1])
    # plt.plot(syn_dataset.x_linspace, syn_dataset.y_linspace)
    # plt.axes().set_aspect('equal')
    # plt.show()
    #
    # def Get_seed(id):
    #     return id
    #
    # data_loader = DataLoader(syn_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    # print("Epoch 0")
    # for i_batch, sample_batched in enumerate(data_loader):
    #     print(i_batch, len(sample_batched), f"\n{sample_batched}")
    # print("Epoch 1")
    # for i_batch, sample_batched in enumerate(data_loader):
    #     print(i_batch, len(sample_batched), f"\n{sample_batched}")

    syn_dataset = Synthetic_Dataset_1D("mog", num_samples=20, seed=1, Pi=[0.5, 0.5], Mu=[-7.5, 7.5], Sigma2=[1., 1.])
    print(len(syn_dataset))
    print(syn_dataset.all_samples)
    # print(syn_dataset.all_samples[:5])
    # print(syn_dataset[:5])
    # print(syn_dataset.sample(5))
    # print(syn_dataset.sample(5))

    data_loader = DataLoader(syn_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    print("Epoch 0")
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, len(sample_batched), f"\n{sample_batched}")
    print("Epoch 1")
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, len(sample_batched), f"\n{sample_batched}")