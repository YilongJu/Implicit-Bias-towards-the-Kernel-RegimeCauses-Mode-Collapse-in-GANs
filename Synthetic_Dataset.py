import torch
from torch.utils.data import Dataset, DataLoader

import random
import numpy as np
from matplotlib import colors
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from ComputationalTools import *


class Synthetic_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, rng=None, std=-1, scale=-1, n=9, num_samples=5000, sample_per_mode=None, seed=2020, with_neg_samples=False):
        if rng is None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = rng

        'Initialization'
        self.n = n
        self.seed = seed
        self.with_neg_samples = with_neg_samples
        np.random.seed(self.seed)
        random.seed(self.seed)
        if data in ['grid', 'grid5']:
            if scale < 0:
                scale = 5.0
            centers_x = [-scale, 0, scale, -scale, 0, scale, -scale, 0, scale]
            centers_y = [-scale, -scale, -scale, 0.0, 0.0, 0.0, scale, scale, scale]
        elif data == 'grid25':
            if scale < 0:
                scale = 5.0
            centers_x = [-scale, -scale/2, 0, scale/2, scale, -scale, -scale/2, 0, scale/2, scale, -scale, -scale/2, 0, scale/2, scale, -scale, -scale/2, 0, scale/2, scale, -scale, -scale/2, 0, scale/2, scale]
            centers_y = [-scale, -scale, -scale, -scale, -scale, -scale/2, -scale/2, -scale/2, -scale/2, -scale/2, 0.0, 0.0, 0.0, 0.0, 0.0, scale/2, scale/2, scale/2, scale/2, scale/2, scale, scale, scale, scale, scale]
        elif data == 'random':
            centers_x = [5.0, 2.5, -4.0, -3.5, -0.0, -2.5, 2.5, 2.5]
            centers_y = [0.5, 4.0, 1.5, -1.5, -0.5, -3.0, -2.5, -1.0]
        elif data == 'random2':
            std = std / 2
            centers_x = [5.0, 2.5, -4.0, -3.5, -0.0, -2.5, 2.5, 2.5]
            centers_y = [0.5, 4.0, 1.5, -1.5, -0.5, -3.0, -2.5, -1.0]
        elif data == 'random9-3_1':
            # std = std
            centers_x = [2.6586781753236854, -3.4416214011233306, -3.483274827067482, 0.9503000540800857, 4.900287150405147, 3.817694173540346, 4.711804246059916, 2.5823126727923285, 3.086367448036949]
            centers_y = [-3.894350616335017, -3.9739054036403387, 2.176015734369308, -3.9227266671218652, -3.1094512212697634, 3.55427342928569, 1.0184759620538983, 3.6695163514368527, 2.73576919691196]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-3_2':
            centers_x = [-0.8464666932481659, -2.80882099265419, 4.466237877195724, 2.562290710831122, -4.8240433269941665, -3.716744013984127, -3.7288118034317472, 3.7916975588333806, -2.3724813363408224]
            centers_y = [4.448142561273555, 1.2847633766038165, 4.872986194894663, -4.786832591107169, -3.9682280091795707, 4.332019943481557, 2.0558569719058557, -4.149320739863258, 4.090451029471874]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-6_1':
            centers_x = [-3.7210898569743334, 3.077005482978633, -0.7092544526694455, 2.9680238502551433, 0.5495040398057771, -3.155151389869022, 1.544074944464791, -0.5080617937969247, 3.004302338583427]
            centers_y = [1.4906527217039178, 2.7802635764349413, 4.222174131582877, -4.804332520871881, -1.3030071497650777, -1.6069707048353266, 4.630149601431157, 1.803952430881452, -2.4155960771485665]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-6_2':
            centers_x = [-3.7210898569743334, 3.077005482978633, -0.7092544526694455, 2.9680238502551433, 0.5495040398057771, -3.155151389869022, 1.544074944464791, -0.5080617937969247, 3.004302338583427]
            centers_y = [1.4906527217039178, 2.7802635764349413, 4.222174131582877, -4.804332520871881, -1.3030071497650777, -1.6069707048353266, 4.630149601431157, 1.803952430881452, -2.4155960771485665]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-9_1':
            centers_x = [-3.7210898569743334, 3.077005482978633, -0.7092544526694455, 2.9680238502551433, 0.5495040398057771, -3.155151389869022, 1.544074944464791, -0.5080617937969247, 3.004302338583427]
            centers_y = [1.4906527217039178, 2.7802635764349413, 4.222174131582877, -4.804332520871881, -1.3030071497650777, -1.6069707048353266, 4.630149601431157, 1.803952430881452, -2.4155960771485665]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-9_2':
            centers_x = [-3.7210898569743334, 3.077005482978633, -0.7092544526694455, 2.9680238502551433, 0.5495040398057771, -3.155151389869022, 1.544074944464791, -0.5080617937969247, 3.004302338583427]
            centers_y = [1.4906527217039178, 2.7802635764349413, 4.222174131582877, -4.804332520871881, -1.3030071497650777, -1.6069707048353266, 4.630149601431157, 1.803952430881452, -2.4155960771485665]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-12_1':
            centers_x = [-0.3896705910411882, 4.537860372606776, -0.6833483826314382, -4.678368418480528, 4.6003441745529905, -4.764142993264687, -3.682800664862922, 2.366108066051819, 1.8044222270151247]
            centers_y = [2.0266631249695566, -1.9462050487811888, -2.7078062104712264, -0.9358654464249554, 2.5892472340305037, 3.0267920905020436, -4.967057264572863, -4.977013592070763, 4.963919815474201]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random9-12_2':
            centers_x = [-3.665740454183939, -3.7064758482839535, 2.1374538291416414, 4.514936302622326, 3.1777238267374397, 0.18158530744234547, -0.6403290024212227, -0.2828227610746392, 4.930053976396218]
            centers_y = [1.9874312853741571, -2.498701016260455, -2.0920081054018103, -4.898850966713023, 1.7826951903016948, 4.577269433935546, -4.450003286128652, 0.6659214978457282, 4.96986193224563]
            centers_x = Normalize_range_pm1(centers_x) * scale
            centers_y = Normalize_range_pm1(centers_y) * scale
        elif data == 'random-3_1':
            if std < 0:
                std = 0.025
            centers_x = [2.6586781753236854, -3.4416214011233306, -3.483274827067482, 0.9503000540800857, 4.900287150405147, 3.817694173540346, 4.711804246059916, 2.5823126727923285, 3.086367448036949, -3.902655907392283, 0.06489097020641399, -3.4057863004566102]
            centers_y = [-3.894350616335017, -3.9739054036403387, 2.176015734369308, -3.9227266671218652, -3.1094512212697634, 3.55427342928569, 1.0184759620538983, 3.6695163514368527, 2.73576919691196, -3.6262103652487685, -4.2835126302949735, 0.692790422675702]
        elif data == 'random-3_2':
            if std < 0:
                std = 0.025
            centers_x = [-0.8464666932481659, -2.80882099265419, 4.466237877195724, 2.562290710831122, -4.8240433269941665, -3.716744013984127, -3.7288118034317472, 3.7916975588333806, -2.3724813363408224, -0.6282047725113831, -0.6287640062205266, -4.4092226096035585]
            centers_y = [4.448142561273555, 1.2847633766038165, 4.872986194894663, -4.786832591107169, -3.9682280091795707, 4.332019943481557, 2.0558569719058557, -4.149320739863258, 4.090451029471874, -0.8634029570818065, 3.9915831617579336, 2.4250313285092817]
        elif data == 'random-6_1':
            if std < 0:
                std = 0.025
            centers_x = [-3.7210898569743334, 3.077005482978633, -0.7092544526694455, 2.9680238502551433, 0.5495040398057771, 4.31654070467779, -3.155151389869022, 1.544074944464791, -0.5080617937969247, 3.004302338583427, -2.0770074856199194, 1.7097076307270695]
            centers_y = [1.4906527217039178, 2.7802635764349413, 4.222174131582877, -4.804332520871881, -1.3030071497650777, -3.628410580376528, -1.6069707048353266, 4.630149601431157, 1.803952430881452, -2.4155960771485665, 4.4072205837879075, -0.3325864333310502]
        elif data == 'random-6_2':
            if std < 0:
                std = 0.025
            centers_x = [3.9635200212963877, 4.879212312602693, 2.3278793821325925, 1.9229185257104477, 4.78955900613667, 0.4736698859053874, -0.21900008067917653, -0.18875928269492626, 0.6699982615648548, 3.6579378235390276, -4.839629095966204, 3.4792050586145375]
            centers_y = [-4.82711707396862, 0.10655671195663707, 3.7443229033917405, 2.8324482372090554, -4.1298537145830805, 0.2987304401078239, -1.523887087179936, -3.655159874031084, 1.6374966849451544, 4.564537423198821, 3.0246771290264824, -0.32610473601043743]
        elif data == 'random-9_1':
            if std < 0:
                std = 0.025
            centers_x = [2.648608269752076, -3.9301658353519233, -4.971160809062598, -1.0186646787267062, -1.619682113715366, -3.42457383316904, 3.7010783708372035, -0.6912372483128681, -2.387392020856832, 0.5442981498355604, 0.6328274533209841, 3.439871923599794]
            centers_y = [4.125106937832518, 3.9511891296637955, -1.6248014993087514, 3.110843092551889, -2.987530540422516, -0.4103833080057253, -2.7886445293608886, 4.840824082120644, 3.724439865615267, 2.852459869711482, 0.9019712762591583, -0.9411542994213695]
        elif data == 'random-9_2':
            if std < 0:
                std = 0.025
            centers_x = [4.802201432329607, -4.675081702892124, 4.764912589031665, 4.473611197578238, 0.32298843287298507, -1.7945203754438555, -0.972755609623178, 4.123869111344048, -4.385010100182919, -1.1921764870192675, 0.037512625863778126, 1.9840663855811709]
            centers_y = [1.0944599940472166, 3.07326198024116, 4.603968835899643, -0.5756852422672187, 3.073910526659297, -4.738019654590476, -2.4744652567644843, -3.861397394413496, -3.9092899906971668, 0.775943992963235, -1.2748365367319545, 3.558700640351889]
        elif data == 'random-12_1':
            if std < 0:
                std = 0.025
            centers_x = [-0.3896705910411882, -3.404651938896879, 1.8376887674802687, 4.537860372606776, -0.6833483826314382, 4.865577451432344, -4.678368418480528, 4.6003441745529905, 2.501744516960861, -2.13224575231962, 2.175742166314266, 0.20207397138606176]
            centers_y = [2.0266631249695566, 1.4229390691699741, -0.5968842720745418, -1.9462050487811888, -2.7078062104712264, -4.888967460837864, -0.9358654464249554, 2.5892472340305037, -3.4896306851185366, -0.29877963868704427, 2.538306800143234, 0.31967002917722454]
        elif data == 'random-12_2':
            if std < 0:
                std = 0.025
            centers_x = [0.5551125333352189, 2.7557925786732493, -0.3726724615826216, 4.2127118579815495, -3.470155761916285, 0.14075639984331367, -0.6515888541270325, 4.455939793736576, 4.955090222453187, -3.1980327446911705, -1.1378980816733808, 1.1961263353987306]
            centers_y = [-4.873933695359468, 4.992547204870142, -3.051684612791421, -1.8873416699264114, -3.9388863184746494, -0.4375631962474147, 1.6731380137178862, 0.0009881305792367456, 3.752648989390801, 2.816440332634773, 4.637995917495566, 1.9615040895763656]
        elif data == 'separated':
            centers_x = [-scale, -scale / 2, 0, scale / 2, scale, 0]
            centers_y = [-scale, -scale, -scale, -scale, -scale, scale]

            # centers_x = [-5.0, -2.5, 0.0, 2.5, 5.0, 0.0]
            # centers_y = [5.0, 3.5, 3.0, 3.5, 5.0, -5.0]
        elif data == 'circle':
            centers_x, centers_y = [0.0], [0.0]
            for i in range(3):
                centers_x.append(2.5 * np.cos(i * np.pi * 2 / 3.0))
                centers_y.append(2.5 * np.sin(i * np.pi * 2 / 3.0))
            for i in range(5):
                centers_x.append(5.0 * np.cos(i * np.pi * 2 / 5.0))
                centers_y.append(5.0 * np.sin(i * np.pi * 2 / 5.0))
        elif data == 'circle2':
            if scale < 0:
                scale = 5
            if std < 0:
                std = 0.05
            centers_x, centers_y = [], []
            for i in range(n):
                centers_x.append(scale * np.cos(i * np.pi * 2.0 / n))
                centers_y.append(scale * np.sin(i * np.pi * 2.0 / n))
        elif data == 'circle2c':
            if scale < 0:
                scale = 5
            if std < 0:
                std = 0.05
            centers_x, centers_y = [0.0], [0.0]
            for i in range(n):
                centers_x.append(scale * np.cos(i * np.pi * 2.0 / n))
                centers_y.append(scale * np.sin(i * np.pi * 2.0 / n))
        else:
            raise NotImplementedError

        self.n = len(centers_x)
        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1. / self.n for _ in range(self.n)]

        self.p = p
        self.size = 2
        self.std = std
        self.centers = np.concatenate([centers_x, centers_y], 1)

        if sample_per_mode is not None:
            self.num_samples = sample_per_mode * self.n
            self.ith_center = np.array([[i] * sample_per_mode for i in range(self.n)]).ravel()
        else:
            self.num_samples = num_samples
            self.ith_center = self.rng.choice(self.n, self.num_samples, p=self.p)

        self.sample_centers = self.centers[self.ith_center]
        self.sample_points = self.rng.normal(loc=self.sample_centers, scale=self.std)

        if sample_per_mode is not None:
            """ Draw negative samples """
            self.range_x = np.array([np.min(self.centers[:, 0]), np.max(self.centers[:, 0])])
            self.range_y = np.array([np.min(self.centers[:, 1]), np.max(self.centers[:, 1])])
            self.span_x = self.range_x[1] - self.range_x[0]
            self.span_y = self.range_y[1] - self.range_y[0]

            self.offset_prop_x = 0.5
            self.offset_prop_y = 0.5
            self.neg_samples_raw = np.concatenate([ \
                np.array([[self.rng.uniform() * self.span_x * (1 + 2 * self.offset_prop_x) + self.range_x[0] - self.span_x * self.offset_prop_x for _ in range(5 * sample_per_mode)]]).T, \
                np.array([[self.rng.uniform() * self.span_y * (1 + 2 * self.offset_prop_y) + self.range_y[0] - self.span_y * self.offset_prop_y for _ in range(5 * sample_per_mode)]]).T], axis=1)
            self.centers_tile = self.centers.reshape(-1, 1, 2)
            self.neg_samples_raw_tile = np.tile(self.neg_samples_raw, [self.n, 1, 1])
            self.dist_to_centers = np.linalg.norm(self.neg_samples_raw_tile - self.centers_tile, axis=(2))
            self.shortest_dist_to_centers = np.min(self.dist_to_centers, axis=0)
            self.neg_samples_selected_ind = self.shortest_dist_to_centers > self.std * 3
            self.neg_samples = self.neg_samples_raw[self.neg_samples_selected_ind, :]
            self.pos_samples = self.neg_samples_raw[~self.neg_samples_selected_ind, :]
            if self.neg_samples.shape[0] > sample_per_mode:
                self.neg_samples = self.neg_samples[:sample_per_mode, :]
            self.neg_labels = self.n * np.ones(shape=self.neg_samples.shape[0])

            self.all_samples = np.concatenate([self.sample_points, self.neg_samples], axis=0)
            self.all_labels = np.concatenate([self.ith_center, self.neg_labels])

            from_list = colors.LinearSegmentedColormap.from_list
            # self.cm = from_list(None, plt.get_cmap("tab20")(range(0, self.n + 1)), self.n + 1)
            half_n = self.n // 2
            self.cm = from_list(None, np.concatenate([plt.get_cmap("tab20b")(range(0, half_n)), plt.get_cmap("tab20c")(range(0, self.n + 1 - half_n))], axis=0), self.n + 1)

            self.encoder = LabelEncoder()
            self.encoder.fit(self.all_labels)
            self.all_labels_onehot = np_utils.to_categorical(self.encoder.transform(self.all_labels)).astype(int)

        # switch to random distribution (harder)

    def random_distribution(self, p=None):
        if p is None:
            p = [self.rng.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

        # switch to uniform distribution

    def uniform_distribution(self):
        p = [1. / self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N, with_neg_samples=False):
        if with_neg_samples:
            index = self.rng.permutation(self.all_samples.shape[0])
            return self.all_samples[index[:N]], self.all_labels[index[:N]]
        else:
            index = self.rng.permutation(self.num_samples)
            return self.sample_points[index[:N]], self.ith_center[index[:N]]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.sample_points)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.with_neg_samples:
            return self.all_samples[index], self.all_labels[index]
        else:
            return self.sample_points[index], self.ith_center[index]

if __name__ == "__main__":
    rng = np.random.RandomState(seed=2020)
    syn_dataset = Synthetic_Dataset("grid5", rng, 0.5, sample_per_mode=4)
    print(len(syn_dataset))
    print(syn_dataset.sample_points[:5])
    print(syn_dataset[:5])
    print(syn_dataset.sample(5))

    plt.scatter(syn_dataset.sample_points[:, 0], syn_dataset.sample_points[:, 1])
    plt.axes().set_aspect('equal')
    # plt.show()

    def Get_seed(id):
        return id

    data_loader = DataLoader(syn_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    print("Epoch 0")
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, len(sample_batched), f"\n{sample_batched}")
    print("Epoch 1")
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, len(sample_batched), f"\n{sample_batched}")