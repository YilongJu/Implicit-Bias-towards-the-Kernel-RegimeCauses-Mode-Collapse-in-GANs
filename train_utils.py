import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, LSUN
from GANs import dc_G, dc_D, \
    GoodGenerator, GoodDiscriminator, GoodDiscriminatorbn, GoodDiscriminatord, \
    DC_generator, DC_discriminator, \
    ResNet32Discriminator, ResNet32Generator, DC_discriminatorW, GoodSNDiscriminator


mnist_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                               ])


def generate_data(model_weight, path, z_dim=96, device='cpu'):
    chk = torch.load(model_weight)
    print('load from %s' % model_weight)
    dataset = get_data(dataname='MNIST', path='../datas/mnist')
    fixed_z = torch.randn((500, z_dim), device=device)
    fixed_D = dc_D().to(device)
    fixed_G = dc_G(z_dim=z_dim).to(device)
    fixed_D.load_state_dict(chk['D'])
    fixed_G.load_state_dict(chk['G'])
    real_loader = DataLoader(dataset=dataset, batch_size=500, shuffle=True,
                             num_workers=4)
    real_set = next(iter(real_loader))
    real_set = real_set[0].to(device)
    with torch.no_grad():
        fake_set = fixed_G(fixed_z)
        fixed_real_d = fixed_D(real_set)
        fixed_fake_d = fixed_D(fake_set)
        fixed_vec = torch.cat([fixed_real_d, fixed_fake_d])
    if not os.path.exists('figs/select'):
        os.makedirs('figs/select')
    torch.save({'real_set': real_set,
                'fake_set': fake_set,
                'real_d': fixed_real_d,
                'fake_d': fixed_fake_d,
                'pred_vec': fixed_vec}, path)
    for i in range(5):
        j = i * 100
        vutils.save_image(real_set[j: j + 100], 'figs/select/real_set_%d.png' % i, nrow=10, normalize=True)
        vutils.save_image(fake_set[j: j + 100], 'figs/select/fake_set_%d.png' % i, nrow=10, normalize=True)


class icrScheduler(object):
    def __init__(self, optimizer, milestone):
        self.optim = optimizer
        self.milestone = milestone

    def step(self, epoch):
        e_key = str(epoch)
        if e_key in self.milestone:
            self.optim.set_state({'lr': self.milestone[e_key][0],
                                  'alpha': self.milestone[e_key][1]})


class lr_scheduler(object):
    def __init__(self, optimizer, milestone):
        self.optim = optimizer
        self.milestone = milestone

    def step(self, epoch, gamma=10):
        e_key = str(epoch)
        if e_key in self.milestone:
            self.optim.set_lr(lr_max=self.milestone[e_key][0],
                              lr_min=self.milestone[e_key][1])


def get_diff(net, model_vec):
    current_model = torch.cat([p.contiguous().view(-1) for p in net.parameters()]).detach()
    weight_vec = current_model - model_vec
    vom = torch.norm(weight_vec, p=2)
    return vom


def get_model(model_name, z_dim):
    if model_name == 'dc':
        D = GoodDiscriminator()
        G = GoodGenerator()
    elif model_name == 'dcBN':
        D = GoodDiscriminatorbn()
        G = GoodGenerator()
    elif model_name == 'dcD':
        D = GoodDiscriminatord()
        G = GoodGenerator()
    elif model_name == 'DCGAN':
        D = DC_discriminator()
        G = DC_generator(z_dim=z_dim)
    elif model_name == 'Resnet':
        D = ResNet32Discriminator(n_in=3, num_filters=128, batchnorm=True)
        G = ResNet32Generator(z_dim=z_dim, num_filters=128, batchnorm=True)
    elif model_name == 'ResnetWBN':
        D = ResNet32Discriminator(n_in=3, num_filters=128, batchnorm=False)
        G = ResNet32Generator(z_dim=z_dim, num_filters=128, batchnorm=True)
    elif model_name == 'DCGAN-WBN':
        D = DC_discriminatorW()
        G = DC_generator(z_dim=z_dim)
    elif model_name == 'dcSN':
        D = GoodSNDiscriminator()
        G = GoodGenerator()
    else:
        print('No matching result of :')
    print(model_name)
    return D, G


def get_data(dataname, path, img_size=64):
    if dataname == 'CIFAR10':
        dataset = CIFAR10(path, train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ]),
                          download=True)
        print('CIFAR10')
    elif dataname == 'MNIST':
        dataset = MNIST(path, train=True,
                        transform=transforms.Compose([transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                      ]),
                        download=True)
        print('MNIST')
    elif dataname == 'LSUN-dining':
        dataset = LSUN(path, classes=['dining_room_train'], transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        print('LSUN-dining')
    elif dataname == 'LSUN-bedroom':
        dataset = LSUN(path, classes=['bedroom_train'], transform=transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
        print('LSUN-bedroom')
    return dataset


def save_checkpoint(path, name, D, G, optimizer=None, g_optimizer=None):
    chk_name = 'checkpoints/%s/' % path
    if not os.path.exists(chk_name):
        os.makedirs(chk_name)
    try:
        d_state_dict = D.module.state_dict()
        g_state_dict = G.module.state_dict()
    except AttributeError:
        d_state_dict = D.state_dict()
        g_state_dict = G.state_dict()
    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0
    if g_optimizer is not None:
        g_optim_dict = g_optimizer.state_dict()
        torch.save({
            'D': d_state_dict,
            'G': g_state_dict,
            'd_optim': optim_dict,
            'g_optim': g_optim_dict,
        }, chk_name + name)
    else:
        torch.save({
            'D': d_state_dict,
            'G': g_state_dict,
            'optim': optim_dict
        }, chk_name + name)
    print('model is saved at %s' % chk_name + name)


def detransform(x):
    return (x + 1.0) / 2.0


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.01)


def model_init(D=None, G=None, init_d=False, init_g=False):
    if D is not None and init_d:
        D.apply(weights_init_d)
        print('initial D with normal')
    if G is not None and init_g:
        G.apply(weights_init_g)
        print('initial G with normal')


def select_n_random(data, n=100):
    """
    Selects n random data points and their corresponding labels from a dataset
    """
    perm = torch.randperm(len(data))
    return data[perm][:n]