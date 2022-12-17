import os
import io
import cv2
import glob
import time
import json
import shutil
import imageio
import colorsys
import argparse
import multiprocessing as mp
from os import path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd

import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
import geoopt # TODO: use geomstats

from data import sample_2d

# plt.style.use('seaborn')
EPS = 1e-6

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, choices=['2spirals', 'checkerboard', 'rings', '8gaussians'])
parser.add_argument('--manifold', type=str, choices=['euclidean', 'sphere', 'torus'])
parser.add_argument('--dataset-size', type=int, default=int(1e6))
parser.add_argument('--min_noise', type=float, default=0.01)
parser.add_argument('--max_noise', type=float, default=1)
parser.add_argument('--seed', type=int, default=6597103364)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENT_ID = '%s_%s_%s' % (args.dataset, args.manifold, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

TOY_DATASETS = {'2spirals', 'checkerboard', 'rings', '8gaussians'}

data_loader_kwargs = dict(shuffle=True, batch_size=args.batch_size, num_workers=4, drop_last=True)
train_loader = torch.utils.data.DataLoader(sample_2d(args.dataset, 100000), **data_loader_kwargs)

if args.manifold == 'sphere':
    manifold = geoopt.Sphere()
elif args.manifold == 'torus':
    sphere = geoopt.Sphere()
    manifold = geoopt.ProductManifold((sphere, 2), (sphere, 2))
elif args.manifold == 'euclidean':
    manifold = geoopt.Euclidean(ndim=2)
else:
    assert False


def project_on_sphere(x, r=5):
    z = torch.sqrt(r ** 2 - x[:, 0] ** 2 - x[:, 1] ** 2)
    return torch.stack((x[:, 0], x[:, 1], z), dim=1) / r


def plane_to_torus(x, r=4):
    x1 = np.sin(x[:, 0] / r * np.pi)
    y1 = np.cos(x[:, 0] / r * np.pi)
    x2 = np.sin(x[:, 1] / r * np.pi)
    y2 = np.cos(x[:, 1] / r * np.pi)
    return torch.stack((x1, y1, x2, y2), dim=1)


def torus_to_plane(x, r=4):
    x1 = np.arctan2(x[:, 0], x[:, 1]) / np.pi * r
    x2 = np.arctan2(x[:, 2], x[:, 3]) / np.pi * r
    return torch.stack((x1, x2), dim=1)


# add Gaussian noise to a batch of data points
def add_noise(x, noise_strengths, manifold):
    # if noise_strengths is not a tensor, make it a tensor
    if not isinstance(noise_strengths, torch.Tensor):
        noise_strengths = torch.tensor(noise_strengths, device=x.device)
    noise_strengths = noise_strengths * torch.ones(x.shape[0:1], device=x.device)
    assert x.shape[0:1] == noise_strengths.shape
    B = x.shape[0]
    # dim = np.prod(x.shape[1:])
    orig_shape = x.shape
    x = x.view(B, -1)
    noise_strengths = noise_strengths.view(B, 1)
    noise = torch.randn_like(x) * noise_strengths
    O = manifold.origin(noise.shape, device=x.device)
    noise = manifold.proju(O, noise)
    noise = manifold.transp(O, x, noise)
    noisy_x = manifold.expmap(x, noise)
    return noisy_x.view(orig_shape), noise


def regular_points_on_sphere(n):
    u, v = np.mgrid[0:2*np.pi:n * 1j, 0:np.pi:n * 1j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    return x, y, z


class Network(nn.Module):
    ''' Wraps around a neural network and adds methods such as `save`, `load`, and `run`. '''
    def __init__(self, net_body):
        super(Network, self).__init__()
        self.net_body = net_body

    def save(self, path, log=True):
        torch.save(self.state_dict(), path)
        if log:
            print('Saved model to `%s`' % path)

    def load(self, path, log=True, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))
        if log:
            print('Loaded model from `%s`' % path)

    def get_device(self):
        '''
        Returns the `torch.device` on which the network resides.
        This method only makes sense when all module parameters reside on the **same** device.
        '''
        return list(self.parameters())[0].device

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, *args, **kwargs):
        return self.net_body(*args, **kwargs)


if args.dataset in TOY_DATASETS:
    class EnergyNet(nn.Module):
        def __init__(self, in_width, manifold, act_fn=F.relu):
            super().__init__()
            self.manifold = manifold
            self.act_fn = act_fn
            self.fc_list = nn.ModuleList([
                nn.Linear(in_width + 1, 500),
                nn.Linear(500, 500),
                nn.Linear(500, 500),
                nn.Linear(500, 1)
            ])

        def forward(self, x, t):
            x = manifold.projx(x)
            t = t * torch.ones(x.shape[0:1], device=x.device)
            x = torch.concat([x, t[:, None]], dim=1)
            depth = len(self.fc_list)
            for i in range(depth):
                x = self.fc_list[i](x)
                x = self.act_fn(x) if i < depth - 1 else x
            return x

    if args.manifold == 'euclidean':
        in_width = 2
    elif args.manifold == 'torus':
        in_width = 4
    elif args.manifold == 'sphere':
        in_width = 3
    else:
        assert False

    net = Network(EnergyNet(in_width=in_width, manifold=manifold)).to(DEVICE)


print('Number of parameters: %d' % net.count_parameters())
print(net)


def plot_score_sphere(ax, net, t, manifold, **kwargs): # TODO: why does this take manifold as input? fix it.
    _kwargs = {
        'length': 0.1,
        'normalize': True,
        'arrow_length_ratio': 0.4,
        'linewidth': 0.4
    }
    _kwargs.update(kwargs)
    x = manifold.random_uniform((1000, 3))
    x, y, z = regular_points_on_sphere(30)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    data = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
    data = torch.tensor(data, dtype=torch.float32)
    data = data.requires_grad_()
    u = -torch.autograd.grad(net(data, t).sum(), data, create_graph=False)[0]
    u = manifold.proju(x=data, u=u)
    u = u.detach().cpu().numpy()
    ax.quiver(x, y, z, u[:, 0], u[:, 1], u[:, 2], **_kwargs)
    ax.set_title('Estimated Score (t=%.2f)' % t)


def plot_energy(ax, net, t, W=6.9):
    from numpy import arange
    from pylab import meshgrid, cm, imshow, colorbar, title, show
    x = arange(-W, W, 0.1)
    y = arange(-W, W, 0.1)
    k = x.shape[0]
    X, Y = meshgrid(x, y)
    data = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        t_tensor = torch.ones(data.shape[0], device=DEVICE) * t
        Z = net(data, t_tensor).cpu().numpy().reshape(k, k)
    im = ax.imshow(Z, cmap=cm.RdBu, extent=[-W, W, -W, W], origin='lower')
    ax.set_title('Estimated Energy (t=%.2f)' % t)


def plot_energy_torus(ax, net, t, W=4):
    from numpy import arange
    from pylab import meshgrid, cm, imshow, colorbar, title, show
    x = arange(-W, W, 0.1)
    y = arange(-W, W, 0.1)
    k = x.shape[0]
    X, Y = meshgrid(x, y)
    data = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        data = plane_to_torus(data)
        t_tensor = torch.ones(data.shape[0], device=DEVICE) * t
        Z = net(data, t_tensor).cpu().numpy().reshape(k, k)
    im = ax.imshow(Z, cmap=cm.RdBu, extent=[-W, W, -W, W], origin='lower')
    # ax.set_title('Estimated Energy (t=%.2f)' % t)


def plot_energy_hemisphere(ax, net, t):
    from numpy import arange
    from pylab import meshgrid, cm, imshow, colorbar, title, show
    x = arange(-1.1, 1.1, 0.02)
    y = arange(-1.1, 1.1, 0.02)
    k = x.shape[0]
    X, Y = meshgrid(x, y)
    Z = np.sqrt(np.maximum(1 - X ** 2 - Y ** 2, 0))
    data = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], axis=1)
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32, device=DEVICE)
        t_tensor = torch.ones(data.shape[0], device=DEVICE) * t
        Z = net(data, t_tensor).cpu().numpy().reshape(k, k)
    im = ax.imshow(Z, cmap=cm.RdBu, extent=[-1.1, 1.1, -1.1, 1.1], origin='lower')
    # ax.set_title('Estimated Energy (t=%.2f)' % t)


def plot_score(ax, net, t, **kwargs):
    _kwargs = {
        'length': 0.1,
        'normalize': True,
        'arrow_length_ratio': 0.4,
        'linewidth': 0.4
    }
    _kwargs.update(kwargs)
    x = torch.randn(1000, 2, device=DEVICE)
    x, y = regular_points_on_sphere(30)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    data = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
    data = torch.tensor(data, dtype=torch.float32)
    data = data.requires_grad_()
    u = -torch.autograd.grad(net(data, t).sum(), data, create_graph=False)[0]
    u = manifold.proju(x=data, u=u)
    u = u.detach().cpu().numpy()
    ax.quiver(x, y, z, u[:, 0], u[:, 1], u[:, 2], **_kwargs)
    # ax.set_title('Estimated Score (t=%.2f)' % t)


def plot_samples(x_hat, net, t, manifold_name, save_path=None, plot_data=True, plot_energy=True, display=False):
    g_cpu = torch.Generator().manual_seed(args.seed)
    x_data = sample_2d(args.dataset, 1000, generator=g_cpu)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    if plot_energy:
        if manifold_name == 'euclidean':
            plot_energy(ax, net, t)
        elif manifold_name == 'torus':
            plot_energy_torus(ax, net, t)
        elif manifold_name == 'sphere':
            plot_energy_hemisphere(ax, net, t)

    if manifold_name == 'torus':
        x_hat = torus_to_plane(x_hat)
    elif manifold_name == 'sphere':
        x_data = project_on_sphere(x_data)

    ax.scatter(x_hat[:, 0], x_hat[:, 1], s=2)
    if plot_data:
        ax.scatter(x_data[:, 0], x_data[:, 1], s=2)
    
    if manifold_name == 'euclidean':
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
    elif manifold_name == 'torus':
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    elif manifold_name == 'sphere':
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    
    if display:
        plt.show()

    plt.close()


def plot_energies(manifold_name, save_path=None):
    if manifold_name == 'sphere':
        plot_fn = plot_energy_hemisphere
    elif manifold_name == 'euclidean':
        plot_fn = plot_energy
    elif manifold_name == 'torus':
        plot_fn = plot_energy_torus
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    plot_fn(ax[0], net, args.min_noise)
    plot_fn(ax[1], net, (args.min_noise + args.max_noise) / 2)
    plot_fn(ax[2], net, args.max_noise)
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.show()
    plt.close()


optimizer = optim.Adam(net.parameters(), lr=args.lr)

# energy_dir = 'progress_energy_%s_%s' % (args.dataset, args.manifold)
# score_dir = 'progress_score_%s_%s' % (args.dataset, args.manifold)

# shutil.rmtree(energy_dir, ignore_errors=True)
# shutil.rmtree(score_dir, ignore_errors=True)

# os.makedirs(energy_dir, exist_ok=True)
# os.makedirs(score_dir, exist_ok=True)

epoch = 0
loss_list = []

pbar = tqdm(range(args.epochs), desc='Loss: None', total=args.epochs, position=epoch, leave=True)
while True:
    # plot_energies(args.manifold, save_path=path.join(energy_dir, '%05d.png' % epoch))

    if epoch == args.epochs:
        break

    cnt = 0
    running_loss = 0.0
    for x in train_loader:
        if args.dataset == 'mnist':
            x = x[0]
        x = x.to(DEVICE)

        if args.manifold == 'sphere':
            x = project_on_sphere(x)
        elif args.manifold == 'torus':
            x = plane_to_torus(x)
        elif args.manifold == 'euclidean':
            pass
        else:
            raise ValueError('Unknown manifold %s' % args.manifold)

        t = args.min_noise + (args.max_noise - args.min_noise) * torch.rand(x.shape[0], device=DEVICE)
        noisy_x, noise = add_noise(x, noise_strengths=t, manifold=manifold)
        noisy_x = noisy_x.requires_grad_()
        energy_pred = net(noisy_x, t)
        score = -torch.autograd.grad(energy_pred.sum(), noisy_x, create_graph=True)[0]
        score = manifold.proju(x=noisy_x, u=score)
        t_ = t.unsqueeze(1)
        loss = torch.sum((t_ * score + noise / t_) ** 2) / (2 * args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cnt += 1
        running_loss += loss.item()

    loss_list.append(running_loss / cnt)
    pbar.set_description(f'Loss: {loss_list[-1]:.4f}')
    pbar.update(1)
    # net.save('ebm_%s_%s.tar' % (args.manifold, args.dataset))
    epoch += 1

pbar.close()

# plt.style.use('seaborn')
# fig = plt.figure(figsize=(4, 4))
# ax = fig.add_subplot(111)
# ax.plot(loss_list)
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Loss')
# plt.show()

# for s in [energy_dir, score_dir]:
#     image_list = []
#     files = sorted(glob.glob('%s/*.png' % s))
#     for file in files:
#         image_list.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
#     imageio.mimsave('%s.gif' % s, image_list, fps=10)

# net.load('ebm_%s_%s.tar' % (args.manifold, args.dataset))


def langevin_step_logp(x_, x, t, net, step_size, manifold):
    x = x.requires_grad_()
    energy_pred = net(x, t)
    score = -torch.autograd.grad(energy_pred.sum(), x, create_graph=True)[0]
    if manifold is not None:
        score = manifold.proju(x=noisy_x, u=score)
    next_dist = torch.distributions.normal.Normal(loc=x + 0.5 * step_size * score, scale=np.sqrt(step_size))
    return next_dist.log_prob(x_).sum(dim=1, keepdim=True)


def unadjusted_langevin_step(x, t, net, step_size, manifold):
    t = torch.ones(x.shape[0], device=x.device) * t
    x = x.requires_grad_()
    energy_pred = net(x, t)
    score = -torch.autograd.grad(energy_pred.sum(), x, create_graph=True)[0]
    score = manifold.proju(x=x, u=score)
    step = 0.5 * step_size * score
    x = manifold.expmap(x, step)
    x, _ = add_noise(x, np.sqrt(step_size), manifold)
    x = manifold.projx(x)
    return x.detach()


def metropolis_adjusted_langevin_step(x, t, net, step_size):
    t = torch.ones(x.shape[0], device=x.device) * t
    x_ = unadjusted_langevin_step(x, t, net, step_size)
    u = torch.rand((x.shape[0], 1), device=DEVICE)
    k = langevin_step_logp(x, x_, t, net, step_size) + net(x, t) - langevin_step_logp(x_, x, t, net, step_size) - net(x_, t)
    c = (torch.exp(k) < u) * 1.0
    x_ = c * x_ + (1 - c) * x 
    return x_.detach()


def tweedie(x, net, t, manifold):
    # t is assumed to be standard deviation of Gaussian noise
    t = torch.ones(x.shape[0], device=x.device) * t
    x = x.requires_grad_()
    energy_pred = net(x, t)
    score = -torch.autograd.grad(energy_pred.sum(), x, create_graph=True)[0]
    score = manifold.proju(x=x, u=score)
    step = (t ** 2)[:, None] * score
    x = manifold.expmap(x, step)
    return x


def annealed_langevin_sample(x0, t_sched, net, step_size_sched, n_steps, save_dir, manifold, metropolis_adjusted=False, use_tweedie=True):
    x = x0
    for step in tqdm(range(n_steps)):
        # plot_samples(x, net, t_sched(step), args.manifold, path.join(save_dir, '%05d.png' % step))
        if metropolis_adjusted:
            x = metropolis_adjusted_langevin_step(x, t_sched(step), net, step_size_sched(step))
        else:
            x = unadjusted_langevin_step(x, t_sched(step), net, step_size_sched(step), manifold)
        assert torch.allclose(manifold.projx(x), x)

    if use_tweedie:
        x = tweedie(x, net, t_sched(n_steps), manifold)

    return x


sampling_dir = 'sampling_%s_%s' % (args.dataset, args.manifold)
shutil.rmtree(sampling_dir, ignore_errors=True)
os.makedirs(sampling_dir, exist_ok=True)

n = 1000
T = 240
# t_sched = lambda step: (args.min_noise / args.max_noise) ** ((step // T) / (L - 1)) * args.max_noise

if args.manifold == 'euclidean':
    x0 = torch.rand((n, 2), device=DEVICE) * 8 - 4
else:
    if args.manifold == 'sphere':
        d = 3
    elif args.manifold == 'torus':
        d = 4
    else:
        assert False
    x0 = manifold.random((n, d)).to(DEVICE)

x_hat = annealed_langevin_sample(
    x0=x0,
    t_sched=lambda step: np.maximum(1e-3, ((T - 2 * step) * args.max_noise) / T), # args.max_noise / (step + 1),
    net=net,
    step_size_sched=lambda _: 0.01, #lambda step: eps * (t_sched(step) / args.min_noise) ** 2,
    n_steps=T,
    save_dir=sampling_dir,
    manifold=manifold,
    metropolis_adjusted=False
).cpu()


s = sampling_dir
# image_list = []
# files = sorted(glob.glob('%s/*.png' % s))
# for file in files:
#     image_list.append(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))
# imageio.mimsave('%s.gif' % s, image_list, fps=24)

plot_samples(x_hat, net, EPS, args.manifold, 'samples_%s_%s.png' % (args.dataset, args.manifold), plot_data=True, display=False)
# plot_samples(x_hat, net, EPS, args.manifold, 'samples_%s_%s_no_energy.png' % (args.dataset, args.manifold), plot_data=True, plot_energy=False, display=True)
