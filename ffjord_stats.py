import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

import torch
import torch.optim as optim

import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

#from diagnostics.viz_toy import save_trajectory, trajectory_to_video

# Added packages
import pandas as pd
import numpy as np

from scipy.stats import skew
from scipy.stats import kurtosis as kurt


def standard_error_skew(n):
    return np.sqrt(6*n*(n-1)/((n-2)*(n+1)*(n+3)))

def standard_error_kurt(n):
    return np.sqrt(24*n*(n-1)**2/((n-3)*(n-2)*(n+3)*(n+5)))

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="elu", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float,default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

regularization_fns, regularization_coeffs = create_regularization_fns(args)
model = build_model_tabular(args, 37, regularization_fns).to(device)

# Load model checkpoint
model.load_state_dict(torch.load(os.path.join(args.save, 'checkpt.pth'),map_location=torch.device('cpu'))["state_dict"])

sample_fn, density_fn = get_transforms(model)


t_skew_real_set = []
t_kurt_real_set = []
t_skew_ff_set = []
t_kurt_ff_set = []

n_samples = 2048
np.random.seed(42)

SE_skew = standard_error_skew(n_samples)
SE_kurt = standard_error_kurt(n_samples)

datapath = "power_spectrum.csv"
data = pd.read_csv(datapath)
data.pop("Unnamed: 0")
ell_bins = [f"ell{i}" for i in range(37)]
ell = np.logspace(np.log10(500), np.log10(5000),37)
train_data = data[ell_bins].to_numpy()

centered_data = train_data - np.mean(train_data,axis=0)
BIG_cov = np.cov(train_data.T)
L = np.linalg.cholesky(np.linalg.inv(BIG_cov))
prepro_data = centered_data @ L

for i in range(30):
    # Randomly sample from training data
    ind = np.random.randint(0,prepro_data.shape[0],n_samples)
    smp_real_data = prepro_data[ind]
    real_skew = skew(smp_real_data,axis=0)
    real_kurt = kurt(smp_real_data,axis=0)
    
    z_data = np.random.normal(size=(n_samples,37))
    z_data = torch.from_numpy(z_data).type(torch.float32).to(device)
    x_data = sample_fn(z_data)
    smp_ff_data = x_data.detach().numpy()

    ff_skew = skew(smp_ff_data,axis=0)
    ff_kurt = kurt(smp_ff_data,axis=0)
    
    t_skew_real = real_skew/SE_skew
    t_skew_real_set.append(t_skew_real)
    
    t_kurt_real = real_kurt/SE_kurt
    t_kurt_real_set.append(t_kurt_real)
    
    t_skew_ff = ff_skew/SE_skew
    t_skew_ff_set.append(t_skew_ff)
    
    t_kurt_ff = ff_kurt/SE_kurt
    t_kurt_ff_set.append(t_kurt_ff)

avg_t_skew_real = np.array(t_skew_real_set).mean(axis=0)
std_t_skew_real = np.array(t_skew_real_set).std(axis=0)

avg_t_kurt_real = np.array(t_kurt_real_set).mean(axis=0)
std_t_kurt_real = np.array(t_kurt_real_set).std(axis=0)

avg_t_skew_ff = np.array(t_skew_ff_set).mean(axis=0)
std_t_skew_ff = np.array(t_skew_ff_set).std(axis=0)

avg_t_kurt_ff = np.array(t_kurt_ff_set).mean(axis=0)
std_t_kurt_ff = np.array(t_kurt_ff_set).std(axis=0)


fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(ell,avg_t_skew_real,'r-')
ax[0].fill_between(ell,avg_t_skew_real+std_t_skew_real,avg_t_skew_real-std_t_skew_real,alpha=0.5,color='r')
ax[0].plot(ell,avg_t_skew_ff,'k-')
ax[0].fill_between(ell,avg_t_skew_ff+std_t_skew_ff,avg_t_skew_ff-std_t_skew_ff,alpha=0.5,color='b')

ax[1].plot(ell,avg_t_kurt_real,'r-')
ax[1].fill_between(ell,avg_t_kurt_real+std_t_kurt_real,avg_t_kurt_real-std_t_kurt_real,alpha=0.5,color='r')
ax[1].plot(ell,avg_t_kurt_ff,'k-')
ax[1].fill_between(ell,avg_t_kurt_ff+std_t_kurt_ff,avg_t_kurt_ff-std_t_kurt_ff,alpha=0.5,color='b')

plt.savefig("ffjord_stats.png")