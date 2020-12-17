import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import lib.transcovariance as tcov
import lib.stat_tests as tests

from sklearn.mixture import GaussianMixture as GMM
import ffjord_args
import os
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular
from scipy.stats import multivariate_normal
from sklearn.decomposition import FastICA, PCA
from sklearn.neighbors import KernelDensity

import torch


def get_ffjord_transforms(model):

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

def sample_ica(n, kde_list, transformer):
    X = np.hstack([kde.sample(n) for kde in kde_list])
    return transformer.inverse_transform(X)

def plot_results(ax,X,skew_list,kurt_list,kl_list,color='r'):
    ft = 14
    mean_skew = np.array(skew_list).mean(axis=0)
    std_skew = np.array(skew_list).std(axis=0)
    mean_kurt = np.array(kurt_list).mean(axis=0)
    std_kurt = np.array(kurt_list).std(axis=0)

    ax[0].plot(ell,mean_skew,color=color)
    ax[0].fill_between(ell,mean_skew+std_skew, mean_skew-std_skew,alpha=0.5,color=color)
    ax[0].set_title("Skewness",fontsize=ft)
    ax[1].plot(ell,mean_kurt,color=color)
    ax[1].fill_between(ell,mean_kurt+std_kurt, mean_kurt-std_kurt,alpha=0.5,color=color)
    ax[1].set_title("Kurtosis",fontsize=ft)

    ax[2].set_title(r"$\epsilon^+$",fontsize=ft)
    tcov.transcovariance_plot(X,ax=ax[2],color=color)

    for ax_i in ax[:3]:
        ax_i.set_xlabel(r"$\ell$",fontsize=ft)

    if color == 'r':
        kl_model = kl_list[0]
        kl_mvn = kl_list[1]
        kl_ref = kl_list[2]

        n1, bins1, patches1 = ax[3].hist(kl_model, 10, density=True, facecolor='g', alpha=0.75,label='mock|DDL')
        n2, bins2, patches2 = ax[3].hist(kl_mvn, 10, density=True, facecolor='r', alpha=0.75,label='mock|MVN')
        n3, bins2, patches3 = ax[3].hist(kl_ref, 10, density=True, facecolor='k', alpha=0.75,label='MVN|MVN')

    ax[3].set_xlabel("KL divergence",fontsize=ft)


rn_ts_set = []
rn_tk_set = []
gmm_ts_set = []
gmm_tk_set = []
gmm_kl = []
ffjord_ts_set = []
ffjord_tk_set = []
ffjord_kl = []
ica_ts_set = []
ica_tk_set = []
ica_kl = []
mock_ts_set = []
mock_tk_set = []
mvn_kl = []
ref_kl = []

n_samples = 2048
np.random.seed(42)

# Load data
datapath = "power_spectrum.csv"
data = pd.read_csv(datapath)
data.pop("Unnamed: 0")
ell_bins = [f"ell{i}" for i in range(37)]
ell = np.logspace(np.log10(500), np.log10(5000),37)
_, unique_idx = np.unique(data[ell_bins].to_numpy(), return_index=True,axis=0)
data = data[ell_bins].to_numpy()[unique_idx]

# Preprocessing of data
centered_data = data - np.mean(data,axis=0)
BIG_cov = np.cov(data.T)
L = np.linalg.cholesky(np.linalg.inv(BIG_cov))
prepro_data = centered_data @ L

# Save Cholesky matrix and centered data
np.save("L_matrix.npy",L)
np.save("centered_data.npy",centered_data)

# Divide in training testing and validation set
train_data = prepro_data[:int(0.8*prepro_data.shape[0])]
test_data = prepro_data[int(0.8*prepro_data.shape[0]):int(0.9*prepro_data.shape[0])]
val_data = prepro_data[int(0.9*prepro_data.shape[0]):]

print("Number of data samples :",prepro_data.shape[0])

# Load ffjord model
args = ffjord_args.get_default_args()
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
regularization_fns, regularization_coeffs = create_regularization_fns(args)
model = build_model_tabular(args, 37,regularization_fns).to(device)
model.load_state_dict(torch.load(os.path.join(args.save, 'checkpt.pth'),map_location=torch.device('cpu'))["state_dict"])

# Get ffjord transformations
sample_fn, density_fn = get_ffjord_transforms(model)


# Fit GMM model on data
gmm = GMM(n_components=2)
gmm.fit(train_data)

# Fit ICA model
transformer = FastICA(n_components=37, random_state=0, tol=0.001, max_iter=300)
#transformer = PCA(n_components=37)
X = transformer.fit_transform(prepro_data)

kde_list = []

for i in range(X.shape[1]):
    X_transformed = KernelDensity(kernel='gaussian', 
                                  bandwidth=X.shape[0]**(-1/(X.shape[1]+4))).fit(X[:, i].reshape(-1, 1))
    kde_list.append(X_transformed)
kde_list = np.array(kde_list)

try:
    t_stats_data = np.load("t_stats_data.npy")
    gmm_ts_set = list(t_stats_data[0])
    gmm_tk_set = list(t_stats_data[1])
    ffjord_ts_set = list(t_stats_data[2])
    ffjord_tk_set = list(t_stats_data[3])
    mock_ts_set = list(t_stats_data[4])
    mock_tk_set = list(t_stats_data[5])
    ica_ts_set = list(t_stats_data[6])
    ica_tk_set = list(t_stats_data[7])
    rn_ts_set = list(t_stats_data[8])
    rn_tk_set = list(t_stats_data[9])
except:
    pass
    print("Did not find t-statistics data file. Re-evaluating t-statistics of all models.")
    for i in range(30):
        # Randomly sample from normal distribution
        rn_sample = np.random.normal(size=(n_samples, 37))
        # Randomly sample from training data
        ind = np.random.randint(0,test_data.shape[0],n_samples)
        mock_sample = test_data[ind]
        # Randomly sample from GMM
        gmm_sample = gmm.sample(n_samples=n_samples)[0]
        # Randomly sample from ffjord
        z_data = np.random.normal(size=(n_samples,37))
        z_data = torch.from_numpy(z_data).type(torch.float32).to(device)
        x_data = sample_fn(z_data)
        ffjord_sample = x_data.detach().numpy()
        # Randomly sample from ICA
        ica_sample = sample_ica(n_samples, kde_list, transformer)

        rn_ts,rn_tk = tests.t_stats(rn_sample)
        gmm_ts,gmm_tk = tests.t_stats(gmm_sample)
        ffjord_ts,ffjord_tk = tests.t_stats(ffjord_sample)
        mock_ts,mock_tk = tests.t_stats(mock_sample)
        ica_ts,ica_tk = tests.t_stats(ica_sample)

        rn_ts_set.append(rn_ts)
        rn_tk_set.append(rn_tk)
        gmm_ts_set.append(gmm_ts)
        gmm_tk_set.append(gmm_tk)
        ffjord_ts_set.append(ffjord_ts)
        ffjord_tk_set.append(ffjord_tk)
        mock_ts_set.append(mock_ts)
        mock_tk_set.append(mock_tk)
        ica_ts_set.append(ica_ts)
        ica_tk_set.append(ica_tk)

    np.save("t_stats_data.npy",np.array([gmm_ts_set,gmm_tk_set,ffjord_ts_set,
        ffjord_tk_set,mock_ts_set,mock_tk_set,ica_ts_set,ica_tk_set,rn_ts_set,rn_tk_set]))

try:
    kl_data = np.load("kl_data.npy")
    gmm_kl = list(kl_data[0])
    ffjord_kl = list(kl_data[1])
    mvn_kl = list(kl_data[2])
    ref_kl = list(kl_data[3])
    ica_kl = list(kl_data[4])
except:
    pass
    print("Did not find kl data file. Re-evaluating kl divergence of all models.")
    for i in range(2):
        if i%5==0: print(str(i)+'...')
        
        mvn1_sample = multivariate_normal.rvs(mean=np.mean(prepro_data,axis=0),
                                         cov=np.cov(prepro_data.T),
                                         size=n_samples)
        
        mvn2_sample = multivariate_normal.rvs(mean=np.mean(prepro_data,axis=0),
                                     cov=np.cov(prepro_data.T),
                                     size=n_samples)
        
        ind = np.random.randint(0,test_data.shape[0],n_samples)
        mock_sample = test_data[ind]

        gmm_sample = gmm.sample(n_samples=n_samples)[0]

        z_data = np.random.normal(size=(n_samples,37))
        z_data = torch.from_numpy(z_data).type(torch.float32).to(device)
        x_data = sample_fn(z_data)
        ffjord_sample = x_data.detach().numpy()

        ica_sample = sample_ica(n_samples, kde_list, transformer)
        
        ica_kl.append(tests.KL(mock_sample,ica_sample))
        ffjord_kl.append(tests.KL(mock_sample,ffjord_sample))
        gmm_kl.append(tests.KL(mock_sample,gmm_sample))
        mvn_kl.append(tests.KL(mock_sample,mvn1_sample))
        ref_kl.append(tests.KL(mvn1_sample,mvn2_sample))
    np.save("kl_data.npy",np.array([gmm_kl,ffjord_kl,mvn_kl,ref_kl,ica_kl]))


t_ft = 14
w = 22
h=4
lft = 14
fig, ax = plt.subplots(1, 4, figsize=(w, h))
plt.rc('xtick',labelsize=lft)
plt.rc('ytick',labelsize=lft)
fig.suptitle('GMM', fontsize=t_ft)
plot_results(ax,gmm_sample,gmm_ts_set,gmm_tk_set,[gmm_kl,mvn_kl,ref_kl])
plot_results(ax,mock_sample,mock_ts_set,mock_tk_set,None,color='b')
plot_results(ax,rn_sample,rn_ts_set,rn_tk_set,None,color='gray')
plt.savefig("gmm_results.png",bbox_inches='tight')

fig, ax = plt.subplots(1, 4, figsize=(w, h))
plt.rc('xtick',labelsize=lft)
plt.rc('ytick',labelsize=lft)
fig.suptitle('FFJORD', fontsize=t_ft)
plot_results(ax,ffjord_sample,ffjord_ts_set,ffjord_tk_set,[ffjord_kl,mvn_kl,ref_kl])
plot_results(ax,mock_sample,mock_ts_set,mock_tk_set,None,color='b')
plot_results(ax,rn_sample,rn_ts_set,rn_tk_set,None,color='gray')
plt.savefig("ffjord_results.png",bbox_inches='tight')

fig, ax = plt.subplots(1, 4, figsize=(w, h))
plt.rc('xtick',labelsize=lft)
plt.rc('ytick',labelsize=lft)
fig.suptitle('ICA', fontsize=t_ft)
plot_results(ax,ica_sample,ica_ts_set,ica_tk_set,[ica_kl,mvn_kl,ref_kl])
plot_results(ax,mock_sample,mock_ts_set,mock_tk_set,None,color='b')
plot_results(ax,rn_sample,rn_ts_set,rn_tk_set,None,color='gray')
plt.savefig("ica_results.png",bbox_inches='tight')

