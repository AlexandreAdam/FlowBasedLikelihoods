import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from definitions import ell



def main(datapath):
    data = data = pd.read_csv(datapath)
    ell_bins = [f"ell{i}" for i in range(len(ell))]

    mean = data[ell_bins].mean(axis=0)
    std = data[ell_bins].std(axis=0)

    c1 = ell * (ell + 1) * (mean + std)/ 2/np.pi
    c1_l = ell * (ell + 1) * (mean - std)/2/np.pi
    c2 = ell * (ell + 1) * (mean + 2* std)/2/np.pi
    c2_l = ell * (ell + 1) * (mean - 2 * std)/2/np.pi

    plt.figure()
#    plt.title("Power spectra", fontsize=13)
    plt.plot(ell, ell * (ell + 1) * mean/2/np.pi, "k-")
    plt.plot(ell, c1, "k-",lw=.5,alpha=0.8)
    plt.plot(ell, c1_l, "k-",lw=.5,alpha=0.8)
    plt.plot(ell, c2, "k-",lw=.5,alpha=0.6)
    plt.plot(ell, c2_l, "k-",lw=.5,alpha=0.6)
    plt.xlabel(r"$\ell$", fontsize=14)
    plt.ylabel(r"$\ell  (\ell + 1)  P_\ell / 2 \pi$", fontsize=14)
    
    plt.xlim(ell.min(),ell.max())

    plt.fill_between(ell, c1, c1_l, color="royalblue", alpha=0.6)
    plt.fill_between(ell, c2, c2_l, color="cornflowerblue", alpha=0.4)

    plt.savefig("../results/confidence_intervals.png", bbox_inches="tight")
    plt.show()





if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", required=False, 
            type=str, default="../power_spectrum.csv", help="path to data, should be csv file")
    args = parser.parse_args()
    main(args.data)
