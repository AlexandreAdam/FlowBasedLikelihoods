import numpy as np
import pandas as pd
from astropy.io import fits
try:
    from lenstools import ConvergenceMap
    _lenstools=True
except ImportError:
    _lenstools=False
import os, glob

"""
    Takes a dataset of convergence maps organized as follows:
        -convergence_maps1/...
        -convergence_maps2/...
        -convergence_maps3/...
    And create a dataset of power spectra
"""
cosmology = ["H", "OMEGA_M", "OMEGA_L", "W0", "WA", "Z", "ANGLE"]
ell_bins = [f"ell{i}" for i in range(38 - 1 )]

def main(datapath, output_file):
    ell = np.logspace(np.log10(500), np.log10(5000), 38)  # multipole bin edges
    data = pd.DataFrame(columns=cosmology+ell_bins)
    data.to_csv(output_file) # save columns to file
    # make a single row to receive data
    data = data.append(pd.DataFrame(
        np.zeros(len(cosmology) + len(ell_bins)).reshape(1, -1),
        columns=cosmology+ell_bins))
    for _, dirs, _ in os.walk(datapath):
        for d in dirs:
            for root, _, files in os.walk(os.path.join(datapath, d)):  
                for conv_map_fit in files:
                    try:
                        conv_map = ConvergenceMap.load(os.path.join(root,conv_map_fit))
                        hdul = fits.open(os.path.join(root, conv_map_fit))[0].header
                        for c_param in cosmology:
                            data[c_param] = hdul[c_param]
                        l, power_spectrum = conv_map.powerSpectrum(ell)
                        data[ell_bins] = power_spectrum
                        data.to_csv(output_file, mode="a", header=False)
                    except:
                        continue

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, required=True, help="Path to the root folder of convergence map")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to the output file (csv)")
    args = parser.parse_args()
    main(datapath=args.datapath, output_file=args.output_file)


