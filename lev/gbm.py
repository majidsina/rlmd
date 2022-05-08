"""
title:                  gbm.py
usage:                  python lev/gbm.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Execute optimal leverage experiments for geometric Brownian motion based on
    https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true,
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.100603, and
    https://arxiv.org/pdf/1802.02939.pdf.

    Historical data for major indices, commodities, and currencies is obtained from
    Stooq at https://stooq.com/.

Instructions:
    1. Choose experiment parameters regarding gamble.
    2. Select whether to use GPU depending on VRAM capacity.
    3. Choose action increments for each of the investors.
    4. Run python file and upon completion all data will be placed into the
       `./results/multiverse` and `./docs/figures` directories.
"""

import sys

sys.path.append("./")

import os
import time

import numpy as np
import numpy.typing as npt
import torch as T
from torch.distributions.normal import Normal

NDArrayFloat = npt.NDArray[np.float_]

import tools.plots_multiverse as plots
from lev.lev_exp import gbm_fixed_final_lev, gbm_smart_lev
from tests.input_tests import gbm_tests

# fmt: off

INVESTORS = 1e6             # number of random investors
HORIZON = 5e2               # total time steps
TOP = INVESTORS * 1e-4      # define number of top performers
VALUE_0 = 1e2               # initial portfolio value of each investor

# lists of drifts, volatilities, and names of GBM environments
drift = [0.05, 0.0540025395205692]
vol = [np.sqrt(0.2), 0.1897916175617430]
name = ["gbm_op", "gbm_snp"]

# fmt: on

# do you have >= 40GB of free VRAM for 1e6 INVESTORS?
# if False, still need 40GB of RAM or reduce number of INVESTORS
VRAM = False

# x_l, x_h, x_i = starting value, ending value, increment
# investor 1 leverages (l) across environments
l0_l, l0_h, l0_i = [-1.0, 0.4], [1.0, 4.0], [0.2, 0.4]
l1_l, l1_h, l1_i = [-1.0, 0.2], [1.0, 2.0], [0.2, 0.2]

if __name__ == "__main__":

    path_results = "./results/multiverse/"  # directory for saving numpy arrays
    path_figs = "./results/figs/"  # directory to save figures

    # conduct tests
    gbm_tests(
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        drift,
        vol,
        name,
        VRAM,
        path_results,
        path_figs,
        l0_l,
        l0_h,
        l0_i,
        l1_l,
        l1_h,
        l1_i,
    )

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    if not os.path.exists(path_figs):
        os.makedirs(path_figs)

    device = T.device("cuda:0") if VRAM == True else T.device("cpu")

    INVESTORS = T.tensor(int(INVESTORS), dtype=T.int32, device=device)
    HORIZON = T.tensor(int(HORIZON), dtype=T.int32, device=device)
    VALUE_0 = T.tensor(VALUE_0, device=device)
    TOP = int(TOP) if TOP > 1 else int(1)  # minimum 1 person in the top sample

    for x in range(0, len(drift)):

        LOG_MEAN = T.tensor(drift[x] - vol[x] ** 2 / 2, device=device)
        VOL = T.tensor(vol[x], device=device)

        # RUN EXPERIMENTS

        start_time = time.perf_counter()

        T.manual_seed(420)  # set fixed seed for reproducibility

        probabilites = Normal(LOG_MEAN, VOL)
        outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

        gbm_fixed_final_lev(
            device,
            outcomes,
            TOP,
            VALUE_0,
            lev_low=l0_l[x],
            lev_high=l0_h[x],
            lev_incr=l0_i[x],
        )

        inv1_val_data, inv1_val_data_T = gbm_smart_lev(
            device,
            outcomes,
            INVESTORS,
            HORIZON,
            TOP,
            VALUE_0,
            lev_low=l1_l[x],
            lev_high=l1_h[x],
            lev_incr=l1_i[x],
        )
        np.save(path_results + name[x] + "_inv1_val.npy", inv1_val_data.cpu().numpy())
        np.save(
            path_results + name[x] + "_inv1_val_T.npy", inv1_val_data_T.cpu().numpy()
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(
            "TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
                total_time, total_time / 60, total_time / 3600
            )
        )

        # LOAD EXPERIMENT DATA AND SAVE FIGURES

        inv1_val_data = np.load(path_results + name[x] + "_inv1_val.npy")
        inv1_val_data_T = np.load(path_results + name[x] + "_inv1_val_T.npy")
        plots.plot_inv1(
            inv1_val_data, inv1_val_data_T, 1e40, path_figs + name[x] + "_inv1"
        )
