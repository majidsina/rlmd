"""
title:                  lev_tests.py
usage:                  python tests/lev_tests.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Conduct tests across all optimal leverage experiments operating at a reduced
    scale with automatic clean-up of tests files.

Instructions:
    1. Select hyperparameters for tests.
    2. Running the file will provide live progress in the terminal.
"""

import sys

sys.path.append("./")

import os
import shutil
import time

import numpy as np
import numpy.typing as npt
import torch as T
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

NDArrayFloat = npt.NDArray[np.float_]

import plotting.plots_multiverse as plots
from lev.lev_exp import (
    coin_big_brain_lev,
    coin_fixed_final_lev,
    coin_galaxy_brain_lev,
    coin_smart_lev,
    dice_big_brain_lev,
    dice_fixed_final_lev,
    dice_sh_fixed_final_lev,
    dice_sh_smart_lev,
    dice_smart_lev,
    gbm_fixed_final_lev,
    gbm_smart_lev,
)
from tests.input_tests import levarge_tests

# fmt: off

# HYPERPARAMETERS
INVESTORS = 1e4             # number of random investors
HORIZON = 3e2               # total time steps
TOP = INVESTORS * 1e-4      # define number of top performers
VALUE_0 = 1e2               # initial portfolio value of each investor
ASYM_LIM = 1e-12            # offset to enforce "optimal" leverage bound
VRAM = False                # use CUDA-based GPU

# COIN FLIP
COIN_UP_PROB = 0.5          # probability of up move
COIN_UP_R = 0.5             # upside return (>0)
COIN_DOWN_R = -0.4          # downside return (<0)

# DICE ROLL
DICE_UP_PROB = 1 / 6        # probability of up move
DICE_DOWN_PROB = 1 / 6      # probability of down move
DICE_UP_R = 0.5             # upside return (>0)
DICE_DOWN_R = -0.5          # downside return (<0)
DICE_MID_R = 0.05           # mid return (UP_R>MID_R>DOWN_R)

# DICE ROLL (SAFE HAVEN)
DICESH_UP_PROB = 1 / 6      # probability of up move
DICESH_DOWN_PROB = 1 / 6    # probability of down move
DICESH_UP_R = 0.5           # upside return (UP_R>0)
DICESH_DOWN_R = -0.5        # downside return (DOWN_R<0)
DICESH_MID_R = 0.05         # mid return (UP_R>MID_R>DOWN_R)
DICESH_SH_UP_R = -1         # safe haven upside return (<0)
DICESH_SH_DOWN_R = 5        # safe haven downside return (>0)
DICESH_SH_MID_R = -1        # safe haven mid return (SH_UP_R<=SH_MID_R<SH_DOWN_R)

# GEOMETRIC BROWNIAN MOTION
GBM_DRIFT = [0.05, 0.0540025395205692]
GBM_VOL = [np.sqrt(0.2), 0.1897916175617430]
GBM_NAME = ["gbm_op", "gbm_snp"]

# fmt: on

# x_l, x_h, x_i = starting value, ending value, increment
# investor 1 leverages (l)
l0_l, l0_h, l0_i = 0.50, 1.00, 0.10
l1_l, l1_h, l1_i = 0.50, 1.00, 0.10
# investor 2 stop-losses (s) and retention ratios (r)
s2_l, s2_h, s2_i = 0.10, 0.10, 0.10
r2_l, r2_h, r2_i = 0.00, 0.00, 0.10
# investor 3 stop-losses (s) and retention ratios (r)
s3_l, s3_h, s3_i = 0.70, 0.80, 0.10
r3_l, r3_h, r3_i = 0.70, 0.80, 0.10
# investor 4 up returns (ru), down returns (rd), and up probabilities (pu)
ru_l, ru_h, ru_i = 0.50, 0.80, 0.10
rd_l, rd_h, rd_i = 0.50, 0.80, 0.10
pu_l, pu_h, pu_i = 0.25, 0.75, 0.25

# GBM investor 1 leverages (l)
GBM_l0_l, GBM_l0_h, GBM_l0_i = [-1.0, 0.4], [1.0, 4.0], [0.2, 0.4]
GBM_l1_l, GBM_l1_h, GBM_l1_i = [-1.0, 0.2], [1.0, 2.0], [0.2, 0.2]

if __name__ == "__main__":

    path_results = "./results/multiverse-test/"  # directory for saving numpy arrays
    path_figs = "./results/figs-test/"  # directory to save figures

    levarge_tests(
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        ASYM_LIM,
        VRAM,
        path_results,
        path_figs,
        COIN_UP_PROB,
        COIN_UP_R,
        COIN_DOWN_R,
        DICE_UP_PROB,
        DICE_DOWN_PROB,
        DICE_UP_R,
        DICE_DOWN_R,
        DICE_MID_R,
        DICESH_UP_PROB,
        DICESH_DOWN_PROB,
        DICESH_UP_R,
        DICESH_DOWN_R,
        DICESH_MID_R,
        DICESH_SH_UP_R,
        DICESH_SH_DOWN_R,
        DICESH_SH_MID_R,
        l0_l,
        l0_h,
        l0_i,
        l1_l,
        l1_h,
        l1_i,
        s2_l,
        s2_h,
        s2_i,
        r2_l,
        r2_h,
        r2_i,
        s3_l,
        s3_h,
        s3_i,
        r3_l,
        r3_h,
        r3_i,
        ru_l,
        ru_h,
        ru_i,
        rd_l,
        rd_h,
        rd_i,
        pu_l,
        pu_h,
        pu_i,
        GBM_DRIFT,
        GBM_VOL,
        GBM_NAME,
        GBM_l0_l,
        GBM_l0_h,
        GBM_l0_i,
        GBM_l1_l,
        GBM_l1_h,
        GBM_l1_i,
    )

    start_time = time.perf_counter()

    # clean-up test files from earlier uncompleted tests
    if os.path.exists(path_results):
        shutil.rmtree(path_results)
    if os.path.exists(path_figs):
        shutil.rmtree(path_figs)

    os.makedirs(path_results)
    os.makedirs(path_figs)

    device = T.device("cuda:0") if VRAM == True else T.device("cpu")
    T.manual_seed(420)

    INVESTORS = T.tensor(int(INVESTORS), dtype=T.int32, device=device)
    HORIZON = T.tensor(int(HORIZON), dtype=T.int32, device=device)
    VALUE_0 = T.tensor(VALUE_0, device=device)
    TOP = int(TOP) if TOP > 1 else int(1)
    ASYM_LIM = T.tensor(ASYM_LIM, device=device)

    # COIN FLIP

    BIGGER_PAYOFF = (
        np.abs(COIN_DOWN_R)
        if np.abs(COIN_UP_R) >= np.abs(COIN_DOWN_R)
        else -np.abs(COIN_UP_R)
    )
    LEV_FACTOR = T.tensor(1 / BIGGER_PAYOFF, device=device)
    LEV_FACTOR = (
        LEV_FACTOR - ASYM_LIM
        if np.abs(COIN_UP_R) > np.abs(COIN_DOWN_R)
        else LEV_FACTOR + ASYM_LIM
    )

    probabilites = Bernoulli(COIN_UP_PROB)
    outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

    coin_fixed_final_lev(
        device,
        outcomes,
        TOP,
        VALUE_0,
        COIN_UP_R,
        COIN_DOWN_R,
        lev_low=l0_l,
        lev_high=l0_h,
        lev_incr=l0_i,
    )

    inv1_val_data, inv1_val_data_T = coin_smart_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        COIN_UP_R,
        COIN_DOWN_R,
        lev_low=l1_l,
        lev_high=l1_h,
        lev_incr=l1_i,
    )
    np.save(path_results + "coin_inv1_val.npy", inv1_val_data.cpu().numpy())
    np.save(path_results + "coin_inv1_val_T.npy", inv1_val_data_T.cpu().numpy())

    inv2_val_data = coin_big_brain_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        COIN_UP_R,
        COIN_DOWN_R,
        LEV_FACTOR,
        stop_min=s2_l,
        stop_max=s2_h,
        stop_incr=s2_i,
        roll_min=r2_l,
        roll_max=r2_h,
        roll_incr=r2_i,
    )
    np.save(path_results + "coin_inv2_val.npy", inv2_val_data.cpu().numpy())

    inv3_val_data = coin_big_brain_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        COIN_UP_R,
        COIN_DOWN_R,
        LEV_FACTOR,
        stop_min=s3_l,
        stop_max=s3_h,
        stop_incr=s3_i,
        roll_min=r3_l,
        roll_max=r3_h,
        roll_incr=r3_i,
    )
    np.save(path_results + "coin_inv3_val.npy", inv3_val_data.cpu().numpy())

    inv4_lev_data = coin_galaxy_brain_lev(
        device,
        ru_min=ru_l,
        ru_max=ru_h,
        ru_incr=ru_i,
        rd_min=rd_l,
        rd_max=rd_h,
        rd_incr=rd_i,
        pu_min=pu_l,
        pu_max=pu_h,
        pu_incr=pu_i,
    )
    np.save(path_results + "coin_inv4_lev.npy", inv4_lev_data.cpu().numpy())

    inv4_lev_data = np.load(path_results + "coin_inv4_lev.npy")
    plots.plot_inv4(inv4_lev_data, path_figs + "coin_inv4")

    inv3_val_data = np.load(path_results + "coin_inv3_val.npy")
    plots.plot_inv3(inv3_val_data, path_figs + "coin_inv3")

    inv2_val_data = np.load(path_results + "coin_inv2_val.npy")
    plots.plot_inv2(inv2_val_data, 30, path_figs + "coin_inv2")

    inv1_val_data = np.load(path_results + "coin_inv1_val.npy")
    inv1_val_data_T = np.load(path_results + "coin_inv1_val_T.npy")
    plots.plot_inv1(inv1_val_data, inv1_val_data_T, 1e30, path_figs + "coin_inv1")

    # DICE ROLL

    DICE_MID_PROB = 1 - (DICE_UP_PROB + DICE_DOWN_PROB)
    DICE_PROBS = T.tensor([DICE_UP_PROB, DICE_DOWN_PROB, DICE_MID_PROB], device=device)

    BIGGER_PAYOFF = (
        np.abs(DICE_DOWN_R)
        if np.abs(DICE_UP_R) >= np.abs(DICE_DOWN_R)
        else -np.abs(DICE_UP_R)
    )
    LEV_FACTOR = T.tensor(1 / BIGGER_PAYOFF, device=device)
    LEV_FACTOR = (
        LEV_FACTOR - ASYM_LIM
        if np.abs(DICE_UP_R) > np.abs(DICE_DOWN_R)
        else LEV_FACTOR + ASYM_LIM
    )

    probabilites = Categorical(DICE_PROBS)
    outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

    dice_fixed_final_lev(
        device,
        outcomes,
        TOP,
        VALUE_0,
        DICE_UP_R,
        DICE_DOWN_R,
        DICE_MID_R,
        lev_low=l0_l,
        lev_high=l0_h,
        lev_incr=l0_i,
    )

    inv1_val_data, inv1_val_data_T = dice_smart_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        DICE_UP_R,
        DICE_DOWN_R,
        DICE_MID_R,
        lev_low=l1_l,
        lev_high=l1_h,
        lev_incr=l1_i,
    )
    np.save(path_results + "dice_inv1_val.npy", inv1_val_data.cpu().numpy())
    np.save(path_results + "dice_inv1_val_T.npy", inv1_val_data_T.cpu().numpy())

    inv2_val_data = dice_big_brain_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        DICE_UP_R,
        DICE_DOWN_R,
        DICE_MID_R,
        LEV_FACTOR,
        stop_min=s2_l,
        stop_max=s2_h,
        stop_incr=s2_i,
        roll_min=r2_l,
        roll_max=r2_h,
        roll_incr=r2_i,
    )
    np.save(path_results + "dice_inv2_val.npy", inv2_val_data.cpu().numpy())

    inv3_val_data = dice_big_brain_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        DICE_UP_R,
        DICE_DOWN_R,
        DICE_MID_R,
        LEV_FACTOR,
        stop_min=s3_l,
        stop_max=s3_h,
        stop_incr=s3_i,
        roll_min=r3_l,
        roll_max=r3_h,
        roll_incr=r3_i,
    )
    np.save(path_results + "dice_inv3_val.npy", inv3_val_data.cpu().numpy())

    inv3_val_data = np.load(path_results + "dice_inv3_val.npy")
    plots.plot_inv3(inv3_val_data, path_figs + "dice_inv3")

    inv2_val_data = np.load(path_results + "dice_inv2_val.npy")
    plots.plot_inv2(inv2_val_data, 90, path_figs + "dice_inv2")

    inv1_val_data = np.load(path_results + "dice_inv1_val.npy")
    inv1_val_data_T = np.load(path_results + "dice_inv1_val_T.npy")
    plots.plot_inv1(inv1_val_data, inv1_val_data_T, 1e40, path_figs + "dice_inv1")

    # DICE ROLL (SAFE HAVEN)

    DICESH_MID_PROB = 1 - (DICESH_UP_PROB + DICESH_DOWN_PROB)
    DICESH_PROBS = T.tensor(
        [DICESH_UP_PROB, DICESH_DOWN_PROB, DICESH_MID_PROB], device=device
    )

    probabilites = Categorical(DICESH_PROBS)
    outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

    dice_sh_fixed_final_lev(
        device,
        outcomes,
        TOP,
        VALUE_0,
        DICESH_UP_R,
        DICESH_DOWN_R,
        DICESH_MID_R,
        DICESH_SH_UP_R,
        DICESH_SH_DOWN_R,
        DICESH_SH_MID_R,
        lev_low=l0_l,
        lev_high=l0_h,
        lev_incr=l0_i,
    )

    inv1_val_data, inv1_val_data_T = dice_sh_smart_lev(
        device,
        outcomes,
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        DICESH_UP_R,
        DICESH_DOWN_R,
        DICESH_MID_R,
        DICESH_SH_UP_R,
        DICESH_SH_DOWN_R,
        DICESH_SH_MID_R,
        lev_low=l1_l,
        lev_high=l1_h,
        lev_incr=l1_i,
    )
    np.save(path_results + "dice_sh_inv1_val.npy", inv1_val_data.cpu().numpy())
    np.save(path_results + "dice_sh_inv1_val_T.npy", inv1_val_data_T.cpu().numpy())

    inv1_val_data = np.load(path_results + "dice_sh_inv1_val.npy")
    inv1_val_data_T = np.load(path_results + "dice_sh_inv1_val_T.npy")
    plots.plot_inv1(inv1_val_data, inv1_val_data_T, 1e40, path_figs + "dice_sh_inv1")

    # GEOMETRIC BROWNIAN MOTION

    for x in range(0, len(GBM_DRIFT)):

        LOG_MEAN = T.tensor(GBM_DRIFT[x] - GBM_VOL[x] ** 2 / 2, device=device)
        VOL = T.tensor(GBM_VOL[x], device=device)

        probabilites = Normal(LOG_MEAN, VOL)
        outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

        gbm_fixed_final_lev(
            device,
            outcomes,
            TOP,
            VALUE_0,
            lev_low=GBM_l0_l[x],
            lev_high=GBM_l0_h[x],
            lev_incr=GBM_l0_i[x],
        )

        inv1_val_data, inv1_val_data_T = gbm_smart_lev(
            device,
            outcomes,
            INVESTORS,
            HORIZON,
            TOP,
            VALUE_0,
            lev_low=GBM_l1_l[x],
            lev_high=GBM_l1_h[x],
            lev_incr=GBM_l1_i[x],
        )
        np.save(
            path_results + GBM_NAME[x] + "_inv1_val.npy", inv1_val_data.cpu().numpy()
        )
        np.save(
            path_results + GBM_NAME[x] + "_inv1_val_T.npy",
            inv1_val_data_T.cpu().numpy(),
        )

        inv1_val_data = np.load(path_results + GBM_NAME[x] + "_inv1_val.npy")
        inv1_val_data_T = np.load(path_results + GBM_NAME[x] + "_inv1_val_T.npy")
        plots.plot_inv1(
            inv1_val_data, inv1_val_data_T, 1e40, path_figs + GBM_NAME[x] + "_inv1"
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(
        "TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
            total_time, total_time / 60, total_time / 3600
        )
    )

    # CLEAN UP TEST FILES

    shutil.rmtree(path_results)
    shutil.rmtree(path_figs)

    print(
        "--------------------------------------------------------------------------------"
    )

    print("All Leverage Tests: Passed")
