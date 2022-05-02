"""
title:                  dice_roll.py
usage:                  python opt_lev/dice_roll.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Execute optimal leverage experiments for the trinary dice roll gamble based on
    https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797.

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
from torch.distributions.categorical import Categorical

NDArrayFloat = npt.NDArray[np.float_]

import tools.plots_multiverse as plots
from opt_lev.lev_exp import dice_big_brain_lev, dice_fixed_final_lev, dice_smart_lev
from tools.input_tests import dice_roll_tests

# fmt: off

INVESTORS = 1e6             # number of random investors
HORIZON = 5e3               # total time steps
TOP = INVESTORS * 1e-4      # define number of top performers
VALUE_0 = 1e2               # initial portfolio value of each investor
UP_PROB = 1 / 6             # probability of up move
DOWN_PROB = 1 / 6           # probability of down move
UP_R = 0.5                  # upside return (>0)
DOWN_R = -0.5               # downside return (<0)
MID_R = 0.05                # mid return (UP_R>MID_R>DOWN_R)
ASYM_LIM = 1e-12            # offset to enforce "optimal" leverage bound

# fmt: on

# do you have >= 108GB of free VRAM for 1e6 INVESTORS?
# if False, still need 108GB of RAM or reduce number of INVESTORS
VRAM = False

# x_l, x_h, x_i = starting value, ending value, increment
# investor 1 leverages (l)
l0_l, l0_h, l0_i = 0.05, 1.00, 0.05
l1_l, l1_h, l1_i = 0.10, 1.00, 0.10
# investor 2 stop-losses (s) and retention ratios (r)
s2_l, s2_h, s2_i = 0.10, 0.10, 0.10
r2_l, r2_h, r2_i = 0.00, 0.00, 0.10
# investor 3 stop-losses (s) and retention ratios (r)
s3_l, s3_h, s3_i = 0.05, 0.95, 0.05
r3_l, r3_h, r3_i = 0.45, 0.95, 0.05

if __name__ == "__main__":

    path_results = "./results/multiverse/"  # directory for saving numpy arrays
    path_figs = "./results/figs/"  # directory to save figures

    # conduct tests
    dice_roll_tests(
        INVESTORS,
        HORIZON,
        TOP,
        VALUE_0,
        UP_PROB,
        DOWN_PROB,
        UP_R,
        DOWN_R,
        MID_R,
        ASYM_LIM,
        VRAM,
        path_results,
        path_figs,
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
    MID_PROB = 1 - (UP_PROB + DOWN_PROB)
    PROBS = T.tensor([UP_PROB, DOWN_PROB, MID_PROB], device=device)
    ASYM_LIM = T.tensor(ASYM_LIM, device=device)

    # theoretical optimal leverage based on "expectations"
    BIGGER_PAYOFF = np.abs(DOWN_R) if np.abs(UP_R) >= np.abs(DOWN_R) else -np.abs(UP_R)
    LEV_FACTOR = T.tensor(1 / BIGGER_PAYOFF, device=device)
    LEV_FACTOR = (
        LEV_FACTOR - ASYM_LIM
        if np.abs(UP_R) > np.abs(DOWN_R)
        else LEV_FACTOR + ASYM_LIM
    )

    # RUN EXPERIMENTS

    start_time = time.perf_counter()

    T.manual_seed(420)  # set fixed seed for reproducibility

    probabilites = Categorical(PROBS)
    outcomes = probabilites.sample(sample_shape=(INVESTORS, HORIZON)).to(device)

    dice_fixed_final_lev(
        device,
        outcomes,
        TOP,
        VALUE_0,
        UP_R,
        DOWN_R,
        MID_R,
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
        UP_R,
        DOWN_R,
        MID_R,
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
        UP_R,
        DOWN_R,
        MID_R,
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
        UP_R,
        DOWN_R,
        MID_R,
        LEV_FACTOR,
        stop_min=s3_l,
        stop_max=s3_h,
        stop_incr=s3_i,
        roll_min=r3_l,
        roll_max=r3_h,
        roll_incr=r3_i,
    )
    np.save(path_results + "dice_inv3_val.npy", inv3_val_data.cpu().numpy())

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(
        "TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
            total_time, total_time / 60, total_time / 3600
        )
    )

    # LOAD EXPERIMENT DATA AND SAVE FIGURES

    inv3_val_data = np.load(path_results + "dice_inv3_val.npy")
    plots.plot_inv3(inv3_val_data, path_figs + "dice_inv3")

    inv2_val_data = np.load(path_results + "dice_inv2_val.npy")
    plots.plot_inv2(inv2_val_data, 90, path_figs + "dice_inv2")

    inv1_val_data = np.load(path_results + "dice_inv1_val.npy")
    inv1_val_data_T = np.load(path_results + "dice_inv1_val_T.npy")
    plots.plot_inv1(inv1_val_data, inv1_val_data_T, 1e40, path_figs + "dice_inv1")
