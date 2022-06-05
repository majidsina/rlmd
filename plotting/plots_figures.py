"""
title:                  plot_figures.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Plotting of all final summary figures for all reinforcement learning experiments.
"""

import sys

sys.path.append("./")

from os import PathLike
from typing import List, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as T
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

NDArrayFloat = npt.NDArray[np.float_]

import tools.critic_loss as closs
import tools.utils as utils


def loss_fn_plot(filename: Union[str, bytes, PathLike]) -> None:
    """
    Plot of critic loss functions about the origin.

    Parameters:
        filename: save path of plot
    """
    pdf = T.distributions.normal.Normal(0, 10)
    q = int(1e6)

    a = pdf.sample((q,))
    b = pdf.sample((q,))
    c = b - a

    mse = closs.mse(a, b, 0)
    mse2 = closs.mse(a, b, 2)
    mse4 = closs.mse(a, b, 4)
    huber = closs.huber(a, b)
    mae = closs.mae(a, b)
    hsc = closs.hypersurface(a, b)
    cauchy = closs.cauchy(a, b, 1)

    size = 3
    cols = ["C" + str(x) for x in range(7)]
    l = ["MSE", "MSE2", "MSE4", "Huber", "MAE", "HSC", "Cauchy"]

    plt.scatter(c, mse, s=size, c=cols[0], rasterized=True)
    plt.scatter(c, mse2, s=size, c=cols[1], rasterized=True)
    plt.scatter(c, mse4, s=size, c=cols[2], rasterized=True)
    plt.scatter(c, huber, s=size, c=cols[3], rasterized=True)
    plt.scatter(c, mae, s=size, c=cols[4], rasterized=True)
    plt.scatter(c, hsc, s=size, c=cols[5], rasterized=True)
    plt.scatter(c, cauchy, s=size, c=cols[6], rasterized=True)

    plt.xlim((-6, 6))
    plt.ylim((0, 5))
    plt.tick_params(axis="both", which="major", labelsize="small")
    plt.title("Loss", size="large")
    plt.legend(
        l, loc="lower right", ncol=1, frameon=False, fontsize="medium", markerscale=6
    )
    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", dpi=400, format="svg")


def plot_smoothing_fn(filename: Union[str, bytes, PathLike]) -> None:
    """
    Plot action smoothing function for multiplicative and market environments.

    Parameters:
        filename: save path of plot
    """
    ratio = np.linspace(0, 1, 10000)
    f = utils.smoothing_func(ratio)

    plt.plot(ratio, f)
    plt.xlabel("Ratio " + r"$t / T_s$")
    plt.ylabel("Action Smoothing " + r"$s_t$")
    plt.grid(True, linewidth=0.5)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_gbm_max_lev(filename: Union[str, bytes, PathLike]) -> None:
    """
    Plot of GBM maximum leverage across down sigma moves.

    Parameters:
        filename: save path of plot
    """
    # S&P500 parameters
    mu = 0.0540025395205692
    sigma = 0.1897916175617430
    l_opt = mu / sigma**2

    v0 = [2, 3, 5, 10]
    vmin = 1

    sig = np.linspace(1, 10, 1000)
    prob = norm.pdf(sig, 0, 1)
    prob = np.log10(prob)

    rho = np.linspace(1, 10, 1000)

    l_ratio = np.log(v0) - np.log(vmin)
    l_factor = 2 / (sigma * (2 * rho + sigma) - 2 * mu)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, label="lev")
    ax2 = fig.add_subplot(1, 1, 1, label="prob", frame_on=False)

    l_eff = []
    for x in range(len(v0)):
        l_eff.append(l_factor * l_ratio[x])
        ax1.plot(rho, l_eff[x], color="C" + str(x + 1))

    ax1.set_xlabel(r"$\rho_d$")
    ax1.yaxis.tick_left()
    ax1.set_ylabel("Maximum Leverage")
    ax1.yaxis.set_label_position("left")
    ax1.tick_params(axis="y")
    ax1.grid(True, linewidth=0.5)

    ax2.plot(sig, prob, color="C0")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Probability (log" + r"$_{10}$" + ")", color="C0")
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(axis="y", colors="C0")

    ax1.hlines(l_opt, 1, 10, colors="black", linestyle="--")

    fig.subplots_adjust(hspace=0.3)
    fig.legend(
        v0,
        loc="upper center",
        ncol=len(v0),
        frameon=False,
        fontsize="medium",
        title=r"$V_{t-1}/V_{\min}$",
        title_fontsize="medium",
    )

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_shuffled_histories(filename: Union[str, bytes, PathLike]) -> None:
    """
    Plot of unique histories as a function of shuffled days.

    Parameters:
        filename: save path of plot
    """
    days, train, eval = [], [], []

    n = 9000
    mt = 1000
    me = 250
    g = 20

    for x in range(1, 16, 1):
        dt, de = x, x
        ut, ue = utils.unique_histories(n, mt, me, g, dt, de)
        days.append(x)
        train.append(ut)
        eval.append(ue)

    plt.plot(days, train, color="C0")
    plt.plot(days, eval, color="C1")
    plt.xlabel("Shuffled Days " + r"$d_k$")
    plt.ylabel("Unique Histories " + r"$u_k$" + " (log" + r"$_{10}$" + ")")
    plt.grid(True, linewidth=0.5)

    l = ["Training", "Evaluation"]
    plt.legend(
        l,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize="large",
        markerscale=6,
        bbox_to_anchor=(0.5, 1.15),
    )

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def market_prices(
    data_path,
    file_path: Union[str, bytes, PathLike],
) -> None:
    """
    Plot market prices of several assets.

    Parameters:
        path_data: location of market data
        filename: save path of plot
    """
    usei = pd.read_pickle(data_path + "stooq_usei.pkl")
    major = pd.read_pickle(data_path + "stooq_major.pkl")
    dji = pd.read_pickle(data_path + "stooq_dji.pkl")

    usei, major, dji = (
        usei["Close"].dropna(),
        major["Close"].dropna(),
        dji["Close"].dropna(),
    )

    usei = usei[::-1] if usei.index[0] > usei.index[-1] else usei
    major = major[::-1] if major.index[0] > major.index[-1] else major
    dji = dji[::-1] if dji.index[0] > dji.index[-1] else dji

    usei_x = np.linspace(1, len(usei["^SPX"]), 10)
    major_x = np.linspace(1, len(major["^SPX"]), 10)
    dji_x = np.linspace(1, len(dji["^SPX"]), 10)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for x in ["^SPX", "^NDX", "^DJI"]:
        ax1.plot(major[x], linewidth=0.75, rasterized=True)

    ax1.margins(x=0)

    ax1.set_ylabel("Value")
    ax1.set_xlabel("Year")

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ax2.set_xticks(usei_x)
    ax2.set_xlabel("Count " + r"$N$" + " (Days)")

    l = ["SPX", "NDX", "DJI"]
    leg = ax1.legend(
        l, loc="upper left", ncol=1, frameon=False, fontsize="medium", markerscale=3
    )

    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=3)

    filename = file_path + "mar_" + "usei"
    plt.savefig(
        filename + ".png", dpi=400, format="png", pad_inches=0.2, bbox_inches="tight"
    )
    plt.savefig(
        filename + ".svg", dpi=300, format="svg", pad_inches=0.2, bbox_inches="tight"
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for x in [
        "AAPL.US",
        "AMGN.US",
        "AXP.US",
        "BA.US",
        "CAT.US",
        "CVX.US",
        "DIS.US",
        "HD.US",
        "IBM.US",
        "INTC.US",
        "JNJ.US",
        "JPM.US",
        "KO.US",
        "MCD.US",
        "MMM.US",
        "MRK.US",
        "MSFT.US",
        "NKE.US",
        "PFE.US",
        "PG.US",
        "VZ.US",
        "WBA.US",
        "WMT.US",
        "CSCO.US",
        "UNH.US",
    ]:
        ax1.plot(dji[x], linewidth=0.75, rasterized=True)

    ax1.margins(x=0)

    ax1.set_ylabel("Value")
    ax1.set_xlabel("Year")

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ax2.set_xticks(dji_x)
    ax2.set_xlabel("Count " + r"$N$" + " (Days)")

    l = [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CVX",
        "DIS",
        "HD",
        "IBM",
        "INTC",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "PFE",
        "PG",
        "VZ",
        "WBA",
        "WMT",
        "CSCO",
        "UNH",
    ]
    leg = ax1.legend(l, loc="upper left", ncol=2, frameon=False, fontsize="medium")

    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=3)

    filename = file_path + "mar_" + "dji"
    plt.savefig(
        filename + ".png", dpi=200, format="png", pad_inches=0.2, bbox_inches="tight"
    )
    plt.savefig(
        filename + ".svg", dpi=300, format="svg", pad_inches=0.2, bbox_inches="tight"
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    for x in [
        "GC.F",
        "SI.F",
        "HG.F",
        "PL.F",
        "CL.F",
        "LS.F",
        "PA.F",
        "RB.F",
        "LE.F",
        "KC.F",
        "OJ.F",
    ]:
        ax1.plot(major[x], linewidth=0.75, rasterized=True)

    ax1.margins(x=0)

    ax1.set_ylabel("Value")
    ax1.set_xlabel("Year")

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ax2.set_xticks(major_x)
    ax2.set_xlabel("Count " + r"$N$" + " (Days)")

    l = [
        "Gold",
        "Silver",
        "Copper",
        "Platinum",
        "WTI",
        "Lumber",
        "Palladium",
        "RBOB",
        "Cattle",
        "Coffee",
        "OJ",
    ]
    leg = ax1.legend(l, loc="upper left", ncol=1, frameon=False, fontsize="medium")

    leg_lines = leg.get_lines()
    plt.setp(leg_lines, linewidth=3)

    filename = file_path + "mar_" + "major"
    plt.savefig(
        filename + ".png", dpi=200, format="png", pad_inches=0.2, bbox_inches="tight"
    )
    plt.savefig(
        filename + ".svg", dpi=300, format="svg", pad_inches=0.2, bbox_inches="tight"
    )


def plot_add(
    inputs: dict,
    env_names: List[str],
    legend: List[str],
    multi: bool,
    reward: NDArrayFloat,
    loss: NDArrayFloat,
    scale: NDArrayFloat,
    kernel: NDArrayFloat,
    tail: NDArrayFloat,
    shadow: NDArrayFloat,
    keqv: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
) -> None:
    """
    Plots additve environments figures for loss functions and multi-step returns.

    Parameters:
        inputs: dictionary containing all execution details
        env_names: list of environment names
        legend: list of labeling across trials
        multi: True or False as to whether plotting multi-stewp returns
        rewards: rewards across trials
        loss: critic losses across trials
        scale: Cauchy scales across trials
        kernel: CIM kernel sizes across trials
        tail: critic tail exponents across trials
        shadow: critic shahow means across trials
        keqv: multiplier for equvilance between shadow and empirical means across trials
        filename: path for file saving
    """
    n_env, n_algo, n_data = reward.shape[0], reward.shape[1], reward.shape[2]

    if multi == True:
        legend = ["m = " + str(legend[x]) for x in range(n_data)]

    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(n_data)]

    patches = [
        mpatches.Patch(color=cols[x], label=legend[x], alpha=0.8) for x in range(n_data)
    ]

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = reward[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC Score")
    axs[1, 0].set_ylabel("TD3 Score")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_score" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_score" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = loss[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = (
                    np.log10(x_mean),
                    np.log10(x_mad_up),
                    np.log10(x_mad_lo),
                )

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC Critic (log" + r"$_{10}$" + ")")
    axs[1, 0].set_ylabel("TD3 Critic (log" + r"$_{10}$" + ")")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_loss" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_loss" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = scale[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = (
                    np.log10(x_mean),
                    np.log10(x_mad_up),
                    np.log10(x_mad_lo),
                )

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC Cauchy Scale " + r"$\gamma$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_ylabel("TD3 Cauchy Scale " + r"$\gamma$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_scale" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_scale" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = kernel[e, a, d]

                # use np.nan conversion to ignore divergences
                x_mean = np.nanmean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.nanmean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = (
                    np.log10(x_mean),
                    np.log10(x_mad_up),
                    np.log10(x_mad_lo),
                )

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC CIM Kernel " + r"$\sigma$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_ylabel("TD3 CIM Kernel " + r"$\sigma$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_kernel" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_kernel" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = tail[e, a, d]

                # use np.nan conversion to ignore divergences
                x_mean = np.nanmean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.nanmean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC Critic Tail " + r"$\alpha$")
    axs[1, 0].set_ylabel("TD3 Critic Tail " + r"$\alpha$")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_tail" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_tail" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = shadow[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = (
                    np.log10(x_mean),
                    np.log10(x_mad_up),
                    np.log10(x_mad_lo),
                )

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("SAC Critic Shadow " + r"$\mu_s$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_ylabel("TD3 Critic Shadow " + r"$\mu_s$" + " (log" + r"$_{10}$" + ")")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_shadow" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_shadow" + ".svg", dpi=125, format="svg")

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

    for a in range(n_algo):
        for e in range(n_env):
            for d in range(n_data):

                var_x = keqv[e, a, d]

                x_mean = np.mean(var_x, axis=1, keepdims=True)

                x_max = np.max(var_x, axis=1, keepdims=True)
                x_min = np.min(var_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)

                x_mean, x_mad_up, x_mad_lo = (
                    np.log10(x_mean),
                    np.log10(x_mad_up),
                    np.log10(x_mad_lo),
                )

                axs[a, e].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[a, e].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[a, e].grid(True, linewidth=0.2)

                if a != 1 or e != 0:
                    axs[a, e].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel(
        "SAC Multiplier " + r"$\kappa_{eqv}$" + " (log" + r"$_{10}$" + ")"
    )
    axs[1, 0].set_ylabel(
        "TD3 Multiplier " + r"$\kappa_{eqv}$" + " (log" + r"$_{10}$" + ")"
    )
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.325, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes
    )
    axs[0, 1].text(
        0.325, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes
    )
    axs[0, 2].text(
        0.325, 1.1, env_names[2], size="large", transform=axs[0, 2].transAxes
    )
    axs[0, 3].text(
        0.325, 1.1, env_names[3], size="large", transform=axs[0, 3].transAxes
    )

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.175)
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=n_data,
        frameon=False,
        fontsize="large",
    )

    plt.savefig(filename + "_keqv" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_keqv" + ".svg", dpi=125, format="svg")


def plot_add_temp(
    inputs: dict,
    env_names: List[str],
    legend: List[str],
    multi: bool,
    logtemp: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
) -> None:
    """
    Plots additve environments SAC entropy temperature for loss functions and multi-step returns.

    Parameters:
        inputs: dictionary containing all execution details
        env_names: list of environment names
        legend: list of labeling across trials
        multi: True or False as to whether plotting multi-stewp returns
        logtemp: log SAC entopy temperature across trials
        filename: path for file saving
    """
    n_env, n_algo, n_data = logtemp.shape[0], logtemp.shape[1], logtemp.shape[2]

    if multi == True:
        legend = ["m = " + str(legend[x]) for x in range(n_data)]

    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(n_data)]

    patches = [
        mpatches.Patch(color=cols[x], label=legend[x], alpha=0.8) for x in range(n_data)
    ]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    for e in range(n_env):
        for d in range(n_data):

            var_x = logtemp[e, 0, d]

            var_x = np.exp(var_x)

            x_mean = np.mean(var_x, axis=1, keepdims=True)

            x_max, x_min = np.max(var_x, axis=1, keepdims=True), np.min(
                var_x, axis=1, keepdims=True
            )
            x_mad = np.mean(np.abs(var_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)

            x_mean, x_mad_up, x_mad_lo = (
                np.log10(x_mean),
                np.log10(x_mad_up),
                np.log10(x_mad_lo),
            )

            if e == 0:
                axs[0, 0].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[0, 0].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[0, 0].grid(True, linewidth=0.2)
                axs[0, 0].xaxis.set_ticklabels([])

            elif e == 1:
                axs[0, 1].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[0, 1].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[0, 1].grid(True, linewidth=0.2)
                axs[0, 1].xaxis.set_ticklabels([])

            elif e == 2:
                axs[1, 0].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[1, 0].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[1, 0].grid(True, linewidth=0.2)

            else:
                axs[1, 1].plot(
                    x_steps, x_mean, color=cols[d], linewidth=0.5, rasterized=True
                )
                axs[1, 1].fill_between(
                    x_steps,
                    x_mad_lo,
                    x_mad_up,
                    facecolor=cols[d],
                    alpha=0.15,
                    rasterized=True,
                )
                axs[1, 1].grid(True, linewidth=0.2)
                axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Entropy Temperature (log" + r"$_{10}$" + ")")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(0.35, 1.1, env_names[0], size="large", transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.35, 1.1, env_names[1], size="large", transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.35, 1.1, env_names[2], size="large", transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.35, 1.1, env_names[3], size="large", transform=axs[1, 1].transAxes)

    fig.tight_layout(h_pad=2)
    fig.subplots_adjust(bottom=0.15)

    if multi == False:
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=int(n_data / 2 + 1),
            frameon=False,
            fontsize="large",
        )
    else:
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=n_data,
            frameon=False,
            fontsize="large",
        )

    plt.savefig(filename + "_temp" + ".png", dpi=200, format="png")
    plt.savefig(filename + "_temp" + ".svg", dpi=125, format="svg")


def plot_inv(
    inputs: dict,
    reward: NDArrayFloat,
    lev: NDArrayFloat,
    stop: NDArrayFloat,
    reten: NDArrayFloat,
    loss: NDArrayFloat,
    tail: NDArrayFloat,
    shadow: NDArrayFloat,
    cmax: NDArrayFloat,
    keqv: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investors for a constant number of assets.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        filename: save path of plot
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = reward.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label="Inv A", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Inv B", alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label="Inv C", alpha=0.8)

    reward = V_0 * reward**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(ninv):

        inv_x = reward[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100

        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for i in range(ninv):

        inv_x = lev[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[1, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[1, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[1, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[1, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[1, 0].grid(True, linewidth=0.2)
        axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Leverage")

    for i in range(1, ninv):

        inv_x = stop[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

        x_mean = x_mean.reshape(-1) * 100
        x_med = x_med.reshape(-1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1) * 100
        x_u = x_u.reshape(-1) * 100

        axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[2, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[2, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[2, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[2, 0].grid(True, linewidth=0.2)
        axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Stop-Loss " + r"$\lambda$ " + "(%)")

    for i in range(2, ninv):

        inv_x = reten[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

        x_mean = x_mean.reshape(-1) * 100
        x_med = x_med.reshape(-1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1) * 100
        x_u = x_u.reshape(-1) * 100

        axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[3, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[3, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[3, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 0].grid(True, linewidth=0.2)

    axs[3, 0].set_ylabel("Retention " + r"$\phi$ " + "(%)")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    for i in range(ninv):

        inv_x = loss[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_ylabel("Critic")

    for i in range(ninv):

        inv_x = tail[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 1].set_ylabel("Critic Tail " + r"$\alpha$")

    for i in range(ninv):

        inv_x = shadow[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max = np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for i in range(ninv):

        inv_x = keqv[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max, x_min = np.max(inv_x, axis=1, keepdims=True), np.min(
            inv_x, axis=1, keepdims=True
        )
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    axs[3, 1].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_inv_all_n_perf(
    inputs: dict,
    reward_1: NDArrayFloat,
    lev_1: NDArrayFloat,
    stop_1: NDArrayFloat,
    reten_1: NDArrayFloat,
    reward_2: NDArrayFloat,
    lev_2: NDArrayFloat,
    stop_2: NDArrayFloat,
    reten_2: NDArrayFloat,
    reward_10: NDArrayFloat,
    lev_10: NDArrayFloat,
    stop_10: NDArrayFloat,
    reten_10: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    n_gambles: List[int],
    g_min: List[float] = [None, None, None],
    g_max: List[float] = [None, None, None],
    l_min: List[float] = [None, None, None],
    l_max: List[float] = [None, None, None],
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investor performance across three counts of assets.

    Parameters:
        inputs: dictionary containing all execution details
        reward_1: 1 + time-average growth rate for n_1 assets
        lev_1: leverages for n_1 assets
        stop_1: stop-losses for n_1 assets
        reten_1: retention ratios for n_1 assets
            ...
        filename: save path of plot
        n_gambles: number of gambles
        g_min, g_max, l_min, l_max: graph bounds
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = reward_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label="Inv A", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Inv B", alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label="Inv C", alpha=0.8)

    reward_1, reward_2, reward_10 = (
        V_0 * reward_1**T,
        V_0 * reward_2**T,
        V_0 * reward_10**T,
    )

    fig, axs = plt.subplots(nrows=4, ncols=ninv, figsize=(10, 12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = reward_1[i]
            elif n == 1:
                inv_x = reward_2[i]
            else:
                inv_x = reward_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
            x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

            x_mean = (x_mean.reshape(-1) - 1) * 100
            x_med = (x_med.reshape(-1) - 1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = (x_d.reshape(-1) - 1) * 100
            x_u = (x_u.reshape(-1) - 1) * 100

            # x_mean = np.log10(x_mean)
            # x_med = np.log10(x_med)
            # x_mad_lo = np.log10(x_mad_lo)
            # x_mad_up = np.log10(x_mad_up)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[0, n].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[0, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[0, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )

            axs[0, n].set_ylim(g_min[n], g_max[n])

            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = lev_1[i]
            elif n == 1:
                inv_x = lev_2[i]
            else:
                inv_x = lev_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[1, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[1, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )

            axs[1, n].set_ylim(l_min[n], l_max[n])

            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Leverage")

    for n in range(3):
        for i in range(1, ninv, 1):

            if n == 0:
                inv_x = stop_1[i]
            elif n == 1:
                inv_x = stop_2[i]
            else:
                inv_x = stop_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[2, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[2, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Stop-Loss " + r"$\lambda$ " + "(%)")

    for n in range(3):
        for i in range(2, ninv, 1):

            if n == 0:
                inv_x = reten_1[i]
            elif n == 1:
                inv_x = reten_2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = reten_10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[3, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[3, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel("Retention " + r"$\phi$ " + "(%)")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[0]),
        size="large",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 1].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[1]),
        size="large",
        transform=axs[0, 1].transAxes,
    )
    axs[0, 2].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[2]),
        size="large",
        transform=axs[0, 2].transAxes,
    )

    fig.subplots_adjust(bottom=0.075, wspace=0.25, hspace=0.3)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_inv_all_n_train(
    inputs: dict,
    loss_1: NDArrayFloat,
    tail_1: NDArrayFloat,
    shadow_1: NDArrayFloat,
    keqv1: NDArrayFloat,
    loss_2: NDArrayFloat,
    tail_2: NDArrayFloat,
    shadow_2: NDArrayFloat,
    keqv2: NDArrayFloat,
    loss_10: NDArrayFloat,
    tail_10: NDArrayFloat,
    shadow_10: NDArrayFloat,
    keqv10: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    n_gambles: List[int],
) -> None:
    """
    Plot summary of investor training across three counts of assets.

    Parameters:
        inputs: dictionary containing all execution details
        loss_1: mean critic loss for n_1 assets
        tail_1: tail exponent for n_1 assets
        shadow_1: critic shadow loss for n_1 assets
        keqv_1: equivilance multiplier for n_1 assets
            ...
        filename: save path of plot
        n_gambles: number of gambles
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = loss_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(ninv)]
    a_col = mpatches.Patch(color=cols[0], label="Inv A", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Inv B", alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label="Inv C", alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=ninv, figsize=(10, 12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = loss_1[i]
            elif n == 1:
                inv_x = loss_2[i]
            else:
                inv_x = loss_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[0, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[0, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Critic")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = tail_1[i]
            elif n == 1:
                inv_x = tail_2[i]
            else:
                inv_x = tail_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[1, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[1, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Critic Tail " + r"$\alpha$")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = shadow_1[i]
            elif n == 1:
                inv_x = shadow_2[i]
            else:
                inv_x = shadow_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[2, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[2, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = keqv1[i]
            elif n == 1:
                inv_x = keqv2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = keqv10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            # axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[3, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            # axs[3, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[0]),
        size="large",
        transform=axs[0, 0].transAxes,
    )
    axs[0, 1].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[1]),
        size="large",
        transform=axs[0, 1].transAxes,
    )
    axs[0, 2].text(
        0.35,
        1.2,
        r"$N = $" + str(n_gambles[2]),
        size="large",
        transform=axs[0, 2].transAxes,
    )

    fig.subplots_adjust(bottom=0.075, wspace=0.25, hspace=0.3)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_safe_haven(
    inputs: dict,
    reward: NDArrayFloat,
    lev: NDArrayFloat,
    stop: NDArrayFloat,
    reten: NDArrayFloat,
    loss: NDArrayFloat,
    tail: NDArrayFloat,
    shadow: NDArrayFloat,
    cmax: NDArrayFloat,
    keqv: NDArrayFloat,
    lev_sh: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    g_min: float = None,
    g_max: float = None,
    l_min: float = None,
    l_max: float = None,
    inv: str = "a",
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investors for safe haven.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        filename: save path of plot
        g_min, g_max, l_min, l_max: graph bounds
        inv: whether "a", "b" or "c"
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = reward.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    if inv == "a":
        inv_col = "C0"
    elif inv == "b":
        inv_col = "C1"
    else:
        inv_col = "C2"

    cols = [inv_col, "C4"]
    a_col = mpatches.Patch(color=cols[0], label="Uninsured", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Insured", alpha=0.8)

    reward = V_0 * reward**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(ninv):

        inv_x = reward[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100

        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[0, 0].set_ylim(g_min, g_max)

        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for i in range(ninv):

        inv_x = lev[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[1, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[1, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[1, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[1, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[1, 0].set_ylim(l_min, l_max)

        axs[1, 0].grid(True, linewidth=0.2)

    axs[1, 0].set_ylabel("Leverage")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    if inv != "a":
        for i in range(ninv):

            inv_x = stop[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[2, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[2, 0].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[2, 0].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[2, 0].grid(True, linewidth=0.2)

        axs[2, 0].set_ylabel("Stop-Loss " + r"$\lambda$ " + "(%)")

    if inv == "c":
        for i in range(ninv):

            inv_x = reten[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[3, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[3, 0].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[3, i].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[3, 0].grid(True, linewidth=0.2)

        axs[3, 0].set_ylabel("Retention " + r"$\phi$ " + "(%)")

    for i in range(ninv):

        inv_x = loss[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_ylabel("Critic")

    for i in range(ninv):

        inv_x = tail[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 1].set_ylabel("Critic Tail " + r"$\alpha$")

    for i in range(ninv):

        inv_x = shadow[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max =  np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean+x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean-x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for i in range(ninv):

        inv_x = keqv[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 1].fill_between(x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1)
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    axs[3, 1].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")

    if inv == "a":
        axs[2, 0].set_axis_off()
        axs[2, 0].xaxis.set_ticklabels([])
        axs[3, 0].set_axis_off()
        axs[3, 0].xaxis.set_ticklabels([])

    if inv == "b":
        axs[3, 0].set_axis_off()
        axs[3, 0].xaxis.set_ticklabels([])

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_sh_perf(
    inputs: dict,
    reward_1: NDArrayFloat,
    lev_1: NDArrayFloat,
    reward_2: NDArrayFloat,
    lev_2: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    g_min: List[float],
    g_max: List[float],
    l_min: List[float],
    l_max: List[float],
    inv: str = "a",
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of safe haven investor performance across both algorithms.

    Parameters:
        inputs: dictionary containing all execution details
        reward_a: 1 + time-average growth rate for invA with and without safe haven
        lev_a: leverages for invA with and without safe haven

            ...
        filename: save path of plot
        g_min, g_max, l_min, l_max: graph bounds
        inv: whether "a", "b" or "c"
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = reward_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    if inv == "a":
        inv_col = "C0"
    elif inv == "b":
        inv_col = "C1"
    else:
        inv_col = "C2"

    cols = [inv_col, "C4"]
    a_col = mpatches.Patch(color=cols[0], label="Uninsured", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Insured", alpha=0.8)

    reward_1, reward_2 = V_0 * reward_1**T, V_0 * reward_2**T

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    for i in range(ninv):

        inv_x = reward_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100

        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[0, 0].set_ylim(g_min[0], g_max[0])

        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")
    axs[0, 0].set_title("SAC", fontsize=14)

    for i in range(ninv):

        inv_x = lev_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[1, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[1, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[1, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[1, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[1, 0].set_ylim(l_min[0], l_max[0])

        axs[1, 0].grid(True, linewidth=0.2)

    axs[1, 0].set_ylabel("Leverage")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    for i in range(ninv):

        inv_x = reward_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100

        # x_mean = np.log10(x_mean)
        # x_med = np.log10(x_med)
        # x_mad_lo = np.log10(x_mad_lo)
        # x_mad_up = np.log10(x_mad_up)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 1].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 1].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 1].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[0, 1].set_ylim(g_min[1], g_max[1])

        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_title("TD3", fontsize=14)

    for i in range(ninv):

        inv_x = lev_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[1, 1].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[1, 1].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[1, 1].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[1, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )

        axs[1, 1].set_ylim(l_min[1], l_max[1])

        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_sh_train(
    inputs: dict,
    loss_1: NDArrayFloat,
    tail_1: NDArrayFloat,
    shadow_1: NDArrayFloat,
    cmax_1: NDArrayFloat,
    keqv_1: NDArrayFloat,
    loss_2: NDArrayFloat,
    tail_2: NDArrayFloat,
    shadow_2: NDArrayFloat,
    cmax_2: NDArrayFloat,
    keqv_2: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    inv: str = "a",
) -> None:
    """
    Plot summary of safe haven investor training across both algorithms.

    Parameters:
        inputs: dictionary containing all execution details
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means

            ...
        filename: save path of plot
        inv: whether "a", "b" or "c"
    """
    ninv = loss_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    if inv == "a":
        inv_col = "C0"
    elif inv == "b":
        inv_col = "C1"
    else:
        inv_col = "C2"

    cols = [inv_col, "C4"]
    a_col = mpatches.Patch(color=cols[0], label="Uninsured", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Insured", alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(ninv):

        inv_x = loss_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Critic")
    axs[0, 0].set_title("SAC", fontsize=14)

    for i in range(ninv):

        inv_x = tail_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 0].grid(True, linewidth=0.2)
        axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Critic Tail " + r"$\alpha$")

    for i in range(ninv):

        inv_x = shadow_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 0].grid(True, linewidth=0.2)
        axs[2, 0].xaxis.set_ticklabels([])

        inv_x = cmax_1[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max = np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 0].grid(True, linewidth=0.2)
        # axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for i in range(ninv):

        inv_x = keqv_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 0].grid(True, linewidth=0.2)
        # axs[3, 0].xaxis.set_ticklabels([])

    axs[3, 0].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    for i in range(ninv):

        inv_x = loss_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_title("TD3", fontsize=14)

    for i in range(ninv):

        inv_x = tail_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    for i in range(ninv):

        inv_x = shadow_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax_2[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max = np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    for i in range(ninv):

        inv_x = keqv_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_inv_sh_perf(
    inputs: dict,
    reward_a: NDArrayFloat,
    lev_a: NDArrayFloat,
    stop_a: NDArrayFloat,
    reten_a: NDArrayFloat,
    levsh_a: NDArrayFloat,
    reward_b: NDArrayFloat,
    lev_b: NDArrayFloat,
    stop_b: NDArrayFloat,
    reten_b: NDArrayFloat,
    levsh_b: NDArrayFloat,
    reward_c: NDArrayFloat,
    lev_c: NDArrayFloat,
    stop_c: NDArrayFloat,
    reten_c: NDArrayFloat,
    levsh_c: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
    g_min: List[float] = [None, None, None],
    g_max: List[float] = [None, None, None],
    l_min: List[float] = [None, None, None],
    l_max: List[float] = [None, None, None],
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investor performance across three counts of assets.

    Parameters:
        inputs: dictionary containing all execution details
        reward_a: 1 + time-average growth rate for invA with and without safe haven
        lev_a: leverages for invA with and without safe haven
        stop_a: stop-losses for invA with and without safe haven
        reten_a: retention ratios for invA with and without safe haven
        levsh_a: safe haven leverage for invA with and without safe haven
            ...
        filename: save path of plot
        g_min, g_max, l_min, l_max: graph bounds
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = reward_a.shape[0]

    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C0", "C4"]
    a_col = mpatches.Patch(color=cols[0], label="Uninsured", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Insured", alpha=0.8)
    c_col = mpatches.Patch(color="C3", label="Safe Haven", alpha=0.8)

    reward_a, reward_b, reward_c = (
        V_0 * reward_a**T,
        V_0 * reward_b**T,
        V_0 * reward_c**T,
    )

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = reward_a[i]
            elif n == 1:
                inv_x = reward_b[i]
            else:
                inv_x = reward_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
            x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

            x_mean = (x_mean.reshape(-1) - 1) * 100
            x_med = (x_med.reshape(-1) - 1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = (x_d.reshape(-1) - 1) * 100
            x_u = (x_u.reshape(-1) - 1) * 100

            # x_mean = np.log10(x_mean)
            # x_med = np.log10(x_med)
            # x_mad_lo = np.log10(x_mad_lo)
            # x_mad_up = np.log10(x_mad_up)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[0, n].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[0, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[0, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )

            axs[0, n].set_ylim(g_min[n], g_max[n])

            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = lev_a[i]
            elif n == 1:
                inv_x = lev_b[i]
            else:
                inv_x = lev_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[1, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[1, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[1, n].grid(True, linewidth=0.2)
            # axs[1, n].xaxis.set_ticklabels([])

            if i == ninv - 1:
                if n == 0:
                    inv_x = levsh_a[i]
                elif n == 1:
                    inv_x = levsh_b[i]
                else:
                    inv_x = levsh_c[i]

                x_mean = np.mean(inv_x, axis=1, keepdims=True)
                x_med = np.percentile(
                    inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
                )

                x_max = np.max(inv_x, axis=1, keepdims=True)
                x_min = np.min(inv_x, axis=1, keepdims=True)
                x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
                x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
                x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

                x_mean = x_mean.reshape(-1)
                x_med = x_med.reshape(-1)

                x_d = np.percentile(
                    inv_x, 25, axis=1, method="median_unbiased", keepdims=True
                )
                x_u = np.percentile(
                    inv_x, 75, axis=1, method="median_unbiased", keepdims=True
                )
                x_d = x_d.reshape(-1)
                x_u = x_u.reshape(-1)

                axs[1, n].plot(x_steps, x_mean, color="C3", linewidth=1)
                axs[1, n].plot(x_steps, x_med, color="C3", linewidth=1, linestyle="--")
                # axs[1, n].plot(x_steps, x_u, color="C3", linewidth=1, linestyle=":")
                axs[1, n].fill_between(
                    x_steps,
                    x_d,
                    x_u,
                    alpha=0.1,
                    facecolor="C3",
                    edgecolor="C3",
                    linewidth=2,
                    linestyle="--",
                )
                # axs[1, n].fill_between(
                #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
                # )

                axs[1, n].set_ylim(l_min[n], l_max[n])

                axs[1, n].grid(True, linewidth=0.2)
                # axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Leverage")
    axs[1, 1].xaxis.set_ticklabels([])
    axs[1, 2].xaxis.set_ticklabels([])

    for n in range(1, 3, 1):
        for i in range(0, ninv, 1):

            if n == 0:
                inv_x = stop_a[i]
            elif n == 1:
                inv_x = stop_b[i]
            else:
                inv_x = stop_c[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[2, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[2, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel("Stop-Loss " + r"$\lambda$ " + "(%)")

    for n in range(2, 3, 1):
        for i in range(0, 2, 1):

            if n == 0:
                inv_x = reten_a[i]
            elif n == 1:
                inv_x = reten_b[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = reten_c[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1) * 100
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1) * 100

            x_mean = x_mean.reshape(-1) * 100
            x_med = x_med.reshape(-1) * 100

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1) * 100
            x_u = x_u.reshape(-1) * 100

            axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            axs[3, n].fill_between(
                x_steps,
                x_d,
                x_u,
                alpha=0.1,
                facecolor=cols[i],
                edgecolor=cols[i],
                linewidth=2,
                linestyle="--",
            )
            # axs[3, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 2].set_ylabel("Retention " + r"$\phi$ " + "(%)")
    axs[1, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[2, 0].set_axis_off()
    axs[3, 0].set_axis_off()
    axs[3, 1].set_axis_off()

    axs[2, 0].xaxis.set_ticklabels([])
    axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 1].xaxis.set_ticklabels([])

    axs[0, 0].text(0.375, 1.2, "Inv A", size="large", transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.375, 1.2, "Inv B", size="large", transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.375, 1.2, "Inv C", size="large", transform=axs[0, 2].transAxes)

    fig.subplots_adjust(bottom=0.075, wspace=0.25, hspace=0.3)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_inv_sh_train(
    inputs: dict,
    loss_1: NDArrayFloat,
    tail_1: NDArrayFloat,
    shadow_1: NDArrayFloat,
    keqv1: NDArrayFloat,
    loss_2: NDArrayFloat,
    tail_2: NDArrayFloat,
    shadow_2: NDArrayFloat,
    keqv2: NDArrayFloat,
    loss_10: NDArrayFloat,
    tail_10: NDArrayFloat,
    shadow_10: NDArrayFloat,
    keqv10: NDArrayFloat,
    filename: Union[str, bytes, PathLike],
) -> None:
    """
    Plot summary of investor training across three counts of assets.

    Parameters:
        inputs: dictionary containing all execution details
        loss_1: mean critic loss for n_1 assets
        tail_1: tail exponent for n_1 assets
        shadow_1: critic shadow loss for n_1 assets
        keqv_1: equivilance multiplier for n_1 assets
            ...
        filename: save path of plot
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    ninv = loss_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C0", "C4"]
    a_col = mpatches.Patch(color=cols[0], label="Uninsured", alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label="Insured", alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 12))

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = loss_1[i]
            elif n == 1:
                inv_x = loss_2[i]
            else:
                inv_x = loss_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[0, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[0, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[0, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[0, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[0, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[0, n].grid(True, linewidth=0.2)
            axs[0, n].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Critic")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = tail_1[i]
            elif n == 1:
                inv_x = tail_2[i]
            else:
                inv_x = tail_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[1, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[1, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[1, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[1, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[1, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[1, n].grid(True, linewidth=0.2)
            axs[1, n].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Critic Tail " + r"$\alpha$")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = shadow_1[i]
            elif n == 1:
                inv_x = shadow_2[i]
            else:
                inv_x = shadow_10[i]

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            axs[2, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            # axs[2, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[2, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[2, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            axs[2, n].fill_between(
                x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            )
            axs[2, n].grid(True, linewidth=0.2)
            axs[2, n].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for n in range(3):
        for i in range(ninv):

            if n == 0:
                inv_x = keqv1[i]
            elif n == 1:
                inv_x = keqv2[i]
                axs[3, n].xaxis.set_ticklabels([])
            else:
                inv_x = keqv10[i]
                axs[3, n].xaxis.set_ticklabels([])

            x_mean = np.mean(inv_x, axis=1, keepdims=True)
            x_med = np.percentile(
                inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
            )

            x_max = np.max(inv_x, axis=1, keepdims=True)
            x_min = np.min(inv_x, axis=1, keepdims=True)
            x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
            x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
            x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

            x_mean = x_mean.reshape(-1)
            x_med = x_med.reshape(-1)

            x_d = np.percentile(
                inv_x, 25, axis=1, method="median_unbiased", keepdims=True
            )
            x_u = np.percentile(
                inv_x, 75, axis=1, method="median_unbiased", keepdims=True
            )
            x_d = x_d.reshape(-1)
            x_u = x_u.reshape(-1)

            # axs[3, n].plot(x_steps, x_mean, color=cols[i], linewidth=1)
            axs[3, n].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
            # axs[3, n].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
            # axs[3, n].fill_between(
            #     x_steps,
            #     x_d,
            #     x_u,
            #     alpha=0.1,
            #     facecolor=cols[i],
            #     edgecolor=cols[i],
            #     linewidth=2,
            #     linestyle="--",
            # )
            # axs[3, n].fill_between(
            #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
            # )
            axs[3, n].grid(True, linewidth=0.2)

    # axs[3, 0].xaxis.set_ticklabels([])
    axs[3, 0].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    axs[0, 0].text(0.375, 1.2, "Inv A", size="large", transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.375, 1.2, "Inv B", size="large", transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.375, 1.2, "Inv C", size="large", transform=axs[0, 2].transAxes)
    fig.subplots_adjust(bottom=0.075, wspace=0.25, hspace=0.3)
    fig.legend(
        handles=[a_col, b_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_mkt_inv(
    inputs: dict,
    reward: NDArrayFloat,
    loss: NDArrayFloat,
    tail: NDArrayFloat,
    shadow: NDArrayFloat,
    cmax: NDArrayFloat,
    keqv: NDArrayFloat,
    eval_start: NDArrayFloat,
    eval_len: NDArrayFloat,
    obs_days: List[int],
    filename: Union[str, bytes, PathLike],
    bin_size: int,
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investors for a single market.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        eval_start: start of market evaluation in history
        eval_len: length of evaluation episodes
        filename: save path of plot
        bin_size: bin sizes used for history aggregation
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    oday = reward.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(oday)]
    a_col = mpatches.Patch(color=cols[0], label=r"$D = $" + str(obs_days[0]), alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label=r"$D = $" + str(obs_days[1]), alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label=r"$D = $" + str(obs_days[2]), alpha=0.8)

    reward = V_0 * reward**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(oday):

        inv_x = reward[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_95 = np.percentile(inv_x, 5, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        # axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[0, 0].plot(x_steps, x_95, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].grid(True, linewidth=0.2)
        # axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")
    axs[0, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    for i in range(oday):

        end = eval_start[i].flatten() + eval_len[i].flatten()

        max_t_end = int(np.max(end))
        space_t_end = int(max_t_end / bin_size)

        time_end = np.linspace(np.min(eval_start[i].flatten()), max_t_end, space_t_end)
        inds_end = np.digitize(end, time_end)

        t_end = np.linspace(time_end.min(), time_end.max(), space_t_end)

        count_end = np.zeros(space_t_end)

        for c in range(0, end.shape[0]):
            x = inds_end[c]
            count_end[x - 1] += 1
        count_end /= end.shape[0]

        spl_end = make_interp_spline(time_end, count_end, k=3, bc_type="not-a-knot")
        count_smooth_end = np.maximum(spl_end(t_end), 0)

        axs[1, 0].plot(
            t_end, count_smooth_end * 100, color=cols[i], linewidth=1, linestyle="-."
        )
        axs[1, 0].grid(True, linewidth=0.2)
        # axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel(r"$P$" + "(Eval End)" + " (%)")

    for i in range(oday):

        inv_x = reward[i].flatten()
        var_x = eval_start[i].flatten() + eval_len[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[2, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[2, 0].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 0].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[2, 0].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[2, 0].grid(True, linewidth=0.2)
        # axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for i in range(oday):

        inv_x = eval_len[i].flatten()
        var_x = eval_start[i].flatten() + eval_len[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)
        x_95 = x_95.reshape(-1)

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[3, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[3, 0].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[3, 0].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[3, 0].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[3, 0].grid(True, linewidth=0.2)
        # axs[3, 0].xaxis.set_ticklabels([])

    axs[3, 0].set_ylabel("Eval Length")
    axs[3, 0].set_xlabel("Count " + r"$N$" + " (Days)")

    for i in range(oday):

        inv_x = loss[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 1].grid(True, linewidth=0.2)
        # axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_ylabel("Critic")

    for i in range(oday):

        inv_x = tail[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 1].grid(True, linewidth=0.2)
        # axs[1, 1].xaxis.set_ticklabels([])

    axs[1, 1].set_ylabel("Critic Tail " + r"$\alpha$")

    for i in range(oday):

        inv_x = shadow[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    axs[2, 1].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for i in range(oday):

        inv_x = keqv[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 1].grid(True, linewidth=0.2)
        # axs[3, 1].xaxis.set_ticklabels([])

    axs[3, 1].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")
    axs[3, 1].set_xlabel("Steps (1e" + str(exp) + ")")

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(inputs["algo"].upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_mkt_inv_perf(
    inputs: dict,
    reward_1: NDArrayFloat,
    eval_start_1: NDArrayFloat,
    eval_len_1: NDArrayFloat,
    reward_2: NDArrayFloat,
    eval_start_2: NDArrayFloat,
    eval_len_2: NDArrayFloat,
    obs_days: List[int],
    mkt_name: str,
    filename: Union[str, bytes, PathLike],
    bin_size: int,
    T: int = 1,
    V_0: float = 1,
) -> None:
    """
    Plot summary of investors for a single market for both algorithms.

    Parameters:
        inputs: dictionary containing all execution details
        reward: 1 + time-average growth rate
        eval_start: start of market evaluation in history
        eval_len: length of evaluation episodes
        bin_size: bin sizes used for history aggregation
        mkt_name: name of environment
        filename: save path of plot
        T: amount of compunding for reward
        V_0: initial value to compound
    """
    oday = reward_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(oday)]
    a_col = mpatches.Patch(color=cols[0], label=r"$D = $" + str(obs_days[0]), alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label=r"$D = $" + str(obs_days[1]), alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label=r"$D = $" + str(obs_days[2]), alpha=0.8)

    reward_1 = V_0 * reward_1**T
    reward_2 = V_0 * reward_2**T

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(oday):

        inv_x = reward_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_95 = np.percentile(inv_x, 5, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        # axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 0].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 0].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[0, 0].plot(x_steps, x_95, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 0].grid(True, linewidth=0.2)
        # axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")
    axs[0, 0].set_xlabel("Steps (1e" + str(exp) + ")")
    axs[0, 0].set_title("SAC", fontsize=14)

    for i in range(oday):

        end = eval_start_1[i].flatten() + eval_len_1[i].flatten()

        max_t_end = int(np.max(end))
        space_t_end = int(max_t_end / bin_size)

        time_end = np.linspace(
            np.min(eval_start_1[i].flatten()), max_t_end, space_t_end
        )
        inds_end = np.digitize(end, time_end)

        t_end = np.linspace(time_end.min(), time_end.max(), space_t_end)

        count_end = np.zeros(space_t_end)

        for c in range(0, end.shape[0]):
            x = inds_end[c]
            count_end[x - 1] += 1
        count_end /= end.shape[0]

        t_end, time_end, count_end = t_end[1:-1], time_end[1:-1], count_end[1:-1]

        spl_end = make_interp_spline(time_end, count_end, k=3, bc_type="not-a-knot")
        count_smooth_end = np.maximum(spl_end(t_end), 0)

        axs[1, 0].plot(
            t_end, count_smooth_end * 100, color=cols[i], linewidth=1, linestyle="-."
        )
        axs[1, 0].grid(True, linewidth=0.2)
        # axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel(r"$P$" + "(Eval End)" + " (%)")

    for i in range(oday):

        inv_x = reward_1[i].flatten()
        var_x = eval_start_1[i].flatten() + eval_len_1[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start_1[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        time, t, bucket = time[1:-1], t[1:-1], bucket[1:-1]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[2, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[2, 0].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 0].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[2, 0].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[2, 0].grid(True, linewidth=0.2)
        # axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Growth " + r"$\bar{g}$" + " (%)")

    for i in range(oday):

        inv_x = eval_len_1[i].flatten()
        var_x = eval_start_1[i].flatten() + eval_len_1[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start_1[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        time, t, bucket = time[1:-1], t[1:-1], bucket[1:-1]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)
        x_95 = x_95.reshape(-1)

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[3, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[3, 0].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[3, 0].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[3, 0].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[3, 0].grid(True, linewidth=0.2)
        # axs[3, 0].xaxis.set_ticklabels([])

    axs[3, 0].set_ylabel("Eval Length")
    axs[3, 0].set_xlabel("Count " + r"$N$" + " (Days)")

    for i in range(oday):

        inv_x = reward_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = (np.minimum(x_max, x_mean + x_mad).reshape(-1) - 1) * 100
        x_mad_lo = (np.maximum(x_min, x_mean - x_mad).reshape(-1) - 1) * 100

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.percentile(inv_x, 25, axis=1, method="median_unbiased", keepdims=True)
        x_u = np.percentile(inv_x, 75, axis=1, method="median_unbiased", keepdims=True)
        x_95 = np.percentile(inv_x, 5, axis=1, method="median_unbiased", keepdims=True)
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        # axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[0, 1].plot(x_steps, x_d, color=cols[i], linewidth=1, linestyle=":")
        # axs[0, 1].plot(x_steps, x_u, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 1].fill_between(
            x_steps,
            x_d,
            x_u,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        # axs[0, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[0, 1].plot(x_steps, x_95, color=cols[i], linewidth=1, linestyle=":")
        axs[0, 1].grid(True, linewidth=0.2)
        # axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_title("TD3", fontsize=14)

    for i in range(oday):

        end = eval_start_2[i].flatten() + eval_len_2[i].flatten()

        max_t_end = int(np.max(end))
        space_t_end = int(max_t_end / bin_size)

        time_end = np.linspace(
            np.min(eval_start_2[i].flatten()), max_t_end, space_t_end
        )
        inds_end = np.digitize(end, time_end)

        t_end = np.linspace(time_end.min(), time_end.max(), space_t_end)

        count_end = np.zeros(space_t_end)

        for c in range(0, end.shape[0]):
            x = inds_end[c]
            count_end[x - 1] += 1
        count_end /= end.shape[0]

        t_end, time_end, count_end = t_end[1:-1], time_end[1:-1], count_end[1:-1]

        spl_end = make_interp_spline(time_end, count_end, k=3, bc_type="not-a-knot")
        count_smooth_end = np.maximum(spl_end(t_end), 0)

        axs[1, 1].plot(
            t_end, count_smooth_end * 100, color=cols[i], linewidth=1, linestyle="-."
        )
        axs[1, 1].grid(True, linewidth=0.2)
        # axs[1, 1].xaxis.set_ticklabels([])

    for i in range(oday):

        inv_x = reward_2[i].flatten()
        var_x = eval_start_2[i].flatten() + eval_len_2[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start_2[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        time, t, bucket = time[1:-1], t[1:-1], bucket[1:-1]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = (x_mean.reshape(-1) - 1) * 100
        x_med = (x_med.reshape(-1) - 1) * 100

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = (x_d.reshape(-1) - 1) * 100
        x_u = (x_u.reshape(-1) - 1) * 100
        x_95 = (x_95.reshape(-1) - 1) * 100

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[2, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[2, 1].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[2, 1].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    for i in range(oday):

        inv_x = eval_len_2[i].flatten()
        var_x = eval_start_2[i].flatten() + eval_len_2[i].flatten()

        max_t = int(np.max(var_x))
        space_t = int(max_t / bin_size)

        time = np.linspace(np.min(eval_start_2[i].flatten()), max_t, space_t)
        inds = np.digitize(var_x, time)

        t = np.linspace(time.min(), time.max(), space_t)

        bucket = np.empty((space_t, inv_x.shape[0]))
        bucket[:] = np.nan

        for v in range(0, inv_x.shape[0]):
            x = inds[v]
            bucket[x - 1, v] = inv_x[v]

        time, t, bucket = time[1:-1], t[1:-1], bucket[1:-1]

        x_mean = np.nanmean(bucket, axis=1, keepdims=True)
        x_med = np.nanpercentile(
            bucket, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        x_d = np.nanpercentile(
            bucket, 25, axis=1, method="median_unbiased", keepdims=True
        )
        x_u = np.nanpercentile(
            bucket, 75, axis=1, method="median_unbiased", keepdims=True
        )
        x_95 = np.nanpercentile(
            bucket, 5, axis=1, method="median_unbiased", keepdims=True
        )
        x_d = x_d.reshape(-1)
        x_u = x_u.reshape(-1)
        x_95 = x_95.reshape(-1)

        spl_mean = make_interp_spline(time, x_mean, k=3, bc_type="not-a-knot")
        spl_med = make_interp_spline(time, x_med, k=3, bc_type="not-a-knot")
        spl_d = make_interp_spline(time, x_d, k=3, bc_type="not-a-knot")
        spl_u = make_interp_spline(time, x_u, k=3, bc_type="not-a-knot")
        spl_95 = make_interp_spline(time, x_95, k=3, bc_type="not-a-knot")

        mean_smooth = spl_mean(t)
        med_smooth = spl_med(t)
        d_smooth = spl_d(t)
        u_smooth = spl_u(t)
        p95_smooth = spl_95(t)

        # axs[3, 0].plot(t, mean_smooth, color=cols[i], linewidth=1)
        axs[3, 1].plot(t, med_smooth, color=cols[i], linewidth=1, linestyle="--")
        axs[3, 1].fill_between(
            t,
            d_smooth,
            u_smooth,
            alpha=0.1,
            facecolor=cols[i],
            edgecolor=cols[i],
            linewidth=2,
            linestyle="--",
        )
        axs[3, 1].plot(t, p95_smooth, color=cols[i], linewidth=1, linestyle=":")
        axs[3, 1].grid(True, linewidth=0.2)
        # axs[3, 1].xaxis.set_ticklabels([])

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(mkt_name.upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")


def plot_mkt_inv_train(
    inputs: dict,
    loss_1: NDArrayFloat,
    tail_1: NDArrayFloat,
    shadow_1: NDArrayFloat,
    cmax_1: NDArrayFloat,
    keqv_1: NDArrayFloat,
    loss_2: NDArrayFloat,
    tail_2: NDArrayFloat,
    shadow_2: NDArrayFloat,
    cmax_2: NDArrayFloat,
    keqv_2: NDArrayFloat,
    obs_days: List[int],
    mkt_name: str,
    filename: Union[str, bytes, PathLike],
) -> None:
    """
    Plot summary of investors training for a single market for both algorithms.

    Parameters:
        inputs: dictionary containing all execution details
        loss: critic loss
        tail: tail exponent
        shadow: shadow critic loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        filename: save path of plot
    """
    oday = loss_1.shape[0]
    cum_steps_log = np.array(
        [
            x
            for x in range(
                int(inputs["eval_freq"]),
                int(inputs["n_cumsteps"]) + int(inputs["eval_freq"]),
                int(inputs["eval_freq"]),
            )
        ]
    )

    exp = utils.get_exponent(cum_steps_log)
    x_steps = cum_steps_log / 10 ** (exp)

    cols = ["C" + str(x) for x in range(oday)]
    a_col = mpatches.Patch(color=cols[0], label=r"$D = $" + str(obs_days[0]), alpha=0.8)
    b_col = mpatches.Patch(color=cols[1], label=r"$D = $" + str(obs_days[1]), alpha=0.8)
    c_col = mpatches.Patch(color=cols[2], label=r"$D = $" + str(obs_days[2]), alpha=0.8)

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

    for i in range(oday):

        inv_x = loss_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 0].grid(True, linewidth=0.2)
        axs[0, 0].xaxis.set_ticklabels([])

    axs[0, 0].set_ylabel("Critic")
    axs[0, 0].set_title("SAC", fontsize=14)

    for i in range(oday):

        inv_x = tail_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 0].grid(True, linewidth=0.2)
        axs[1, 0].xaxis.set_ticklabels([])

    axs[1, 0].set_ylabel("Critic Tail " + r"$\alpha$")

    for i in range(oday):

        inv_x = shadow_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 0].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 0].grid(True, linewidth=0.2)
        axs[2, 0].xaxis.set_ticklabels([])

        inv_x = cmax_1[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max = np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 0].grid(True, linewidth=0.2)
        # axs[2, 0].xaxis.set_ticklabels([])

    axs[2, 0].set_ylabel("Critic Shadow " + r"$\mu_s$")

    for i in range(oday):

        inv_x = keqv_1[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 0].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 0].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 0].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 0].grid(True, linewidth=0.2)
        # axs[3, 0].xaxis.set_ticklabels([])

    axs[3, 0].set_ylabel("Multiplier " + r"$\kappa_{eqv}$")
    axs[3, 0].set_xlabel("Steps (1e" + str(exp) + ")")

    for i in range(oday):

        inv_x = loss_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[0, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[0, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[0, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[0, 1].grid(True, linewidth=0.2)
        axs[0, 1].xaxis.set_ticklabels([])

    axs[0, 1].set_title("TD3", fontsize=14)

    for i in range(oday):

        inv_x = tail_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[1, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[1, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[1, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[1, 1].grid(True, linewidth=0.2)
        axs[1, 1].xaxis.set_ticklabels([])

    for i in range(oday):

        inv_x = shadow_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        axs[2, 1].fill_between(
            x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        )
        axs[2, 1].grid(True, linewidth=0.2)
        axs[2, 1].xaxis.set_ticklabels([])

        inv_x = cmax_2[i]

        # x_mean = np.mean(inv_x, axis=1, keepdims=True)
        # x_med = np.percentile(
        #     inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        # )

        # x_max = np.max(inv_x, axis=1, keepdims=True)
        # x_min = np.min(inv_x, axis=1, keepdims=True)
        # x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        # x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        # x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        # x_mean = x_mean.reshape(-1)
        # x_med = x_med.reshape(-1)

        # axs[2, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle=":")
        # axs[2, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        # axs[2, 1].grid(True, linewidth=0.2)
        # axs[2, 1].xaxis.set_ticklabels([])

    for i in range(oday):

        inv_x = keqv_2[i]

        x_mean = np.mean(inv_x, axis=1, keepdims=True)
        x_med = np.percentile(
            inv_x, q=50, method="median_unbiased", axis=1, keepdims=True
        )

        x_max = np.max(inv_x, axis=1, keepdims=True)
        x_min = np.min(inv_x, axis=1, keepdims=True)
        x_mad = np.mean(np.abs(inv_x - x_mean), axis=1, keepdims=True)
        x_mad_up = np.minimum(x_max, x_mean + x_mad).reshape(-1)
        x_mad_lo = np.maximum(x_min, x_mean - x_mad).reshape(-1)

        x_mean = x_mean.reshape(-1)
        x_med = x_med.reshape(-1)

        # axs[3, 1].plot(x_steps, x_mean, color=cols[i], linewidth=1)
        axs[3, 1].plot(x_steps, x_med, color=cols[i], linewidth=1, linestyle="--")
        # axs[3, 1].fill_between(
        #     x_steps, x_mad_lo, x_mad_up, facecolor=cols[i], alpha=0.1
        # )
        axs[3, 1].grid(True, linewidth=0.2)
        axs[3, 1].xaxis.set_ticklabels([])

    fig.subplots_adjust(bottom=0.1, wspace=0.3, hspace=0.4)
    fig.legend(
        handles=[a_col, b_col, c_col],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize="medium",
    )

    plt.suptitle(mkt_name.upper(), fontsize=16)

    plt.savefig(filename + ".png", dpi=400, format="png")
    plt.savefig(filename + ".svg", format="svg")
