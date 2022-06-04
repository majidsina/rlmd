"""
title:                  utils.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Responsible for various additional tools required for file naming,
    directory generation, shadow means, and some figure plotting.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.optimize as op
import scipy.special as sp
import torch as T

NDArrayFloat = npt.NDArray[np.float_]


def device_details(inputs: dict) -> None:
    """
    Summary of CPU thread allocation and CUDA GPU details.

    Parameters:
        inputs: dictionary containing all execution details
    """
    print(
        "--------------------------------------------------------------------------------"
    )
    print("PyTorch {}:".format(T.__version__))

    print(
        "    Number of CPU threads used for intraop/interop parallelism: {}/{}".format(
            T.get_num_threads(), T.get_num_interop_threads()
        )
    )

    print("    CUDA device available:", T.cuda.is_available())

    device = T.device(inputs["gpu"] if T.cuda.is_available() else "cpu")

    if inputs["gpu"][0:4] == "cuda":

        print("    CUDA version:", T.version.cuda)

        print("    CUDA device ID:", T.cuda.current_device())

        print(
            "    CUDA device name: {} w/ compatible capability {}.{}".format(
                T.cuda.get_device_name(device),
                T.cuda.get_device_capability(device)[0],
                T.cuda.get_device_capability(device)[1],
            )
        )

        print("    CUDA device properties:", T.cuda.get_device_properties(device))

        print(
            "    CUDA architectures this library was compiled for:",
            T.cuda.get_arch_list(),
        )

    print(
        "--------------------------------------------------------------------------------"
    )


def input_initialisation(
    inputs: dict,
    envs: List[int],
    algo: List[str],
    critic: List[str],
    multi_steps: List[int],
) -> dict:
    """
    Add training initialisation details to the inputs dictionary.

    Parameters:
        inputs: dictionary containing all user-specified agent execution details.

    Returns:
        new_inputs: inputs dictionary with initialisation details
    """
    new_inputs = {
        "test_agent": False,
        "envs": envs,
        "ENV_KEY": None,
        "algo_name": [a.upper() for a in algo],
        "critic_loss": [l.upper() for l in critic],
        "bootstraps": multi_steps,
        **inputs,
    }

    return new_inputs


def env_dynamics(
    gym_envs: Dict[str, list]
) -> Tuple[int, int, int, List[int], int, int, int]:
    """
    Obtain environment key limits based on reward dynamics.

    Parameters:
        gym_envs: all environment details

    Returns:
        multi_key: first multiplicative environment key
        sh_key: first multiplicative safe haven environment key
        market_key: first market environment key
        market_env_keys: final keys of each market environments
        gud_key: first guidance environment key
        two_key: first two-stage environment key
        counter_key: first countermeasure environment key
    """
    multi_key = [int(k) for k, v in gym_envs.items() if v[0] == "Coin_InvA"][0]
    sh_key = [int(k) for k, v in gym_envs.items() if v[0] == "Dice_SH_INSURED"][0]
    market_key = [int(k) for k, v in gym_envs.items() if v[0] == "SNP_InvA"][0]
    gud_key = [int(k) for k, v in gym_envs.items() if v[0] == "Laminar_2D_NW"][0]
    two_key = [int(k) for k, v in gym_envs.items() if v[0] == "2Stage_NW"][0]
    counter_key = [int(k) for k, v in gym_envs.items() if v[0] == "Counter_NW"][0]

    market_env_keys = [int(k) for k, v in gym_envs.items() if v[0][-5:] == "_InvC"]
    market_env_keys = [k for k in market_env_keys if k >= market_key]

    return multi_key, sh_key, market_key, market_env_keys, gud_key, two_key, counter_key


def load_market_data(
    key: int, market_env_keys: List[int], inputs: dict
) -> NDArrayFloat:
    """
    Locate and load previously generated historical financial market data.

    Parameters:
        key: environment key
        market_env_keys: seperator for different market data
        inputs: dictionary containing all execution details

    Returns:
        data: loaded financial data
    """
    if key <= market_env_keys[0]:
        return np.load(inputs["market_dir"] + "stooq_snp.npy")
    elif key <= market_env_keys[1]:
        return np.load(inputs["market_dir"] + "stooq_usei.npy")
    elif key <= market_env_keys[2]:
        return np.load(inputs["market_dir"] + "stooq_minor.npy")
    elif key <= market_env_keys[3]:
        return np.load(inputs["market_dir"] + "stooq_medium.npy")
    elif key <= market_env_keys[4]:
        return np.load(inputs["market_dir"] + "stooq_major.npy")
    elif key <= market_env_keys[5]:
        return np.load(inputs["market_dir"] + "stooq_dji.npy")
    elif key <= market_env_keys[6]:
        return np.load(inputs["market_dir"] + "stooq_full.npy")


def save_directory(inputs: dict, results: bool) -> str:
    """
    Provides string directory for data and plot saving names.

    Parameters:
        inputs: dictionary containing all execution details
        results: whether results (True) or model (False)

    Returns:
        directory: file path and name to give to current experiment plots
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    match inputs["dynamics"]:
        case "A":
            dyna = "additive/"
        case "M":
            dyna = "multiplicative/"
        case "MKT":
            dyna = "market/"
        case "GUD":
            dyna = "guidance/"

    dir = [
        "./results/",
        dyna,
        "data/",
        inputs["env_id"] + "/",
        inputs["env_id"] + "--",
        inputs["dynamics"] + "_",
        inputs["algo"] + "-",
        inputs["s_dist"],
        "_" + inputs["loss_fn"],
        "-" + str(inputs["critic_mean_type"]),
        "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
        "_M" + str(inputs["multi_steps"]),
        "_S" + str(int(inputs["n_cumsteps"]))[0:2] + "e" + str(step_exp - 1),
        "_N" + str(inputs["n_trials"]),
    ]

    if results == False:
        dir[2] = "models/"
        dir.append("t" + str(inputs["trial"]))

    if inputs["test_agent"]:
        dir[1] = dir[1][:-1]
        dir[1] += "-test/"

    directory = "".join(dir)

    return directory


def plot_subtitles(inputs: dict):
    """
    Generate subtitles for plots and figures.

    Parameters:
        inputs: dictionary containing all execution details

    Returns:
        sub: subtitle to be used in plots
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    sub = [
        inputs["env_id"] + "--",
        inputs["dynamics"] + "_",
        inputs["algo"] + "-",
        inputs["s_dist"],
        "_" + inputs["loss_fn"],
        "-" + str(inputs["critic_mean_type"]),
        "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
        "_M" + str(inputs["multi_steps"]),
        "_S" + str(int(inputs["n_cumsteps"]))[0:2] + "e" + str(step_exp - 1),
        "_N" + str(inputs["n_trials"]),
    ]

    sub = "".join(sub)

    return sub


def multi_log_dim(inputs: dict, n_gambles: int) -> int:
    """
    Generates risk-related parameter log dimension for multiplicative experiments
    with dimensions dependent on the environment characteristics.

    Parameters
        inputs: dictionary containing all execution details
        n_gambles: number of simultaneous identical gambles

    Returns:
        dim: dimensions for log array
    """
    env = inputs["env_id"]

    dim = 4

    if n_gambles > 1:
        dim += n_gambles

    if "_InvB" in env:
        dim += 1
    if "_InvC" in env:
        dim += 2

    if "_SH" in env:
        dim = 4 + 2 + 1

    return dim


def market_log_dim(inputs: dict, n_assets: int) -> int:
    """
    Generates risk-related parameter log dimension for market experiments
    with dimensions dependent on the environment characteristics.

    Parameters
        inputs: dictionary containing all execution details
        n_assets: number of assets for leverages

    Returns:
        dim: dimensions for log array
    """
    env = inputs["env_id"]

    dim = 4

    if n_assets > 1:
        dim += n_assets

    if "_InvB" in env:
        dim += 1
    if "_InvC" in env:
        dim += 2

    return dim


def get_exponent(array: NDArrayFloat) -> int:
    """
    Obtain expoenent for maximum array value used for scaling and axis labels.

    Parameters:
        array: array of usually cumulative steps in trial

    Returns:
        exp: exponent of max cumulative steps
    """
    max_step = np.max(array)

    if str(max_step)[0] == 1:
        exp = int(len(str(int(max_step))))
    else:
        exp = int(len(str(int(max_step))) - 1)

    return exp


def smoothing_func(ratio: float) -> float:
    """
    Smoothing function with a similar form to sigmoid but bounded between [0, 1] from
    https://math.stackexchange.com/questions/459872/adjustable-sigmoid-curve-s-curve-from-0-0-to-1-1.

    Parameters:
        ratio: current relative position within window

    Returns:
        ratio: smoothed ratio
    """
    return (np.sin(np.pi * (ratio - 1 / 2)) + 1) / 2


def action_window(
    action: NDArrayFloat,
    max_action: float,
    min_action: float,
    cum_step: int,
    max_step: int,
    warmup: int,
) -> NDArrayFloat:
    """
    Provide a widening smoothing window for the (absolute) maximum value of actions
    more in line with innate human decision making and to also prevent repeat selection
    of maximum values indicative of local extrema.

    Parameters:
        action: raw actions outputs from neural network
        cum_steps: cumulative training steps
        max_steps: maximum step limit to use smoothing action window
        warmup: length of initial random steps period

    Returns:
        action: smoothed action outputs
    """
    # only smooth post-initial warm-up actions
    if cum_step > warmup:
        ratio = cum_step / max_step
        width = smoothing_func(ratio)
        return np.clip(action, width * min_action, width * max_action)

    return action


def shadow_means(
    alpha: NDArrayFloat,
    min: NDArrayFloat,
    max: NDArrayFloat,
    min_mul: float,
    max_mul: float,
) -> NDArrayFloat:
    """
    Construct shadow mean given the tail exponent and sample min/max for
    varying multipliers.

    Parameters:
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form threshold of interest
        max_mul: upper bound multiplier of sample maximum to form upper limit

    Returns:
        shadow: shadow mean
    """
    low, high = min * min_mul, max * max_mul
    up_gamma = sp.gamma(1 - alpha) * sp.gammaincc(1 - alpha, alpha / high)
    shadow = (
        low + (high - low) * np.exp(alpha / high) * (alpha / high) ** alpha * up_gamma
    )

    return shadow


def shadow_equiv(
    mean: NDArrayFloat,
    alpha: NDArrayFloat,
    min: NDArrayFloat,
    max: NDArrayFloat,
    min_mul: float = 1,
) -> NDArrayFloat:
    """
    Estimate max multiplier required for equivalence between empirical (arthmetic)
    mean and shadow mean estimate. Utilises Powell hybrid method as implemented in
    MINPACK from
    https://digital.library.unt.edu/ark:/67531/metadc283470/m2/1/high_res_d/metadc283470.pdf.

    Parameters:
        mean: empirical mean
        alpha: sample tail index
        min: sample minimum critic loss
        max: sample maximum critic loss
        low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest

    Returns:
        max_mul: upper bound multiplier of maximum of distributions for equivalent
    """
    # select initial guess of equivilance multiplier
    x0 = 1

    if alpha < 1:
        f = lambda max_mul: shadow_means(alpha, min, max, min_mul, max_mul) - mean
        max_mul_solve = op.root(f, x0, method="hybr")  # Powell hybrid method
        return max_mul_solve.x

    else:
        return x0


def agent_shadow_mean(inputs: dict, loss: List[NDArrayFloat]) -> List[NDArrayFloat]:
    """
    Calculate shadow means for both critics performed at both start of each evaluation
    episode interval and when the training episode is terminated.

    Parameters:
        inputs: all training and evaluation details
        loss: empirical mean / min / max / (empty) shadow mean, tail exponents, mean actor loss

    Returns:
        shadow_means: power law heuristic estimates for the shadow means of both critics
    """
    low_mul, high_mul = inputs["shadow_low_mul"], inputs["shadow_high_mul"]

    low1, low2, high1, high2, alpha1, alpha2 = (
        loss[2],
        loss[3],
        loss[4],
        loss[5],
        loss[8],
        loss[9],
    )

    shadow1 = (
        shadow_means(alpha1, low1, high1, low_mul, high_mul) if alpha1 < 1 else loss[0]
    )
    shadow2 = (
        shadow_means(alpha2, low2, high2, low_mul, high_mul) if alpha2 < 1 else loss[1]
    )

    return [shadow1, shadow2]


def unique_histories(N, M_T, M_E, G_max, d_T, d_E) -> Tuple[float, float]:
    """
    Count (log10) number of unique possible histories after shuffling.

    Parameters:
        N: total length of history
        M_T: length of training interval
        M_E: length of evaluation interval
        G_max: maximum gap between training and evaluation
        d_T: training interval to be shuffled
        d_E: evaluation interval to be shuffled

    Returns:
        u_T: number of unique training histories (log10)
        u_E: number of unique evaluation histories (log10)
    """
    starts = math.log10(N - M_T - G_max - M_E)

    u_T = starts + M_T / d_T * math.log10(math.factorial(d_T))
    u_E = starts + M_E / d_E * math.log10(math.factorial(d_E))

    return u_T, u_E
