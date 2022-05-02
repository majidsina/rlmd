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
    directory generation, shadow means, and aggregating output training
    data for final figure plotting.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.optimize as op
import scipy.special as sp
import torch as T

NDArrayFloat = npt.NDArray[np.float_]


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
        "envs": envs,
        "ENV_KEY": None,
        "algo_name": [a.upper() for a in algo],
        "critic_loss": [l.upper() for l in critic],
        "bootstraps": multi_steps,
        **inputs,
    }

    return new_inputs


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


def env_dynamics(gym_envs: Dict[str, list]) -> list:
    """
    Obtain environment key limits based on reward dynamics.

    Parameters:
        gym_envs: all environment details

    Returns:
        multi_key: first multiplicative env key
        sh_key: first multiplicative safe haven env key
        market_key: first market env key
        market_env_keys: final keys of each market envS
        gud_key: first guidance env key
        two_key: first two-stage key
        counter_key: first countermeasure key
    """
    multi_key = [int(k) for k, v in gym_envs.items() if v[0] == "Coin_InvA"][0]
    sh_key = [int(k) for k, v in gym_envs.items() if v[0] == "Dice_SH_INSURED"][0]
    market_key = [int(k) for k, v in gym_envs.items() if v[0] == "SNP_InvA"][0]
    gud_key = [int(k) for k, v in gym_envs.items() if v[0] == "Laminar_2D_NW"][0]
    two_key = [int(k) for k, v in gym_envs.items() if v[0] == "2Stage_NW"][0]
    counter_key = [int(k) for k, v in gym_envs.items() if v[0] == "Counter_NW"][0]

    market_env_keys = [int(k) for k, v in gym_envs.items() if v[0][-5:] == "_InvC"]
    market_env_keys = [k for k in market_env_keys if k >= market_key]

    seperator = [
        multi_key,
        sh_key,
        market_key,
        market_env_keys,
        gud_key,
        two_key,
        counter_key,
    ]

    return seperator


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


def sac_critic_stability(
    step: int,
    q1: T.FloatTensor,
    q2: T.FloatTensor,
    q_soft: T.FloatTensor,
    q_target: T.FloatTensor,
) -> None:
    """
    Check whether SAC critic losses contain internal backpropagation errors and print
    mini-batch components.

    Parameters:
        step: learning step
        q1, q2: critic losses 1 and 2
        q_target, q_soft: target critic losses
    """
    combine = T.concat([q1, q2, q_target])

    if T.any(T.isnan(combine) == True):

        q1, q2, q_soft, q_target = (
            q1.view(-1),
            q2.view(-1),
            q_soft.view(-1),
            q_target.view(-1),
        )

        print(
            """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within SAC critic losses.

            Learning Step: {}

            Critic Loss 1:
            {}

            Criict Loss 2:
            {}

            Critic Soft Target Loss:
            {}

            Critic Target Loss:
            {}
            """.format(
                step, q1, q2, q_soft, q_target
            )
        )


def td3_critic_stability(
    step: int, q1: T.FloatTensor, q2: T.FloatTensor, q_target: T.FloatTensor
) -> None:
    """
    Check whether SAC critic losses contain internal backpropagation errors and print
    mini-batch components.

    Parameters:
        step: learning step
        q1, q2: critic losses 1 and 2
        q_target, q_soft: target critic losses
    """
    combine = T.concat([q1, q2, q_target])

    if T.any(T.isnan(combine) == True):

        q1, q2, q_target = q1.view(-1), q2.view(-1), q_target.view(-1)

        print(
            """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within TD3 critic losses.

            Learning Step: {}

            Critic Loss 1:
            {}

            Criict Loss 2:
            {}

            Critic Target Loss:
            {}
            """.format(
                step, q1, q2, q_target
            )
        )


def critic_learning(
    cum_step: int,
    batch_size: int,
    episode: int,
    step: int,
    loss: List[float],
    loss_params: List[float],
    logtemp: NDArrayFloat,
    state: NDArrayFloat,
    action: NDArrayFloat,
    reward: float,
    next_state: NDArrayFloat,
    done: bool,
    learn_done: bool = None,
    risk: NDArrayFloat = None,
) -> None:
    """
    Check whether critic losses contain internal backpropagation errors causing the
    entire script to terminate as learning ceases to occur.

    Parameters:
        cum_step: current amount of cumulative steps
        batch_size: mini-batch size
        episode: current episode number
        step: current step in episode
        loss: loss values of critic 1, critic 2 and actor
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
        logtemp: log entropy adjustment factor (temperature)
        state: initial state
        action: array of actions to be taken determined by actor network
        reward: agent signal to maximise
        next_state: state arrived at from taking action
        done: Boolean flag for episode termination
        learn_done: Boolean flag for whether genuine termination
        risk: collection of additional data retrieved
    """
    if cum_step > batch_size:

        critic = np.array(loss[0:6] + loss[8:10], dtype=np.float32).flatten()

        if np.any(np.isnan(critic) == True):

            print(
                """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within critic losses
            indicating failed agent neural network backpropagation. This issue is
            likely due to several reasons either individual or combined.

            For additive environments, mini-batch losses might be excessively supressed
            by highly smoothing loss functions.

            For multiplicative/market environments it may be due to several reasons
            such as the previous cause, state components diverge due to the possibility
            of unbounded environments, and/or other mysterious events.

            Cumulative Step: {}
            Episode: {}
            Step: {}

            Critic Loss 1:
            Mean: {}
            Min: {}
            Max: {}
            Tail: {}
            Scale: {}
            Kernel: {}

            Critic Loss 2:
            Mean: {}
            Min: {}
            Max: {}
            Tail: {}
            Scale: {}
            Kernel: {}

            Actor Loss Mean:
            {}

            Log Entropy Temperature (only SAC):
            {}

            State:
            {}

            Action:
            {}

            Reward:
            {}

            Next State:
            {}

            Done:
            {}
            """.format(
                    cum_step,
                    episode,
                    step,
                    loss[0],
                    loss[2],
                    loss[4],
                    loss[8],
                    loss_params[0],
                    loss_params[2],
                    loss[1],
                    loss[3],
                    loss[5],
                    loss[9],
                    loss_params[1],
                    loss_params[3],
                    logtemp,
                    loss[-1],
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            if learn_done != None:
                print(
                    """
                Learn Done:
                {}

                Risk:
                {}
                """.format(
                        learn_done, risk
                    )
                )

            # terminate script as agent learning is compromised
            exit()


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


def add_loss_aggregate(
    env_keys: list,
    gym_envs: dict,
    inputs: dict,
    algos: list = ["TD3"],
    loss: list = ["MSE"],
) -> NDArrayFloat:
    """
    Combine environment loss evaluation data for additive experiments.

    Parameters:
        env_keys: list of environments
        gym_envvs: dictionary of all environment details
        inputs: dictionary of additive execution parameters
        algos: list of RL algorithms
        loss: list of critic loss functions

    Retuens:
        data: aggregated evaluation data across all ioss functions
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    data = np.zeros(
        (
            len(env_keys),
            len(algos),
            len(loss),
            int(inputs["n_trials"]),
            int(inputs["n_cumsteps"] / inputs["eval_freq"]),
            int(inputs["n_eval"]),
            20,
        )
    )

    name = [gym_envs[str(key)][0] for key in env_keys]
    path = ["./results/additive/data/" + n + "/" for n in name]

    num1 = 0
    for env in name:
        num2 = 0
        for a in algos:
            num3 = 0
            for l in loss:

                dir = [
                    "--",
                    "A_",
                    a + "-",
                    inputs["s_dist"],
                    "_" + l,
                    "-" + str(inputs["critic_mean_type"]),
                    "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
                    "_M" + str(inputs["multi_steps"]),
                    "_S"
                    + str(int(inputs["n_cumsteps"]))[0:2]
                    + "e"
                    + str(step_exp - 1),
                    "_N" + str(inputs["n_trials"]),
                ]

                dir = "".join(dir)

                data_path = path[num1] + env + dir

                file = np.load(data_path + "_eval.npy")

                data[num1, num2, num3] = file

                num3 += 1
            num2 += 1
        num1 += 1

    return data


def add_multi_aggregate(
    env_keys: list,
    gym_envs: dict,
    inputs: dict,
    algos: list = ["TD3"],
    multi: list = [1],
) -> NDArrayFloat:
    """
    Combine environment multi-step evaluation data for additive experiments.

    Parameters:
        env_keys: list of environments
        gym_envvs: dictionary of all environment details
        inputs: dictionary of additive execution parameters
        algos: list of RL algorithms
        loss: list of multi-steps

    Retuens:
        data: aggregated evaluation data across all ioss functions
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    data = np.zeros(
        (
            len(env_keys),
            len(algos),
            len(multi),
            int(inputs["n_trials"]),
            int(inputs["n_cumsteps"] / inputs["eval_freq"]),
            int(inputs["n_eval"]),
            20,
        )
    )

    name = [gym_envs[str(key)][0] for key in env_keys]
    path = ["./results/additive/data/" + n + "/" for n in name]

    num1 = 0
    for env in name:
        num2 = 0
        for a in algos:
            num3 = 0
            for m in multi:

                dir = [
                    "--",
                    "A_",
                    a + "-",
                    inputs["s_dist"],
                    "_MSE",
                    "-" + str(inputs["critic_mean_type"]),
                    "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
                    "_M" + str(m),
                    "_S"
                    + str(int(inputs["n_cumsteps"]))[0:2]
                    + "e"
                    + str(step_exp - 1),
                    "_N" + str(inputs["n_trials"]),
                ]

                dir = "".join(dir)

                data_path = path[num1] + env + dir

                file = np.load(data_path + "_eval.npy")

                data[num1, num2, num3] = file

                num3 += 1
            num2 += 1
        num1 += 1

    return data


def add_summary(
    inputs: dict, data: NDArrayFloat
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Seperate and aggregate arrays into variables.

    Parameters:
        inputs: dictionary of execution parameters
        data: aggregated evaluation data across all experiments

    Retuens:
        reward: final scores
        loss: critic loss
        scale: Cauchy scales
        kernerl: CIM kernel size
        logtemp: SAC log entropy temperature
        tail: critic tail exponent
        shadow: critic shadow loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means

    """
    n_env, n_algo, n_data = data.shape[0], data.shape[1], data.shape[2]

    count_x = int(inputs["n_cumsteps"] / inputs["eval_freq"])
    count_y = int(inputs["n_trials"] * int(inputs["n_eval"]))
    count_z = int(inputs["n_trials"])

    reward = np.zeros((n_env, n_algo, n_data, count_x, count_y))
    loss = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    scale = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    kernel = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    logtemp = np.zeros((n_env, n_algo, n_data, count_x, count_z))

    shadow = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    tail = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    cmin = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    cmax = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))
    keqv = np.zeros((n_env, n_algo, n_data, count_x, count_z * 2))

    for e in range(n_env):
        for a in range(n_algo):
            for d in range(n_data):
                for t in range(count_x):
                    for n in range(inputs["n_trials"]):

                        loss[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 3:5
                        ]
                        scale[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 15:17
                        ]
                        kernel[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 17:19
                        ]
                        logtemp[e, a, d, t, n] = data[e, a, d, n, t, 0, 14]

                        shadow[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 9:11
                        ]
                        tail[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 11:13
                        ]
                        cmin[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 5:7
                        ]
                        cmax[e, a, d, t, (n * 2) : (n * 2) + 2] = data[
                            e, a, d, n, t, 0, 7:9
                        ]

                        for s in range(int(inputs["n_eval"])):
                            reward[e, a, d, t, s + n * int(inputs["n_eval"])] = data[
                                e, a, d, n, t, s, 1
                            ]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for e in range(n_env):
        for a in range(n_algo):
            for d in range(n_data):
                for t in range(count_x):
                    for n in range(inputs["n_trials"] * 2):
                        keqv[e, a, d, t, n] = shadow_equiv(
                            loss[e, a, d, t, n],
                            tail[e, a, d, t, n],
                            cmin[e, a, d, t, n],
                            loss[e, a, d, t, n],
                            1,
                        )

    return reward, loss, scale, kernel, logtemp, tail, shadow, cmax, keqv


def mul_inv_aggregate(
    env_keys: list,
    n_gambles: int,
    gym_envs: dict,
    inputs: dict,
    safe_haven: bool = False,
) -> NDArrayFloat:
    """
    Combine environment evaluation data for investors across the same number of assets.

    Parameters:
        env_keys: list of environments
        n_gambles: number of simultaneous identical gambles
        gym_envvs: dictionary of all environment details
        inputs: dictionary of multiplicative execution parameters
        safe_have: whether investor is using insurance safe haven

    Retuens:
        eval: aggregated evaluation data across all investors
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    dir = [
        "--",
        "M_",
        inputs["algo"] + "-",
        inputs["s_dist"],
        "_" + inputs["loss_fn"],
        "-" + str(inputs["critic_mean_type"]),
        "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
        "_M" + str(inputs["multi_steps"]),
        "_S" + str(int(inputs["n_cumsteps"]))[0:2] + "e" + str(step_exp - 1),
        "_N" + str(inputs["n_trials"]),
    ]

    dir = "".join(dir)

    sh = 1 if safe_haven == True else 0

    name = [gym_envs[str(key)][0] + "_n" + str(n_gambles) for key in env_keys]
    path = ["./results/multiplicative/data/" + n + "/" for n in name]

    eval = np.zeros(
        (
            len(name),
            int(inputs["n_trials"]),
            int(inputs["n_cumsteps"] / inputs["eval_freq"]),
            int(inputs["n_eval"]),
            20 + sh + 50,
        )
    )
    num = 0
    for env in name:
        data_path = path[num] + env + dir

        file1 = np.load(data_path + "_eval.npy")
        file2 = np.load(data_path + "_eval_risk.npy")
        file = np.concatenate((file1, file2), axis=3)

        eval[num, :, :, :, : 20 + file2.shape[3]] = file

        num += 1

    return eval


def mul_inv_n_summary(
    mul_inputs: dict, aggregate_n: NDArrayFloat, safe_haven: bool = False
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Seperate and aggregate arrays into variables.

    Parameters:
        inputs: dictionary of execution parameters
        aggregate_n: aggregated evaluation data across all investors
        safe_have: whether investor is using insurance safe haven

    Retuens:
        reward: 1 + time-average growth rate
        lev: leverages
        stop: stop-losses
        reten: retention ratios
        loss: critic loss
        tail: critic tail exponent
        shadow: critic shadow loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        lev_sh: leverage for safe haven
    """
    ninv = aggregate_n.shape[0]

    count_x = int(mul_inputs["n_cumsteps"] / mul_inputs["eval_freq"])
    count_y = int(mul_inputs["n_trials"] * int(mul_inputs["n_eval"]))
    count_z = int(mul_inputs["n_trials"])

    loss = np.zeros((ninv, count_x, count_z * 2))
    shadow = np.zeros((ninv, count_x, count_z * 2))
    tail = np.zeros((ninv, count_x, count_z * 2))
    cmin = np.zeros((ninv, count_x, count_z * 2))
    cmax = np.zeros((ninv, count_x, count_z * 2))
    keqv = np.zeros((ninv, count_x, count_z * 2))

    reward = np.zeros((ninv, count_x, count_y))
    lev = np.zeros((ninv, count_x, count_y))
    stop = np.zeros((ninv, count_x, count_y))
    reten = np.zeros((ninv, count_x, count_y))
    lev_sh = np.zeros((ninv, count_x, count_y))

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs["n_trials"]):

                loss[i, t, (n * 2) : (n * 2) + 2] = aggregate_n[i, n, t, 0, 3:5]
                shadow[i, t, (n * 2) : (n * 2) + 2] = aggregate_n[i, n, t, 0, 9:11]
                tail[i, t, (n * 2) : (n * 2) + 2] = aggregate_n[i, n, t, 0, 11:13]
                cmin[i, t, (n * 2) : (n * 2) + 2] = aggregate_n[i, n, t, 0, 5:7]
                cmax[i, t, (n * 2) : (n * 2) + 2] = aggregate_n[i, n, t, 0, 7:9]

                for s in range(int(mul_inputs["n_eval"])):
                    reward[i, t, s + n * int(mul_inputs["n_eval"])] = aggregate_n[
                        i, n, t, s, 20
                    ]
                    lev[i, t, s + n * int(mul_inputs["n_eval"])] = aggregate_n[
                        i, n, t, s, 23
                    ]
                    stop[i, t, s + n * int(mul_inputs["n_eval"])] = aggregate_n[
                        i, n, t, s, 24
                    ]
                    reten[i, t, s + n * int(mul_inputs["n_eval"])] = aggregate_n[
                        i, n, t, s, 25
                    ]

                    if safe_haven == True:
                        lev_sh[i, t, s + n * int(mul_inputs["n_eval"])] = aggregate_n[
                            i, n, t, s, 26
                        ]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for i in range(ninv):
        for t in range(count_x):
            for n in range(mul_inputs["n_trials"] * 2):
                keqv[i, t, n] = shadow_equiv(
                    loss[i, t, n], tail[i, t, n], cmin[i, t, n], loss[i, t, n], 1
                )

    return reward, lev, stop, reten, loss, tail, shadow, cmax, keqv, lev_sh


def mkt_obs_aggregate(
    env_keys: List[int],
    obs_days: List[int],
    action_days: int,
    gym_envs: dict,
    inputs: dict,
) -> NDArrayFloat:
    """
    Combine environment evaluation data for market common investor across the
    days observed.

    Parameters:
        env_keys: list of environments
        obs_days: number of previous observed days
        action_days : number of days between agent actions
        gym_envvs: dictionary of all environment details
        inputs: dictionary of multiplicative execution parameters

    Retuens:
        eval: aggregated evaluation data across all investors
    """
    step_exp = int(len(str(int(inputs["n_cumsteps"]))) - 1)
    buff_exp = int(len(str(int(inputs["buffer"]))) - 1)

    dir = [
        "--",
        "MKT_",
        inputs["algo"] + "-",
        inputs["s_dist"],
        "_" + inputs["loss_fn"],
        "-" + str(inputs["critic_mean_type"]),
        "_B" + str(int(inputs["buffer"]))[0:2] + "e" + str(buff_exp - 1),
        "_M" + str(inputs["multi_steps"]),
        "_S" + str(int(inputs["n_cumsteps"]))[0:2] + "e" + str(step_exp - 1),
        "_N" + str(inputs["n_trials"]),
    ]

    dir = "".join(dir)

    name = [
        gym_envs[str(env_keys)][0] + "_D" + str(obs) + "_T" + str(action_days)
        for obs in obs_days
    ]
    path = ["./results/market/data/" + n + "/" for n in name]

    eval = np.zeros(
        (
            len(name),
            int(inputs["n_trials"]),
            int(inputs["n_cumsteps"] / inputs["eval_freq"]),
            int(inputs["n_eval"]),
            20 + 50,
        )
    )
    num = 0
    for env in name:
        data_path = path[num] + env + dir

        file1 = np.load(data_path + "_eval.npy")
        file2 = np.load(data_path + "_eval_risk.npy")
        file = np.concatenate((file1, file2), axis=3)

        eval[num, :, :, :, : 20 + file2.shape[3]] = file

        num += 1

    return eval


def mkt_obs_summary(
    mar_inputs: dict, aggregate_o: NDArrayFloat
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Seperate and aggregate arrays into variables.

    Parameters:
        inputs: dictionary of execution parameters
        aggregate_o: aggregated evaluation data across all investors

    Retuens:
        reward: 1 + time-average growth rate
        loss: critic loss
        tail: critic tail exponent
        shadow: critic shadow loss
        cmax: maximum critic loss
        keqv: max multiplier for equvilance between shadow and empirical means
        mkt_start: start of market evaluation in history
        eval_len: length of evaluation episodes
    """
    oday = aggregate_o.shape[0]

    count_x = int(mar_inputs["n_cumsteps"] / mar_inputs["eval_freq"])
    count_y = int(mar_inputs["n_trials"] * int(mar_inputs["n_eval"]))
    count_z = int(mar_inputs["n_trials"])

    loss = np.zeros((oday, count_x, count_z * 2))
    shadow = np.zeros((oday, count_x, count_z * 2))
    tail = np.zeros((oday, count_x, count_z * 2))
    cmin = np.zeros((oday, count_x, count_z * 2))
    cmax = np.zeros((oday, count_x, count_z * 2))
    keqv = np.zeros((oday, count_x, count_z * 2))

    reward = np.zeros((oday, count_x, count_y))
    mkt_start = np.zeros((oday, count_x, count_y))
    eval_len = np.zeros((oday, count_x, count_y))

    for i in range(oday):
        for t in range(count_x):
            for n in range(mar_inputs["n_trials"]):

                loss[i, t, (n * 2) : (n * 2) + 2] = aggregate_o[i, n, t, 0, 3:5]
                shadow[i, t, (n * 2) : (n * 2) + 2] = aggregate_o[i, n, t, 0, 9:11]
                tail[i, t, (n * 2) : (n * 2) + 2] = aggregate_o[i, n, t, 0, 11:13]
                cmin[i, t, (n * 2) : (n * 2) + 2] = aggregate_o[i, n, t, 0, 5:7]
                cmax[i, t, (n * 2) : (n * 2) + 2] = aggregate_o[i, n, t, 0, 7:9]

                for s in range(int(mar_inputs["n_eval"])):
                    mkt_start[i, t, s + n * int(mar_inputs["n_eval"])] = aggregate_o[
                        i, n, t, s, 20
                    ]
                    eval_len[i, t, s + n * int(mar_inputs["n_eval"])] = aggregate_o[
                        i, n, t, s, 2
                    ]
                    reward[i, t, s + n * int(mar_inputs["n_eval"])] = aggregate_o[
                        i, n, t, s, 21
                    ]

    shadow[np.isnan(shadow)] = loss[np.isnan(shadow)]

    for i in range(oday):
        for t in range(count_x):
            for n in range(mar_inputs["n_trials"] * 2):
                keqv[i, t, n] = shadow_equiv(
                    loss[i, t, n], tail[i, t, n], cmin[i, t, n], loss[i, t, n], 1
                )

    return reward, loss, tail, shadow, cmax, keqv, mkt_start, eval_len
