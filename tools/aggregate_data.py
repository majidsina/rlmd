"""
title:                  aggregate_data.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Responsible for extracting and aggregating outputted agent training data for
    final figure plotting and performance analysis.
"""

import sys

sys.path.append("./")

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]

from tools.utils import shadow_equiv


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
