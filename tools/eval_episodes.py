"""
title:                  eval_episodes.py
python version:         3.10
torch verison:          1.11
gym version:            0.23
pybullet version:       3.2

code style:             black==22.3
import style:           isort==5.10

author:                 J. S. Grewal (2022)
email:                  <raja (_) grewal1 [at] pm {dot} me>
linkedin:               https://www.linkedin.com/in/rajagrewal
copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html

Description:
    Responsible for performing agent evaluation episodes for both
    additive and multiplicative environments.
"""

import sys

sys.path.append("./")

import time
from datetime import datetime
from typing import List

import gym
import numpy as np
import numpy.typing as npt
import pybullet_envs

NDArrayFloat = npt.NDArray[np.float_]

import envs.coin_flip_envs as coin_flip_envs
import envs.dice_roll_envs as dice_roll_envs
import envs.dice_roll_sh_envs as dice_roll_sh_envs
import envs.gbm_envs as gbm_envs
import envs.market_envs as market_envs
import tools.env_resources as env_resources
import tools.utils as utils

# import envs.laminar_envs as laminar_envs


def eval_additive(
    agent: object,
    inputs: dict,
    eval_log: NDArrayFloat,
    multi_step: int,
    cum_steps: int,
    round: int,
    eval_run: int,
    loss: List[NDArrayFloat],
    logtemp: NDArrayFloat,
    loss_params: List[NDArrayFloat],
) -> None:
    """
    Evaluates agent policy on addtive environment without learning for a fixed
    number of episodes.

    Parameters:
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        multi_step: current bootstrapping
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print(
        "{} E{}_m{} {}-{}-{}-{} cst {} w/ {} evaluations: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            inputs["ENV_KEY"],
            multi_step,
            inputs["algo"],
            inputs["s_dist"],
            inputs["loss_fn"],
            round + 1,
            cum_steps,
            int(inputs["n_eval"]),
            np.mean(loss[0:2]),
            np.mean(loss[4:6]),
            np.mean(loss[6:8]),
            np.mean(loss[8:10]),
            np.mean(loss_params[0:2]),
            np.mean(loss_params[2:4]),
            loss[8],
            np.exp(logtemp),
        )
    )

    eval_env = gym.make(inputs["env_id"])

    for eval in range(int(inputs["n_eval"])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            run_action = agent.eval_next_action(run_state)
            run_next_state, eval_reward, run_done, _ = eval_env.step(run_action)

            run_reward += eval_reward
            run_state = run_next_state
            run_step += 1

            # prevent evaluation from running forever
            if run_reward >= int(inputs["max_eval_reward"]):
                break

        end_time = time.perf_counter()

        eval_log[round, eval_run, eval, 0] = end_time - start_time
        eval_log[round, eval_run, eval, 1] = run_reward
        eval_log[round, eval_run, eval, 2] = run_step
        eval_log[round, eval_run, eval, 3:14] = loss
        eval_log[round, eval_run, eval, 14] = logtemp
        eval_log[round, eval_run, eval, 15:19] = loss_params
        eval_log[round, eval_run, eval, 19] = cum_steps

        print(
            "{} Episode {}: r/st {:1.0f}/{}".format(
                datetime.now().strftime("%d %H:%M:%S"), eval + 1, run_reward, run_step
            )
        )

    reward = eval_log[round, eval_run, :, 1]
    mean_reward = np.mean(reward)
    med_reward = np.percentile(reward, q=50, method="median_unbiased")
    mad_reward = np.mean(np.abs(reward - mean_reward))
    std_reward = np.std(reward, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.percentile(step, q=50, method="median_unbiased")
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [
        mean_reward,
        med_reward,
        mad_reward,
        std_reward,
        mean_step,
        med_step,
        mad_step,
        std_step,
    ]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(
        eval_log[round, eval_run, :, 0]
    )

    print(
        "{} Summary {:1.0f}/s mean/med/mad/std: r {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            steps_sec,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            stats[4],
            stats[5],
            stats[6],
            stats[7],
        )
    )


def eval_multiplicative(
    n_gambles: int,
    agent: object,
    inputs: dict,
    eval_log: NDArrayFloat,
    eval_risk_log: NDArrayFloat,
    multi_step: int,
    cum_steps: int,
    round: int,
    eval_run: int,
    loss: List[NDArrayFloat],
    logtemp: NDArrayFloat,
    loss_params: List[NDArrayFloat],
) -> None:
    """
    Evaluates agent policy on multiplicative environment without learning for a fixed
    number of episodes.

    Parameters:
        n_gambles: number of simultaneous identical gambles
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        eval_risk_log: array of exiting evalaution risk results
        multi_step: current bootstrapping
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print(
        "{} E{}_m{}_n{} {}-{}-{}-{} cst {} w/ {} evaluations: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            inputs["ENV_KEY"],
            multi_step,
            n_gambles,
            inputs["algo"],
            inputs["s_dist"],
            inputs["loss_fn"],
            round + 1,
            cum_steps,
            int(inputs["n_eval"]),
            np.mean(loss[0:2]),
            np.mean(loss[4:6]),
            np.mean(loss[6:8]),
            np.mean(loss[8:10]),
            np.mean(loss_params[0:2]),
            np.mean(loss_params[2:4]),
            loss[8],
            np.exp(logtemp),
        )
    )

    eval_env = eval(inputs["env_gym"])

    for eval_epis in range(int(inputs["n_eval"])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done, run_step, run_reward = False, 0, 0

        # constant action per episode for Markovian environments
        run_action = agent.eval_next_action(run_state)

        if cum_steps <= int(inputs["smoothing_window"]):
            run_action = utils.action_window(
                run_action,
                inputs["max_action"],
                inputs["min_action"],
                cum_steps,
                int(inputs["smoothing_window"]),
                int(inputs["random"]),
            )

        while not run_done:
            run_next_state, eval_reward, done_flags, run_risk = eval_env.step(
                run_action
            )

            run_done = done_flags[0]

            run_reward = eval_reward
            run_state = run_next_state
            run_step += 1

            if run_step == int(inputs["max_eval_steps"]):
                break

        end_time = time.perf_counter()

        eval_log[round, eval_run, eval_epis, 0] = end_time - start_time
        eval_log[round, eval_run, eval_epis, 1] = run_reward
        eval_log[round, eval_run, eval_epis, 2] = run_step
        eval_log[round, eval_run, eval_epis, 3:14] = loss
        eval_log[round, eval_run, eval_epis, 14] = logtemp
        eval_log[round, eval_run, eval_epis, 15:19] = loss_params
        eval_log[round, eval_run, eval_epis, 19] = cum_steps
        eval_risk_log[round, eval_run, eval_epis, :] = run_risk

    reward = eval_log[round, eval_run, :, 1]
    mean_reward = np.mean(reward)
    med_reward = np.percentile(reward, q=50, method="median_unbiased")
    reward_95 = np.percentile(reward, q=5, method="median_unbiased")
    mad_reward = np.mean(np.abs(reward - mean_reward))
    std_reward = np.std(reward, ddof=0)

    val = eval_risk_log[round, eval_run, :, 1]
    mean_val = np.mean(val)
    med_val = np.percentile(val, q=50, method="median_unbiased")
    val_95 = np.percentile(val, q=5, method="median_unbiased")
    mad_val = np.mean(np.abs(val - mean_val))

    lev = eval_risk_log[round, eval_run, :, 3]
    mean_lev = np.mean(lev)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.percentile(step, q=50, method="median_unbiased")
    step_95 = np.percentile(step, q=5, method="median_unbiased")
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [
        mean_lev * 100,
        (mean_reward - 1) * 100,
        (med_reward - 1) * 100,
        (reward_95 - 1) * 100,
        mad_reward * 100,
        std_reward * 100,
        mean_val,
        med_val,
        val_95,
        mad_val,
        mean_step,
        med_step,
        step_95,
        mad_step,
        std_step,
    ]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(
        eval_log[round, eval_run, :, 0]
    )

    if "InvA" in inputs["env_id"] or "Dice_SH_INSURED" in inputs["env_id"]:
        print(
            "{} Summary {:1.0f}/s l% {:1.0f} mean/med/95/mad/std: g% {:1.1f}/{:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} V$ {:1.1E}/{:1.1E}/{:1.1E}/{:1.0E} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
                datetime.now().strftime("%d %H:%M:%S"),
                steps_sec,
                stats[0],
                stats[1],
                stats[2],
                stats[3],
                stats[4],
                stats[5],
                stats[6],
                stats[7],
                stats[8],
                stats[9],
                stats[10],
                stats[11],
                stats[12],
                stats[13],
                stats[14],
            )
        )

    elif "InvB" in inputs["env_id"]:
        stop = eval_risk_log[round, eval_run, :, 4]
        mean_stop = np.mean(stop)

        print(
            "{} Summary {:1.0f}/s l%/s% {:1.0f}/{:1.0f} mean/med/95/mad/std: g% {:1.1f}/{:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} V$ {:1.1E}/{:1.1E}/{:1.1E}/{:1.0E} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
                datetime.now().strftime("%d %H:%M:%S"),
                steps_sec,
                stats[0],
                mean_stop * 100,
                stats[1],
                stats[2],
                stats[3],
                stats[4],
                stats[5],
                stats[6],
                stats[7],
                stats[8],
                stats[9],
                stats[10],
                stats[11],
                stats[12],
                stats[13],
                stats[14],
            )
        )
    else:
        stop = eval_risk_log[round, eval_run, :, 4]
        mean_stop = np.mean(stop)

        ret = eval_risk_log[round, eval_run, :, 5]
        mean_ret = np.mean(ret)

        print(
            "{} Summary {:1.0f}/s l%/s%/r% {:1.0f}/{:1.0f}/{:1.0f} mean/med/95/mad/std: g% {:1.1f}/{:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} V$ {:1.1E}/{:1.1E}/{:1.1E}/{:1.0E} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
                datetime.now().strftime("%d %H:%M:%S"),
                steps_sec,
                stats[0],
                mean_stop * 100,
                mean_ret * 100,
                stats[1],
                stats[2],
                stats[3],
                stats[4],
                stats[5],
                stats[6],
                stats[7],
                stats[8],
                stats[9],
                stats[10],
                stats[11],
                stats[12],
                stats[13],
                stats[14],
            )
        )


def eval_market(
    market_data: NDArrayFloat,
    obs_days: int,
    eval_start_idx: int,
    agent: object,
    inputs: dict,
    eval_log: NDArrayFloat,
    eval_risk_log: NDArrayFloat,
    multi_step: int,
    cum_steps: int,
    round: int,
    eval_run: int,
    loss: List[NDArrayFloat],
    logtemp: NDArrayFloat,
    loss_params: List[NDArrayFloat],
) -> None:
    """
    Evaluates agent policy on market environment without learning for a fixed
    number of episodes.

    Parameters:
        market_data: extracted time sliced data from complete time series
        obs_days: number of previous days agent uses for decision-making
        eval_start_idx: starting index for evaluation episodes without gap
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        eval_risk_log: array of exiting evalaution risk results
        multi_step: current bootstrapping
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    n_assets = market_data.shape[1]
    action_days = int(inputs["action_days"])
    test_length = int(inputs["test_days"] + obs_days - 1)

    print(
        "{} E{}_m{}_d{}_t{} {}-{}-{}-{} cst {} w/ {} evaluations: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            inputs["ENV_KEY"],
            multi_step,
            obs_days,
            action_days,
            inputs["algo"],
            inputs["s_dist"],
            inputs["loss_fn"],
            round + 1,
            cum_steps,
            int(inputs["n_eval"]),
            np.mean(loss[0:2]),
            np.mean(loss[4:6]),
            np.mean(loss[6:8]),
            np.mean(loss[8:10]),
            np.mean(loss_params[0:2]),
            np.mean(loss_params[2:4]),
            loss[8],
            np.exp(logtemp),
        )
    )

    if obs_days == 1:
        eval_env = eval(
            "market_envs.Market_"
            + inputs["env_id"][-10:-6]
            + "_D1"
            + "(n_assets, test_length, obs_days)"
        )
    else:
        eval_env = eval(
            "market_envs.Market_"
            + inputs["env_id"][-10:-6]
            + "_Dx"
            + "(n_assets, test_length, obs_days)"
        )

    gap = np.random.randint(
        int(inputs["gap_days_min"]),
        int(inputs["gap_days_max"]) + 1,
        size=int(inputs["n_eval"]),
    )
    gap += eval_start_idx

    for eval_epis in range(int(inputs["n_eval"])):
        start_time = time.perf_counter()

        eval_gap = gap[eval_epis]
        market_slice = market_data[eval_gap : eval_gap + test_length * action_days + 1]
        market_extract = env_resources.shuffle_data(
            market_slice, inputs["test_shuffle_days"]
        )

        time_step = 0
        obs_state = env_resources.observed_market_state(
            market_extract, time_step, action_days, obs_days
        )

        run_state = eval_env.reset(obs_state)
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            time_step += 1
            run_action = agent.eval_next_action(run_state)

            if cum_steps <= int(inputs["smoothing_window"]):
                run_action = utils.action_window(
                    run_action,
                    inputs["max_action"],
                    inputs["min_action"],
                    cum_steps,
                    int(inputs["smoothing_window"]),
                    int(inputs["random"]),
                )

            obs_state = env_resources.observed_market_state(
                market_extract, time_step, action_days, obs_days
            )

            run_next_state, eval_reward, done_flags, run_risk = eval_env.step(
                run_action, obs_state
            )

            run_done = done_flags[0]

            run_reward = eval_reward
            run_state = run_next_state
            run_step += 1

        end_time = time.perf_counter()

        eval_log[round, eval_run, eval_epis, 0] = end_time - start_time
        eval_log[round, eval_run, eval_epis, 1] = run_reward
        eval_log[round, eval_run, eval_epis, 2] = run_step
        eval_log[round, eval_run, eval_epis, 3:14] = loss
        eval_log[round, eval_run, eval_epis, 14] = logtemp
        eval_log[round, eval_run, eval_epis, 15:19] = loss_params
        eval_log[round, eval_run, eval_epis, 19] = cum_steps
        eval_risk_log[round, eval_run, eval_epis, 0] = eval_gap
        eval_risk_log[round, eval_run, eval_epis, 1:] = run_risk

    market_end = np.mean(gap + test_length * action_days + 1)

    reward = eval_log[round, eval_run, :, 1]
    mean_reward = np.mean(reward)
    med_reward = np.percentile(reward, q=50, method="median_unbiased")
    reward_95 = np.percentile(reward, q=5, method="median_unbiased")
    mad_reward = np.mean(np.abs(reward - mean_reward))
    std_reward = np.std(reward, ddof=0)

    val = eval_risk_log[round, eval_run, :, 1]
    mean_val = np.mean(val)
    med_val = np.percentile(val, q=50, method="median_unbiased")
    val_95 = np.percentile(val, q=5, method="median_unbiased")
    mad_val = np.mean(np.abs(val - mean_val))

    lev = eval_risk_log[round, eval_run, :, 3]
    mean_lev = np.mean(lev)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.percentile(step, q=50, method="median_unbiased")
    step_95 = np.percentile(step, q=5, method="median_unbiased")
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [
        (mean_reward - 1) * 100,
        (med_reward - 1) * 100,
        (reward_95 - 1) * 100,
        mad_reward * 100,
        std_reward * 100,
        mean_val,
        med_val,
        val_95,
        mad_val,
        mean_step,
        med_step,
        step_95,
        mad_step,
        std_step,
    ]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(
        eval_log[round, eval_run, :, 0]
    )

    print(
        "{} Summary {:1.0f}/s T {:1.0f} mean/med/95/mad/std: g% {:1.1f}/{:1.1f}/{:1.1f}/{:1.0f}/{:1.0f} V$ {:1.1E}/{:1.1E}/{:1.1E}/{:1.0E} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            steps_sec,
            market_end,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            stats[4],
            stats[5],
            stats[6],
            stats[7],
            stats[8],
            stats[9],
            stats[10],
            stats[11],
            stats[12],
            stats[13],
        )
    )


def eval_laminar(
    agent: object,
    inputs: dict,
    eval_log: NDArrayFloat,
    multi_step: int,
    cum_steps: int,
    round: int,
    eval_run: int,
    loss: List[NDArrayFloat],
    logtemp: NDArrayFloat,
    loss_params: List[NDArrayFloat],
) -> None:
    """
    Evaluates agent policy on guidance (laminar) environment without learning for a
    fixed number of episodes.

    Parameters:
        agent: RL agent algorithm
        inputs: dictionary containing all execution details
        eval_log: array of existing evalaution results
        multi_step: current bootstrapping
        cum_steps: current amount of cumulative steps
        round: current round of trials
        eval_run: current evaluation count
        loss: loss values of critic 1, critic 2 and actor
        logtemp: log entropy adjustment factor (temperature)
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
    """
    print(
        "{} E{}_m{} {}-{}-{}-{} cst {} w/ {} evaluations: C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            inputs["ENV_KEY"],
            multi_step,
            inputs["algo"],
            inputs["s_dist"],
            inputs["loss_fn"],
            round + 1,
            cum_steps,
            int(inputs["n_eval"]),
            np.mean(loss[0:2]),
            np.mean(loss[4:6]),
            np.mean(loss[6:8]),
            np.mean(loss[8:10]),
            np.mean(loss_params[0:2]),
            np.mean(loss_params[2:4]),
            loss[8],
            np.exp(logtemp),
        )
    )

    eval_env = eval(inputs["env_gym"])

    for eval_epis in range(int(inputs["n_eval"])):
        start_time = time.perf_counter()
        run_state = eval_env.reset()
        run_done, run_step, run_reward = False, 0, 0

        while not run_done:
            run_action = agent.eval_next_action(run_state)
            run_next_state, eval_reward, done_flags, _ = eval_env.step(run_action)

            run_done = done_flags[0]

            run_reward = eval_reward
            run_state = run_next_state
            run_step += 1

            if run_step == int(inputs["max_eval_steps"]):
                break

        end_time = time.perf_counter()

        eval_log[round, eval_run, eval_epis, 0] = end_time - start_time
        eval_log[round, eval_run, eval_epis, 1] = run_reward
        eval_log[round, eval_run, eval_epis, 2] = run_step
        eval_log[round, eval_run, eval_epis, 3:14] = loss
        eval_log[round, eval_run, eval_epis, 14] = logtemp
        eval_log[round, eval_run, eval_epis, 15:19] = loss_params
        eval_log[round, eval_run, eval_epis, 19] = cum_steps

    reward = eval_log[round, eval_run, :, 1]
    mean_reward = np.mean(reward)
    med_reward = np.percentile(reward, q=50, method="median_unbiased")
    reward_95 = np.percentile(reward, q=5, method="median_unbiased")
    mad_reward = np.mean(np.abs(reward - mean_reward))
    std_reward = np.std(reward, ddof=0)

    step = eval_log[round, eval_run, :, 2]
    mean_step = np.mean(step)
    med_step = np.percentile(step, q=50, method="median_unbiased")
    step_95 = np.percentile(step, q=5, method="median_unbiased")
    mad_step = np.mean(np.abs(step - mean_step))
    std_step = np.std(step, ddof=0)

    stats = [
        mean_reward * 100,
        med_reward * 100,
        reward_95 * 100,
        mad_reward * 100,
        std_reward * 100,
        mean_step,
        med_step,
        step_95,
        mad_step,
        std_step,
    ]

    steps_sec = np.sum(eval_log[round, eval_run, :, 2]) / np.sum(
        eval_log[round, eval_run, :, 0]
    )

    print(
        "{} Summary {:1.0f}/s mean/med/95/mad/std: r% {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f} st {:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}/{:1.0f}".format(
            datetime.now().strftime("%d %H:%M:%S"),
            steps_sec,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            stats[4],
            stats[5],
            stats[6],
            stats[7],
            stats[8],
            stats[9],
        )
    )
