"""
title:                  rl_market.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Responsible for performing agent training in market environments.
"""

import sys

sys.path.append("./")

import os
import time
from datetime import datetime

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]

import envs.market_envs as market_envs
import tests.learning_tests as learning_tests
import tools.env_resources as env_resources
import tools.eval_episodes as eval_episodes
import tools.plots_summary as plots
import tools.utils as utils
from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3


def market_env(
    gym_envs: dict, inputs: dict, market_data: NDArrayFloat, obs_days: int
) -> None:
    """
    Conduct experiments for market environments.

    Parameters:
        gym_envs: all environment details
        inputs: all training and evaluation details
        market_data: extracted time sliced data from complete time series
        obs_days: number of previous days agent uses for decision-making
    """
    n_assets = market_data.shape[1]
    action_days = int(inputs["action_days"])
    train_length = int(inputs["train_days"] + obs_days - 1)
    test_length = int(inputs["test_days"] + obs_days - 1)
    gap_max = int(inputs["gap_days_max"])

    sample_length = int(
        action_days * (train_length + test_length) + obs_days + gap_max - 1
    )

    obs_days_str, action_days_str = "_D" + str(obs_days), "_T" + str(action_days)
    inputs = {
        "env_id": gym_envs[str(inputs["ENV_KEY"])][0] + obs_days_str + action_days_str,
        **inputs,
    }

    if obs_days == 1:
        env = eval(
            "market_envs.Market_"
            + gym_envs[str(inputs["ENV_KEY"])][0][-4:]
            + "_D1"
            + "(n_assets, train_length, obs_days)"
        )
    else:
        env = eval(
            "market_envs.Market_"
            + gym_envs[str(inputs["ENV_KEY"])][0][-4:]
            + "_Dx"
            + "(n_assets, train_length, obs_days)"
        )

    inputs: dict = {
        "input_dims": env.observation_space.shape,
        "num_actions": env.action_space.shape[0],
        "max_action": env.action_space.high.max(),
        "min_action": env.action_space.low.min(),  # assume all actions span equal domain
        "random": gym_envs[str(inputs["ENV_KEY"])][3],
        "dynamics": "MKT",  # gambling dynamics "MKT" (market)
        "n_trials": inputs["n_trials_mkt"],
        "n_cumsteps": inputs["n_cumsteps_mkt"],
        "trial": 0,
        "eval_freq": inputs["eval_freq_mkt"],
        "n_eval": inputs["n_eval_mkt"],
        "smoothing_window": inputs["smoothing_window_mkt"],
        "actor_percentile": inputs["actor_percentile_mkt"],
        "critic_percentile": inputs["critic_percentile_mkt"],
        "algo": "TD3",
        "s_dist": "N",
        "mini_batch_size": 1,
        "loss_fn": "MSE",
        "multi_steps": 1,
        **inputs,
    }

    risk_dim = utils.market_log_dim(inputs, n_assets)

    for algo in inputs["algo_name"]:

        inputs["s_dist"] = inputs["sample_dist"][algo]

        batch_size = int(inputs["batch_size"][algo])
        actor_batch = int(batch_size / inputs["actor_percentile"] * 100)
        critic_batch = int(batch_size / inputs["critic_percentile"] * 100)
        inputs["mini_batch_size"] = (
            actor_batch if actor_batch >= critic_batch else critic_batch
        )

        for loss_fn in inputs["critic_loss"]:
            for mstep in inputs["bootstraps"]:

                inputs["loss_fn"], inputs["algo"], inputs["multi_steps"] = (
                    loss_fn,
                    algo,
                    mstep,
                )

                trial_log = np.zeros(
                    (inputs["n_trials"], int(inputs["n_cumsteps"]), 19),
                    dtype=np.float32,
                )
                eval_log = np.zeros(
                    (
                        inputs["n_trials"],
                        int(inputs["n_cumsteps"] / inputs["eval_freq"]),
                        int(inputs["n_eval"]),
                        20,
                    ),
                    dtype=np.float32,
                )
                directory = utils.save_directory(inputs, results=True)

                trial_risk_log = np.zeros(
                    (inputs["n_trials"], int(inputs["n_cumsteps"]), risk_dim),
                    dtype=np.float32,
                )
                eval_risk_log = np.zeros(
                    (
                        inputs["n_trials"],
                        int(inputs["n_cumsteps"] / inputs["eval_freq"]),
                        int(inputs["n_eval"]),
                        risk_dim + 1,
                    ),
                    dtype=np.float32,
                )

                for round in range(inputs["n_trials"]):

                    round_start = time.perf_counter()

                    inputs["trial"] = round + 1

                    (
                        time_log,
                        score_log,
                        step_log,
                        logtemp_log,
                        loss_log,
                        loss_params_log,
                    ) = ([], [], [], [], [], [])
                    risk_log = []
                    cum_steps, eval_run, episode = 0, 0, 1
                    best_score = env.reward_range[0]

                    if round > 0 and inputs["continue"] == True:
                        # load existing SAC parameter to continue learning
                        inputs["initial_logtemp"] = logtemp

                    agent = (
                        Agent_td3(inputs)
                        if inputs["algo"] == "TD3"
                        else Agent_sac(inputs)
                    )
                    if round > 0 and inputs["continue"] == True:
                        # load existing actor-critic parameters to continue learning
                        agent.load_models()

                    while cum_steps < int(inputs["n_cumsteps"]):
                        start_time = time.perf_counter()

                        # randomly extract sequential time series from history and shuffle
                        market_slice, start_idx = env_resources.time_slice(
                            market_data, train_length, action_days, sample_length
                        )
                        market_extract = env_resources.shuffle_data(
                            market_slice, inputs["train_shuffle_days"]
                        )

                        time_step = 0
                        obs_state = env_resources.observed_market_state(
                            market_extract, time_step, action_days, obs_days
                        )

                        state = env.reset(obs_state)
                        done, step, score = False, 0, 0

                        while not done:
                            time_step += 1

                            if cum_steps >= int(inputs["random"]):
                                action = agent.select_next_action(state)
                            else:
                                # take random actions during initial warmup period to generate new seed
                                action = env.action_space.sample()

                            if cum_steps <= int(inputs["smoothing_window"]):
                                action = utils.action_window(
                                    action,
                                    inputs["max_action"],
                                    inputs["min_action"],
                                    cum_steps,
                                    int(inputs["smoothing_window"]),
                                    int(inputs["random"]),
                                )

                            obs_state = env_resources.observed_market_state(
                                market_extract, time_step, action_days, obs_days
                            )

                            next_state, reward, env_done, risk = env.step(
                                action, obs_state
                            )
                            # environment done flags
                            done, learn_done = env_done[0], env_done[1]

                            agent.store_transistion(
                                state, action, reward, next_state, learn_done
                            )

                            # gradient update interval (perform backpropagation)
                            if (
                                cum_steps % int(inputs["grad_step"][inputs["algo"]])
                                == 0
                            ):
                                loss, logtemp, loss_params = agent.learn()

                                learning_tests.critic_learning(
                                    cum_steps,
                                    inputs["mini_batch_size"],
                                    episode,
                                    step,
                                    loss,
                                    loss_params,
                                    logtemp,
                                    state,
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                    learn_done,
                                    risk,
                                )

                            state = next_state
                            score = reward
                            step += 1
                            cum_steps += 1
                            end_time = time.perf_counter()

                            # conduct periodic agent evaluation episodes without learning
                            if cum_steps % int(inputs["eval_freq"]) == 0:

                                loss[6:8] = utils.agent_shadow_mean(inputs, loss)
                                eval_start_idx = start_idx + step
                                eval_episodes.eval_market(
                                    market_data,
                                    obs_days,
                                    eval_start_idx,
                                    agent,
                                    inputs,
                                    eval_log,
                                    eval_risk_log,
                                    mstep,
                                    cum_steps,
                                    round,
                                    eval_run,
                                    loss,
                                    logtemp,
                                    loss_params,
                                )
                                eval_run += 1

                            if cum_steps >= int(inputs["n_cumsteps"]):
                                break

                        loss[6:8] = utils.agent_shadow_mean(inputs, loss)

                        time_log.append(end_time - start_time)
                        score_log.append(score)
                        step_log.append(step)
                        loss_log.append(loss)
                        logtemp_log.append(logtemp)
                        loss_params_log.append(loss_params)
                        risk_log.append(risk)

                        # save actor-critic neural network weights for checkpointing
                        trail_score = np.mean(score_log[-inputs["trail"] :])
                        if trail_score > best_score:
                            best_score = trail_score
                            agent.save_models()
                            print(
                                datetime.now().strftime("%d %H:%M:%S"),
                                "New high trailing score!",
                            )

                        if "InvA" in inputs["env_id"]:
                            print(
                                "{} E{}_m{}_d{}_t{} {}-{}-{}-{} ep {} cst/st {}/{} T {:1.0f} {:1.0f}/s: l% {:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    obs_days,
                                    action_days,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
                                    start_idx + time_step,
                                    step / time_log[-1],
                                    risk[3] * 100,
                                    (reward - 1) * 100,
                                    risk[1],
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

                        elif "InvB" in inputs["env_id"]:
                            print(
                                "{} E{}_m{}_d{}_t{} {}-{}-{}-{} ep {} cst/st {}/{} T {:1.0f} {:1.0f}/s: l%/s% {:1.0f}/{:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    obs_days,
                                    action_days,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
                                    start_idx + time_step,
                                    step / time_log[-1],
                                    risk[3] * 100,
                                    risk[4] * 100,
                                    (reward - 1) * 100,
                                    risk[1],
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

                        else:
                            print(
                                "{} E{}_m{}_d{}_t{} {}-{}-{}-{} ep {} cst/st {}/{} T {:1.0f} {:1.0f}/s: l%/s%/r% {:1.0f}/{:1.0f}/{:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    obs_days,
                                    action_days,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
                                    start_idx + time_step,
                                    step / time_log[-1],
                                    risk[3] * 100,
                                    risk[4] * 100,
                                    risk[5] * 100,
                                    (reward - 1) * 100,
                                    risk[1],
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

                        # EPISODE PRINT STATEMENT
                        # date,
                        # time,
                        # ENV_KEY-multi_step-observed_days_action_spacing,
                        # rl_algorithm-sampling_distribution-loss_function-trial,
                        # ep = episode,
                        # cst/st = cumulative_steps/steps,
                        # T_end = end_time,
                        # /s = training_steps_per_second,
                        # l%/s%/r% = leverage/stop-loss/retention,
                        # g%/V$ = time-average-growth-rate/valuation,
                        # C/Cm/Cs = avg_critic_loss/max_critic_loss/shadow_critic_loss,
                        # c/k/a/A/T = avg_Cauchy_scale/avg_CIM_kernel_size/avg_tail_exponent/avg_actor_loss/sac_entropy_temperature

                        episode += 1

                    count = len(score_log)

                    trial_log[round, :count, 0], trial_log[round, :count, 1] = (
                        time_log,
                        score_log,
                    )
                    trial_log[round, :count, 2], trial_log[round, :count, 3:14] = (
                        step_log,
                        loss_log,
                    )
                    trial_log[round, :count, 14], trial_log[round, :count, 15:] = (
                        logtemp_log,
                        loss_params_log,
                    )
                    trial_risk_log[round, :count, :] = risk_log

                    if not os.path.exists("./results/market/data/" + inputs["env_id"]):
                        os.makedirs("./results/market/data/" + inputs["env_id"])

                    if inputs["n_trials"] == 1:
                        plots.plot_learning_curve(
                            inputs, trial_log[round], directory + ".png"
                        )

                    round_end = time.perf_counter()
                    round_time = round_end - round_start

                    print(
                        "E{}_m{}_d{}_t{} Trial {} TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
                            inputs["ENV_KEY"],
                            mstep,
                            obs_days,
                            action_days,
                            round + 1,
                            round_time,
                            round_time / 60,
                            round_time / 3600,
                        )
                    )

                # truncate training trial log arrays up to maximum episodes
                count_episodes = [
                    np.min(np.where(trial_log[trial, :, 0] == 0))
                    for trial in range(int(inputs["n_trials"]))
                ]
                max_episode = np.max(count_episodes)
                trial_log, trial_risk_log = (
                    trial_log[:, :max_episode, :],
                    trial_risk_log[:, :max_episode, :],
                )

                np.save(directory + "_trial.npy", trial_log)
                np.save(directory + "_eval.npy", eval_log)
                np.save(directory + "_trial_risk.npy", trial_risk_log)
                np.save(directory + "_eval_risk.npy", eval_risk_log)

                if inputs["n_trials"] > 1:
                    # plot of agent evaluation round scores and training critic losses across all trials
                    plots.plot_eval_loss_2d(inputs, eval_log, directory + "_2d.png")

                    # plot of agent training with linear interpolation across all trials
                    plots.plot_trial_curve(inputs, trial_log, directory + "_trial.png")
