"""
title:                  rl_multiplicative.py
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
    Responsible for performing agent training in multiplicative environments.
"""

import sys

sys.path.append("./")

import os
import time
from datetime import datetime

import numpy as np

import envs.coin_flip_envs as coin_flip_envs
import envs.dice_roll_envs as dice_roll_envs
import envs.dice_roll_sh_envs as dice_roll_sh_envs
import envs.gbm_envs as gbm_envs
import plotting.plots_summary as plots
import tests.test_live_learning as test_live_learning
import tools.eval_episodes as eval_episodes
import tools.utils as utils
from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3


def multiplicative_env(gym_envs: dict, inputs: dict, n_gambles: int) -> None:
    """
    Conduct experiments for multiplicative environments.

    Parameters:
        gym_envs: all environment details
        inputs: all training and evaluation details
        n_gambles: number of simultaneous identical gambles
    """
    n_gambles_str = "_n" + str(n_gambles)
    inputs = {"env_id": gym_envs[str(inputs["ENV_KEY"])][0] + n_gambles_str, **inputs}

    multi_key, sh_key, _, _, _, _, _ = utils.env_dynamics(gym_envs)

    if inputs["ENV_KEY"] <= multi_key + 2:
        env_gym = str(
            "coin_flip_envs." + gym_envs[str(inputs["ENV_KEY"])][0] + "(n_gambles)"
        )
    elif inputs["ENV_KEY"] <= multi_key + 5:
        env_gym = str(
            "dice_roll_envs." + gym_envs[str(inputs["ENV_KEY"])][0] + "(n_gambles)"
        )
    elif inputs["ENV_KEY"] <= multi_key + 8:
        env_gym = str("gbm_envs." + gym_envs[str(inputs["ENV_KEY"])][0] + "(n_gambles)")
    else:
        env_gym = str("dice_roll_sh_envs." + gym_envs[str(inputs["ENV_KEY"])][0] + "()")

    env = eval(env_gym)

    inputs: dict = {
        "input_dims": env.observation_space.shape,
        "num_actions": env.action_space.shape[0],
        "max_action": env.action_space.high.max(),
        "min_action": env.action_space.low.min(),  # assume all actions span equal domain
        "random": gym_envs[str(inputs["ENV_KEY"])][3],
        "dynamics": "M",  # gambling dynamics "M" (multiplicative)
        "n_trials": inputs["n_trials_mul"],
        "n_cumsteps": inputs["n_cumsteps_mul"],
        "trial": 0,
        "eval_freq": inputs["eval_freq_mul"],
        "n_eval": inputs["n_eval_mul"],
        "max_eval_steps": inputs["max_eval_steps_mul"],
        "smoothing_window": inputs["smoothing_window_mul"],
        "actor_percentile": inputs["actor_percentile_mul"],
        "critic_percentile": inputs["critic_percentile_mul"],
        "algo": "TD3",
        "s_dist": "N",
        "mini_batch_size": 1,
        "loss_fn": "MSE",
        "multi_steps": 1,
        "env_gym": env_gym,
        **inputs,
    }

    risk_dim = utils.multi_log_dim(inputs, n_gambles)

    if inputs["test_agent"]:
        if not os.path.exists("./results/test_multiplicative/data/" + inputs["env_id"]):
            os.makedirs("./results/test_multiplicative/data/" + inputs["env_id"])
    else:
        if not os.path.exists("./results/multiplicative/data/" + inputs["env_id"]):
            os.makedirs("./results/multiplicative/data/" + inputs["env_id"])

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
                        risk_dim,
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
                        state = env.reset()
                        done, step, score = False, 0, 0

                        while not done:

                            if cum_steps >= int(inputs["random"]):
                                action = agent.select_next_action(state)
                            else:
                                # take random actions during initial warmup period to generate new seed
                                action = env.action_space.sample()
                                action = (
                                    action
                                    if "GBM" in inputs["env_id"]
                                    else np.abs(action)
                                )

                            if cum_steps <= int(inputs["smoothing_window"]):
                                action = utils.action_window(
                                    action,
                                    inputs["max_action"],
                                    inputs["min_action"],
                                    cum_steps,
                                    int(inputs["smoothing_window"]),
                                    int(inputs["random"]),
                                )

                            next_state, reward, env_done, risk = env.step(action)

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

                                test_live_learning.critic_learning(
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
                                eval_episodes.eval_multiplicative(
                                    n_gambles,
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

                        if "InvA" in inputs["env_id"] or inputs["ENV_KEY"] == sh_key:
                            print(
                                "{} E{}_m{}_n{} {}-{}-{}-{} ep {} cst/st {}/{} {:1.0f}/s: l% {:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    n_gambles,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
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
                                "{} E{}_m{}_n{} {}-{}-{}-{} ep {} cst/st {}/{} {:1.0f}/s: l%/s% {:1.0f}/{:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    n_gambles,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
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
                                "{} E{}_m{}_n{} {}-{}-{}-{} ep {} cst/st {}/{} {:1.0f}/s: l%/s%/r% {:1.0f}/{:1.0f}/{:1.0f}, g%/V$ {:1.1f}/{:1.1E}, C/Cm/Cs {:1.2f}/{:1.2f}/{:1.2f}, a/c/k/A/T {:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}/{:1.2f}".format(
                                    datetime.now().strftime("%d %H:%M:%S"),
                                    inputs["ENV_KEY"],
                                    mstep,
                                    n_gambles,
                                    inputs["algo"],
                                    inputs["s_dist"],
                                    inputs["loss_fn"],
                                    round + 1,
                                    episode,
                                    cum_steps,
                                    step,
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
                        # ENV_KEY-multi_step-number_of_gambles,
                        # rl_algorithm-sampling_distribution-loss_function-trial,
                        # ep = episode,
                        # cst/st = cumulative_steps/steps,
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

                    if inputs["n_trials"] == 1:
                        plots.plot_learning_curve(
                            inputs, trial_log[round], directory + ".png"
                        )

                    round_end = time.perf_counter()
                    round_time = round_end - round_start

                    print(
                        "E{}_m{}_n{} Trial {} TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
                            inputs["ENV_KEY"],
                            mstep,
                            n_gambles,
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
