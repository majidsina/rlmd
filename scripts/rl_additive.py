"""
title:                  rl_additive.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://github.com/rajabinks

Description:
    Responsible for performing agent training in additive environments.
"""

import sys

sys.path.append("./")

import os
import time
from datetime import datetime

import gym
import numpy as np
import pybullet_envs

import plotting.plots_summary as plots
import tests.learning_tests as learning_tests
import tools.eval_episodes as eval_episodes
import tools.utils as utils
from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3


def additive_env(gym_envs: dict, inputs: dict) -> None:
    """
    Conduct experiments for additive environments.

    Parameters:
        gym_envs: all environment details
        inputs: all training and evaluation details
    """
    env = gym.make(gym_envs[str(inputs["ENV_KEY"])][0], disable_env_checker=False)

    inputs: dict = {
        "input_dims": env.observation_space.shape,
        "num_actions": env.action_space.shape[0],
        "max_action": env.action_space.high.max(),
        "min_action": env.action_space.low.min(),  # assume all actions span equal domain
        "env_id": gym_envs[str(inputs["ENV_KEY"])][0],
        "random": gym_envs[str(inputs["ENV_KEY"])][3],
        "dynamics": "A",  # gambling dynamics "A" (additive)
        "n_trials": inputs["n_trials_add"],
        "n_cumsteps": inputs["n_cumsteps_add"],
        "trial": 0,
        "eval_freq": inputs["eval_freq_add"],
        "n_eval": inputs["n_eval_add"],
        "actor_percentile": inputs["actor_percentile_add"],
        "critic_percentile": inputs["critic_percentile_add"],
        "algo": "TD3",
        "s_dist": "N",
        "mini_batch_size": 1,
        "loss_fn": "MSE",
        "multi_steps": 1,
        **inputs,
    }

    # allow access to setting enviroment state and remove episode step limit
    env = env.env

    if inputs["test_agent"]:
        if not os.path.exists("./results/additive-test/data/" + inputs["env_id"]):
            os.makedirs("./results/additive-test/data/" + inputs["env_id"])
    else:
        if not os.path.exists("./results/additive/data/" + inputs["env_id"]):
            os.makedirs("./results/additive/data/" + inputs["env_id"])

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

                            next_state, reward, done, info = env.step(action)
                            agent.store_transistion(
                                state, action, reward, next_state, done
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
                                )

                            state = next_state
                            score += reward
                            step += 1
                            cum_steps += 1
                            end_time = time.perf_counter()

                            # conduct periodic agent evaluation episodes without learning
                            if cum_steps % int(inputs["eval_freq"]) == 0:

                                loss[6:8] = utils.agent_shadow_mean(inputs, loss)
                                eval_episodes.eval_additive(
                                    agent,
                                    inputs,
                                    eval_log,
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

                        # save actor-critic neural network weights for checkpointing
                        trail_score = np.mean(score_log[-inputs["trail"] :])
                        if trail_score > best_score:
                            best_score = trail_score
                            agent.save_models()
                            print(
                                datetime.now().strftime("%d %H:%M:%S"),
                                "New high trailing score!",
                            )

                        print(
                            "{} E{}_m{} {}-{}-{}-{} ep {} cst/st {}/{} {:1.0f}/s: r {:1.0f}, tr{} {:1.0f}, C/Cm/Cs {:1.1f}/{:1.1f}/{:1.0f}, a/c/k {:1.2f}/{:1.2f}/{:1.2f}, A/T {:1.1f}/{:1.2f}".format(
                                datetime.now().strftime("%d %H:%M:%S"),
                                inputs["ENV_KEY"],
                                mstep,
                                inputs["algo"],
                                inputs["s_dist"],
                                inputs["loss_fn"],
                                round + 1,
                                episode,
                                cum_steps,
                                step,
                                step / time_log[-1],
                                score,
                                inputs["trail"],
                                trail_score,
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
                        # ENV_KEY-multi_step,
                        # rl_algorithm-sampling_distribution-loss_function-trial,
                        # ep = episode,
                        # cst/st = cumulative_steps/steps,
                        # /s = training_steps_per_second,
                        # r = episode_reward,
                        # tr = trailing_episode_reward,
                        # C/Cm/Cs = avg_critic_loss/max_critic_loss/shadow_critic_loss,
                        # c/k/a = avg_Cauchy_scale/avg_CIM_kernel_size/avg_tail_exponent,
                        # A/T = avg_actor_loss/sac_entropy_temperature

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

                    if inputs["n_trials"] == 1:
                        plots.plot_learning_curve(
                            inputs, trial_log[round], directory + ".png"
                        )

                    round_end = time.perf_counter()
                    round_time = round_end - round_start

                    print(
                        "E{}_m{} Trial {} TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
                            inputs["ENV_KEY"],
                            mstep,
                            round + 1,
                            round_time,
                            round_time / 60,
                            round_time / 3600,
                        )
                    )

                # truncate training trial log array up to maximum episodes
                count_episodes = [
                    np.min(np.where(trial_log[trial, :, 0] == 0))
                    for trial in range(int(inputs["n_trials"]))
                ]
                max_episode = np.max(count_episodes)
                trial_log = trial_log[:, :max_episode, :]

                np.save(directory + "_trial.npy", trial_log)
                np.save(directory + "_eval.npy", eval_log)

                if inputs["n_trials"] > 1:
                    # plot of agent evaluation round scores across all trials
                    # plots.plot_eval_curve(inputs, eval_log, directory +"_eval.png")

                    # plot of agent evaluation round scores and training critic losses across all trials
                    plots.plot_eval_loss_2d(inputs, eval_log, directory + "_2d.png")

                    # 3D plot of agent evaluation round scores and training critic losses across all trials
                    # plots.plot_eval_loss_3d(inputs, eval_log, directory+"_3d.png")

                    # plot of agent training with linear interpolation across all trials
                    plots.plot_trial_curve(inputs, trial_log, directory + "_trial.png")
