"""
title:                  test_input_agent.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Responsible for conducting tests on all user inputs for agent training.
"""

import sys

sys.path.append("./")

import os
from os import PathLike
from typing import Dict, List, Union

import numpy as np

from algos.algo_sac import Agent_sac
from algos.algo_td3 import Agent_td3
from algos.networks_sac import ActorNetwork as ActorNetwork_sac
from algos.networks_sac import CriticNetwork as CriticNetwork_sac
from algos.networks_td3 import ActorNetwork as ActorNetwork_td3
from algos.networks_td3 import CriticNetwork as CriticNetwork_td3
from tools.replay import ReplayBuffer
from tools.replay_torch import ReplayBufferTorch
from tools.utils import env_dynamics

# default assertion errors
tb: str = "variable must be of type bool"
td: str = "variable must be of type dict"
tf: str = "variable must be of type float"
tfi: str = "variable must be of type float for int"
ti: str = "variable must be of type int"
tl: str = "variable must be of type list"
ts: str = "variable must be of type str"
gte0: str = "quantity must be greater than or equal to 0"
gt0: str = "quantity must be greater than 0"
gte1: str = "quantity must be greater than or equal to 1"


def learning_tests(inputs: dict) -> None:
    """
    Conduct tests on all agent algorithm and training parameters.

    Parameters:
        inputs: all training and evaluation details
    """
    assert isinstance(inputs, dict), td

    # training input tests
    assert isinstance(inputs["test_agent"], bool), tb
    assert isinstance(inputs["algo_name"], list), tl
    assert set(inputs["algo_name"]).issubset(
        set(["SAC", "TD3"])
    ), "algo_name must be a list containing only 'SAC' and/or 'TD3'"
    assert (
        1 <= len(inputs["algo_name"]) <= 2
    ), "only upto two possible algorithms selectable"
    assert len(inputs["algo_name"]) == len(
        set(inputs["algo_name"])
    ), "algo_name must contain only unique elements"
    assert isinstance(inputs["critic_loss"], list), tl
    assert set(inputs["critic_loss"]).issubset(
        set(["MSE", "HUB", "MAE", "HSC", "CAU", "TCAU", "CIM", "MSE2", "MSE4", "MSE6"])
    ), "critic_loss must be a list containing 'MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', and/or 'MSE6'"
    assert (
        1 <= len(inputs["critic_loss"]) <= 10
    ), "only ten possible critic_loss functions selectable"
    assert len(inputs["critic_loss"]) == len(
        set(inputs["critic_loss"])
    ), "critic_loss must contain only unique elements"
    assert isinstance(inputs["bootstraps"], list), tl
    assert all(isinstance(mstep, int) for mstep in inputs["bootstraps"]), ti
    assert (
        len(inputs["bootstraps"]) >= 1
    ), "bootstraps (multi-steps) must have at least one multi-step"
    assert len(inputs["bootstraps"]) == len(
        set(inputs["bootstraps"])
    ), "bootstraps (multi-steps) must contain only unique elements"
    assert all(
        mstep >= 1 for mstep in inputs["bootstraps"]
    ), "bootstraps (multi-steps) must be a list of (non-zero) positive integers"

    # additive environment execution tests
    assert isinstance(inputs["n_trials_add"], (float, int)), tfi
    assert int(inputs["n_trials_add"]) >= 1, gte1
    assert isinstance(inputs["n_cumsteps_add"], (float, int)), tfi
    assert set(list(str(inputs["n_cumsteps_add"])[2:])).issubset(
        set(["0", "."])
    ), "n_cumsteps_add must consist of only 2 leading non-zero digits"
    assert int(inputs["n_cumsteps_add"]) >= 1, gte1
    assert isinstance(inputs["eval_freq_add"], (float, int)), tfi
    assert int(inputs["eval_freq_add"]) >= 1, gte1
    assert int(inputs["eval_freq_add"]) <= int(
        inputs["n_cumsteps_add"]
    ), "eval_freq_add must be less than or equal to n_cumsteps_add"
    assert isinstance(inputs["n_eval_add"], (float, int)), tfi
    assert int(inputs["n_eval_add"]) >= 1, gte1
    assert isinstance(inputs["max_eval_reward"], (float, int)), tfi
    assert inputs["max_eval_reward"] > 0, gt0
    assert isinstance(inputs["actor_percentile_add"], (float, int)), tfi
    assert (
        0 < inputs["actor_percentile_add"] <= 100
    ), "actor_percentile_add must be within (0, 100] interval"
    assert isinstance(inputs["critic_percentile_add"], (float, int)), tfi
    assert (
        0 < inputs["critic_percentile_add"] <= 100
    ), "critic_percentile_add must be within (0, 100] interval"

    # multiplicative environment execution tests
    assert isinstance(inputs["n_trials_mul"], (float, int)), tfi
    assert int(inputs["n_trials_mul"]) >= 1, gte1
    assert isinstance(inputs["n_cumsteps_mul"], (float, int)), tfi
    assert set(list(str(inputs["n_cumsteps_mul"])[2:])).issubset(
        set(["0", "."])
    ), "n_cumsteps_mul must consist of only 2 leading non-zero digits"
    assert int(inputs["n_cumsteps_mul"]) >= 1, gte1
    assert isinstance(inputs["eval_freq_mul"], (float, int)), tfi
    assert int(inputs["eval_freq_mul"]) >= 1, gte1
    assert int(inputs["eval_freq_mul"]) <= int(
        inputs["n_cumsteps_mul"]
    ), "eval_freq_mul must be less than or equal to n_cumsteps_mul"
    assert isinstance(inputs["n_eval_mul"], (float, int)), tfi
    assert int(inputs["n_eval_mul"]) >= 1, gte1
    assert isinstance(inputs["max_eval_steps_mul"], (float, int)), tfi
    assert int(inputs["max_eval_steps_mul"]) >= 1, gte1
    assert isinstance(inputs["smoothing_window_mul"], (float, int)), tfi
    assert int(inputs["smoothing_window_mul"]) >= 0, gte0
    assert isinstance(inputs["actor_percentile_mul"], (float, int)), tfi
    assert (
        0 < inputs["actor_percentile_mul"] <= 100
    ), "actor_percentile_mul must be within (0, 100] interval"
    assert isinstance(inputs["critic_percentile_mul"], (float, int)), tfi
    assert (
        0 < inputs["critic_percentile_mul"] <= 100
    ), "critic_percentile_mul must be within (0, 100] interval"

    assert isinstance(inputs["n_gambles"], list), tl
    assert all(isinstance(gambles, int) for gambles in inputs["n_gambles"]), ti
    assert (
        len(inputs["n_gambles"]) >= 1
    ), "n_gambles (number of gambles) must have at least one count of gambles"
    assert len(inputs["n_gambles"]) == len(
        set(inputs["n_gambles"])
    ), "n_gambles (number of gambles) must contain only unique elements"
    assert all(
        gambles >= 1 for gambles in inputs["n_gambles"]
    ), "n_gambles (number of gambles) must be a list of positive (non-zero) integers"

    # market environment execution tests
    assert isinstance(inputs["market_dir"], Union[str, bytes, PathLike]), ts
    assert (
        inputs["market_dir"][0:2] == "./" and inputs["market_dir"][-1] == "/"
    ), "market_dir file path must be in a sub-directory relative to main.py"

    assert isinstance(inputs["n_trials_mkt"], (float, int)), tfi
    assert int(inputs["n_trials_mkt"]) >= 1, gte1
    assert isinstance(inputs["n_cumsteps_mkt"], (float, int)), tfi
    assert set(list(str(inputs["n_cumsteps_mkt"])[2:])).issubset(
        set(["0", "."])
    ), "n_cumsteps_mkt must consist of only 2 leading non-zero digits"
    assert int(inputs["n_cumsteps_mkt"]) >= 1, gte1
    assert isinstance(inputs["eval_freq_mkt"], (float, int)), tfi
    assert int(inputs["eval_freq_mkt"]) >= 1
    assert int(inputs["eval_freq_mkt"]) <= int(
        inputs["n_cumsteps_mkt"]
    ), "eval_freq_mkt must be less than or equal to n_cumsteps_mkt"
    assert isinstance(inputs["n_eval_mkt"], (float, int)), tfi
    assert int(inputs["n_eval_mkt"]) >= 1, gte1
    assert isinstance(inputs["smoothing_window_mkt"], (float, int)), tfi
    assert int(inputs["smoothing_window_mkt"]) >= 0, gte0
    assert isinstance(inputs["actor_percentile_mkt"], (float, int)), tfi
    assert (
        0 < inputs["actor_percentile_mkt"] <= 100
    ), "actor_percentile_mkt must be within (0, 100] interval"
    assert isinstance(inputs["critic_percentile_mkt"], (float, int)), tfi
    assert (
        0 < inputs["critic_percentile_mkt"] <= 100
    ), "critic_percentile_mkt must be within (0, 100] interval"

    assert isinstance(inputs["action_days"], (float, int)), tfi
    assert int(inputs["action_days"]) >= 1, gte1
    assert isinstance(inputs["train_days"], (float, int)), tfi
    assert int(inputs["train_days"]) > 0, gt0
    assert isinstance(inputs["test_days"], (float, int)), tfi
    assert int(inputs["test_days"]) > 0, gt0
    assert isinstance(inputs["train_shuffle_days"], int), ti
    assert int(inputs["train_shuffle_days"]) >= 1, gte1
    assert isinstance(inputs["test_shuffle_days"], int), ti
    assert int(inputs["test_shuffle_days"]) >= 1, gte1
    assert inputs["train_shuffle_days"] <= int(
        inputs["train_days"]
    ), "train_shuffle_days must be less than or equal to train_years"
    assert inputs["test_shuffle_days"] <= int(
        inputs["test_days"]
    ), "test_shuffle_days must be less than or equal to test_years"
    assert isinstance(inputs["gap_days_min"], int), ti
    assert int(inputs["gap_days_min"]) >= 0, gte0
    assert isinstance(inputs["gap_days_max"], int), ti
    assert int(inputs["gap_days_max"]) >= 0, gte0
    assert int(inputs["gap_days_min"]) <= int(
        inputs["gap_days_max"]
    ), "gap_days_max must be greater than or equal to gap_days_min"

    assert isinstance(inputs["past_days"], list), tl
    assert all(isinstance(days, int) for days in inputs["past_days"]), ti
    assert (
        len(inputs["past_days"]) >= 1
    ), "past_days (observed days) must have at least one count of days"
    assert len(inputs["past_days"]) == len(
        set(inputs["past_days"])
    ), "past_days (observed days) must contain only unique elements"
    assert all(
        days >= 1 for days in inputs["past_days"]
    ), "past_days (observed days) must be a list of (non-zero) positive integers"

    # guidance environment execution tests
    assert isinstance(inputs["n_trials_gud"], (float, int)), tfi
    assert int(inputs["n_trials_gud"]) >= 1, gte1
    assert isinstance(inputs["n_cumsteps_gud"], (float, int)), tfi
    assert set(list(str(inputs["n_cumsteps_gud"])[2:])).issubset(
        set(["0", "."])
    ), "n_cumsteps_gud must consist of only 2 leading non-zero digits"
    assert int(inputs["n_cumsteps_gud"]) >= 1, gte1
    assert isinstance(inputs["eval_freq_gud"], (float, int)), tfi
    assert int(inputs["eval_freq_gud"]) >= 1, gte1
    assert int(inputs["eval_freq_gud"]) <= int(
        inputs["n_cumsteps_gud"]
    ), "eval_freq_gud must be less than or equal to n_cumsteps_gud"
    assert isinstance(inputs["n_eval_gud"], (float, int)), tfi
    assert int(inputs["n_eval_gud"]) >= 1, gte1
    assert isinstance(inputs["max_eval_steps_gud"], (float, int)), tfi
    assert int(inputs["max_eval_steps_gud"]) >= 1, gte1
    assert isinstance(inputs["actor_percentile_gud"], (float, int)), tfi
    assert (
        0 < inputs["actor_percentile_gud"] <= 100
    ), "actor_percentile_gud must be within (0, 100] interval"
    assert isinstance(inputs["critic_percentile_gud"], (float, int)), tfi
    assert (
        0 < inputs["critic_percentile_gud"] <= 100
    ), "critic_percentile_gud must be within (0, 100] interval"

    assert isinstance(inputs["targets"], list), tl
    assert all(isinstance(targets, int) for targets in inputs["targets"]), ti
    assert (
        len(inputs["targets"]) >= 1
    ), "targets must have at least one number of past days"
    assert len(inputs["targets"]) == len(
        set(inputs["targets"])
    ), "targets must contain only unique elements"
    assert all(
        targets >= 1 for targets in inputs["targets"]
    ), "targets must be a list of (non-zero) positive integers"

    # learning varaible tests
    assert isinstance(inputs["gpu"], str), ts
    if inputs["gpu"] == "cpu":
        pass
    else:
        assert inputs["gpu"][0:5] == "cuda:"
        assert isinstance(int(inputs["gpu"][-1]), int), ti
        assert int(inputs["gpu"][-1]) >= 0, gte0
    assert isinstance(inputs["buffer_gpu"], bool), tb
    assert isinstance(inputs["buffer"], (float, int)), tfi
    assert set(list(str(inputs["buffer"])[2:])).issubset(
        set(["0", "."])
    ), "buffer must consist of only 2 leading non-zero digits"

    assert int(inputs["buffer"]) >= 1, gte1
    assert (
        inputs["buffer"] >= inputs["n_cumsteps_add"]
    ), "buffer must be greater than or equal to n_cumsteps_add training steps"
    assert (
        inputs["buffer"] >= inputs["n_cumsteps_mul"]
    ), "buffer must be greater than or equal to n_cumsteps_mul training steps"
    assert inputs["buffer"] >= int(
        inputs["n_cumsteps_mkt"]
    ), "buffer must be greater than or equal to n_cumsteps_mkt training steps"
    assert inputs["buffer"] >= int(
        inputs["n_cumsteps_gud"]
    ), "buffer must be greater than or equal to n_cumsteps_gud training steps"
    assert (
        inputs["discount"] >= 0 and inputs["discount"] < 1
    ), "discount must be within [0, 1) interval"
    assert isinstance(inputs["trail"], (float, int)), tfi
    assert int(inputs["trail"]) >= 1, gte1
    assert isinstance(inputs["cauchy_scale"], (float, int)), tfi
    assert inputs["cauchy_scale"] > 0, gt0
    assert (
        isinstance(inputs["r_abs_zero"], (float, int)) or inputs["r_abs_zero"] == None
    ), "r_abs_zero must be either real number or None"
    assert isinstance(inputs["continue"], bool), tb

    # critic loss aggregation tests
    assert (
        inputs["critic_mean_type"] == "E"
    ), "critic_mean_type must be 'E' ('S' not currently possible)"
    assert isinstance(inputs["shadow_low_mul"], (float, int)), tfi
    assert inputs["shadow_low_mul"] >= 0, gte0
    assert isinstance(inputs["shadow_high_mul"], (float, int)), tfi
    assert inputs["shadow_high_mul"] > 0, gt0

    # SAC hyperparameter tests
    assert isinstance(inputs["sac_actor_learn_rate"], (float, int)), tfi
    assert inputs["sac_actor_learn_rate"] > 0, gt0
    assert isinstance(inputs["sac_critic_learn_rate"], (float, int)), tfi
    assert inputs["sac_critic_learn_rate"] > 0, gt0
    assert isinstance(inputs["sac_temp_learn_rate"], (float, int)), tfi
    assert inputs["sac_temp_learn_rate"] > 0, gt0
    assert isinstance(inputs["sac_layer_1_units"], (float, int)), tfi
    assert int(inputs["sac_layer_1_units"]) >= 1, gte1
    assert isinstance(inputs["sac_layer_2_units"], (float, int)), tfi
    assert int(inputs["sac_layer_2_units"]) >= 1, gte1
    assert isinstance(inputs["sac_actor_step_update"], (float, int)), tfi
    assert int(inputs["sac_actor_step_update"]) >= 1, gte1
    assert isinstance(inputs["sac_temp_step_update"], (float, int)), tfi
    assert int(inputs["sac_temp_step_update"]) >= 1, gte1
    assert isinstance(inputs["sac_target_critic_update"], (float, int)), tfi
    assert int(inputs["sac_target_critic_update"]) >= 1, gte1
    assert isinstance(inputs["sac_target_update_rate"], (float, int)), tfi
    assert inputs["sac_target_update_rate"] > 0, gt0

    assert isinstance(inputs["initial_logtemp"], (float, int)), tfi
    assert isinstance(inputs["log_scale_min"], (float, int)), tfi
    assert isinstance(inputs["log_scale_max"], (float, int)), tfi
    assert (
        inputs["log_scale_min"] < inputs["log_scale_max"]
    ), "SAC scale limits must be valid"
    assert isinstance(inputs["reparam_noise"], float), tf
    assert (
        inputs["reparam_noise"] > 1e-7 and inputs["reparam_noise"] < 1e-5
    ), "SAC reparam_noise must be a real number in the vicinity of 1e-6"

    # TD3 hyperparameter tests
    assert isinstance(inputs["td3_actor_learn_rate"], (float, int)), tfi
    assert inputs["td3_actor_learn_rate"] > 0, gt0
    assert isinstance(inputs["td3_critic_learn_rate"], (float, int)), tfi
    assert inputs["td3_critic_learn_rate"] > 0, gt0
    assert isinstance(inputs["td3_layer_1_units"], (float, int)), tfi
    assert int(inputs["td3_layer_1_units"]) >= 1, gte1
    assert isinstance(inputs["td3_layer_2_units"], (float, int)), tfi
    assert int(inputs["td3_layer_2_units"]) >= 1, gte1
    assert isinstance(inputs["td3_actor_step_update"], (float, int)), tfi
    assert int(inputs["td3_actor_step_update"]) >= 1, gte1
    assert isinstance(inputs["td3_target_actor_update"], (float, int)), tfi
    assert int(inputs["td3_target_actor_update"]) >= 1, gte1
    assert isinstance(inputs["td3_target_critic_update"], (float, int)), tfi
    assert int(inputs["td3_target_critic_update"]) >= 1, gte1
    assert isinstance(inputs["td3_target_critic_update"], (float, int)), tfi
    assert int(inputs["td3_target_critic_update"]) >= 1, gte1
    assert isinstance(inputs["td3_target_update_rate"], (float, int)), tfi
    assert inputs["td3_target_update_rate"] > 0, gt0

    assert isinstance(inputs["policy_noise"], (float, int)), tfi
    assert inputs["policy_noise"] > 0, gt0
    assert isinstance(inputs["target_policy_noise"], (float, int)), tfi
    assert inputs["target_policy_noise"] > 0, gt0
    assert isinstance(inputs["target_policy_clip"], (float, int)), tfi
    assert inputs["target_policy_clip"] > 0, gt0

    # shared algorithm training parameter tests
    assert isinstance(inputs["sample_dist"], dict), td
    assert set(inputs["sample_dist"].keys()).issubset(
        set(["SAC", "TD3"])
    ), "must contain the two main algorithms"
    assert (
        inputs["sample_dist"]["SAC"] == "N"
        or inputs["sample_dist"]["SAC"] == "L"
        or inputs["sample_dist"]["SAC"] == "MVN"
    ), "SAC sample_dist must be either 'N' (normal = Gaussian) or 'L' (2x exponential = Laplace), or 'MVN' (multi-variate normal)"
    assert isinstance(inputs["sample_dist"]["TD3"], str), ts
    assert (
        inputs["sample_dist"]["TD3"] == "N" or inputs["sample_dist"]["TD3"] == "L"
    ), "TD3 sample_dist must be either 'N' (normal = Gaussian) or 'L' (2x exponential = Laplace)"
    assert isinstance(inputs["batch_size"], dict), td
    assert set(inputs["batch_size"].keys()).issubset(
        set(["SAC", "TD3"])
    ), "must contain the two main algorithms"
    assert isinstance(inputs["batch_size"]["TD3"], (float, int)), tfi
    assert int(inputs["batch_size"]["TD3"]) >= 1, gte1
    assert isinstance(inputs["batch_size"]["SAC"], (float, int)), tfi
    assert int(inputs["batch_size"]["SAC"]) >= 1, gte1
    assert isinstance(inputs["grad_step"], dict), td
    assert set(inputs["grad_step"].keys()).issubset(
        set(["SAC", "TD3"])
    ), "must contain the two main algorithms"
    assert isinstance(inputs["grad_step"]["TD3"], (float, int)), tfi
    assert int(inputs["grad_step"]["TD3"]) >= 1, gte1
    assert isinstance(inputs["grad_step"]["SAC"], (float, int)), tfi
    assert int(inputs["grad_step"]["SAC"]) >= 1, gte1
    assert isinstance(inputs["log_noise"], float), tf
    assert (
        inputs["log_noise"] > 1e-7 and inputs["log_noise"] < 1e-5
    ), "log_noise must be a real number in the vicinity of 1e-6"

    print("Learning Parameter Tests: Passed")


def env_tests(gym_envs: Dict[str, list], inputs: dict) -> None:
    """
    Conduct tests on details of all selected environments.

    Parameters:
        gym_envs: all environment details
        inputs: all training and evaluation details
    """
    assert isinstance(gym_envs, dict), td

    assert all(
        isinstance(int(env), int) for env in gym_envs
    ), "all environment keys must be strings that are convertible to integers"
    keys: List[int] = [int(env) for env in gym_envs]

    assert isinstance(inputs["envs"], list), tl
    assert set(inputs["envs"]).issubset(
        set(keys)
    ), "environments must be selected from gym_envs dict keys"

    assert inputs["ENV_KEY"] == None

    # obtain environment key limits based on reward dynamics
    multi_key, _, market_key, market_env_keys, gud_key, _, _ = env_dynamics(gym_envs)

    for key in inputs["envs"]:

        assert isinstance(gym_envs[str(key)], list), tl
        assert (
            len(gym_envs[str(key)]) == 4
        ), "environment {} list musst be of length 4".format(key)
        assert isinstance(gym_envs[str(key)][0], str), ts
        assert all(
            isinstance(x, (float, int)) for x in gym_envs[str(key)][1:]
        ), "environment {} details must be a list of the form [string, int>0, int>0, real>0]".format(
            key
        )
        assert (
            int(gym_envs[str(key)][1]) >= 1
        ), "environment {} must have at least one state".format(key)
        assert (
            int(gym_envs[str(key)][2]) >= 1
        ), "environment {} must have at least one action".format(key)

        if key < multi_key:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) < int(
                inputs["n_cumsteps_add"]
            ), "environment {}: warm-up must be less than total training steps".format(
                key
            )
            assert int(2 * inputs["eval_freq_add"]) <= int(
                inputs["n_cumsteps_add"]
            ), "environment {}: 2x evaluation frequency must be less than or equal to total training steps".format(
                key
            )

        elif key < market_key:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) < int(
                inputs["n_cumsteps_mul"]
            ), "environment {}: warm-up must be less than total training steps".format(
                key
            )
            assert int(2 * inputs["eval_freq_mul"]) <= int(
                inputs["n_cumsteps_mul"]
            ), "environment {}: 2x evaluation frequency must be less than or equal to total training steps".format(
                key
            )

        elif key < gud_key:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) < int(
                inputs["n_cumsteps_mkt"]
            ), "environment {}: warm-up must be less than total training steps".format(
                key
            )
            assert int(2 * inputs["eval_freq_mkt"]) <= int(
                inputs["n_cumsteps_mkt"]
            ), "environment {}: 2x evaluation frequency must be less than or equal to total training steps".format(
                key
            )

        else:
            assert int(gym_envs[str(key)][3]) >= 0, gte0
            assert int(gym_envs[str(key)][3]) < int(
                inputs["n_cumsteps_gud"]
            ), "environment {}: warm-up must be less than total training steps".format(
                key
            )
            assert int(2 * inputs["eval_freq_gud"]) <= int(
                inputs["n_cumsteps_gud"]
            ), "environment {}: 2x evaluation frequency must be less than or equal to total training steps".format(
                key
            )

    # market environment data integrity checks
    if any(market_key <= key < gud_key for key in inputs["envs"]):
        for key in inputs["envs"]:
            if market_key <= key < gud_key:

                if key <= market_env_keys[0]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_snp.npy"
                    ), "stooq_snp.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_snp.npy")

                elif key <= market_env_keys[1]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_usei.npy"
                    ), "stooq_usei.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_usei.npy")

                elif key <= market_env_keys[2]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_minor.npy"
                    ), "stooq_minor.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_minor.npy")

                elif key <= market_env_keys[3]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_medium.npy"
                    ), "stooq_medium.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_medium.npy")

                elif key <= market_env_keys[4]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_major.npy"
                    ), "stooq_major.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_major.npy")

                elif key <= market_env_keys[5]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_dji.npy"
                    ), "stooq_dji.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_dji.npy")

                elif key <= market_env_keys[6]:
                    assert os.path.isfile(
                        inputs["market_dir"] + "stooq_full.npy"
                    ), "stooq_full.npy not generated or found in {}".format(
                        inputs["market_dir"]
                    )
                    data = np.load(inputs["market_dir"] + "stooq_full.npy")

                time_length = data.shape[0]

                for days in inputs["past_days"]:
                    train_length = int(inputs["train_days"])
                    test_length = int(inputs["test_days"])
                    gap_max = int(inputs["gap_days_max"])
                    action_days = int(inputs["action_days"])

                    sample_length = int(
                        action_days * (train_length + test_length) + gap_max + days - 1
                    )

                    assert (
                        time_length >= sample_length
                    ), "ENV_KEY {}: total time {} period with {} day(s) observed and {} day(s) action spacing must be greater than sample length = {}".format(
                        key, time_length, days, action_days, sample_length
                    )

    print("Gym Environment Tests: Passed")


def algo_method_checks(inputs: dict) -> None:
    """
    Confirm the presence of numerous class methods required for agent learning.

    Parameters:
        inputs: all training and evaluation details
    """
    # SAC algorithm method checks
    if "SAC" in inputs["algo_name"]:
        assert hasattr(
            ActorNetwork_sac, "forward"
        ), "missing SAC actor forward propagation"
        assert hasattr(
            ActorNetwork_sac, "stochastic_uv_gaussian"
        ), "missing SAC univariate Gaussian sampling"
        assert hasattr(
            ActorNetwork_sac, "stochastic_uv_laplace"
        ), "missing SAC univariate Laplace sampling"
        assert hasattr(
            ActorNetwork_sac, "stochastic_mv_gaussian"
        ), "missing SAC multi-variate Gaussian sampling"
        assert hasattr(
            ActorNetwork_sac, "deterministic_policy"
        ), "missing SAC deterministic sampling"
        assert hasattr(
            ActorNetwork_sac, "save_checkpoint"
        ), "missing SAC actor saving functionality"
        assert hasattr(
            ActorNetwork_sac, "load_checkpoint"
        ), "missing SAC actor load functionality"
        assert hasattr(
            CriticNetwork_sac, "forward"
        ), "missing SAC critic forward propagation"
        assert hasattr(
            CriticNetwork_sac, "save_checkpoint"
        ), "missing SAC critic saving functionality"
        assert hasattr(
            CriticNetwork_sac, "load_checkpoint"
        ), "missing SAC critic load functionality"
        assert hasattr(
            Agent_sac, "store_transistion"
        ), "missing SAC transition storage functionality"
        assert hasattr(
            Agent_sac, "select_next_action"
        ), "missing SAC agent action selection"
        assert hasattr(
            Agent_sac, "eval_next_action"
        ), "missing SAC agent evaluation action selection"
        assert hasattr(
            Agent_sac, "_mini_batch"
        ), "missing SAC mini-batch sampling functionality"
        assert hasattr(
            Agent_sac, "_multi_step_target"
        ), "missing SAC target Q-value generation functionality"
        assert hasattr(Agent_sac, "learn"), "missing SAC agent learning functionality"
        assert hasattr(
            Agent_sac, "_update_critic_parameters"
        ), "missing SAC network update functionality"
        assert hasattr(Agent_sac, "save_models"), "missing SAC agent save functionality"
        assert hasattr(Agent_sac, "load_models"), "missing SAC agent load functionality"

    # TD3 algorithm method checks
    if "TD3" in inputs["algo_name"]:
        assert hasattr(
            ActorNetwork_td3, "forward"
        ), "missing TD3 actor forward propagation"
        assert hasattr(
            ActorNetwork_td3, "save_checkpoint"
        ), "missing TD3 actor saving functionality"
        assert hasattr(
            ActorNetwork_td3, "load_checkpoint"
        ), "missing TD3 actor load functionality"
        assert hasattr(
            CriticNetwork_td3, "forward"
        ), "missing TD3 critic forward propagation"
        assert hasattr(
            CriticNetwork_td3, "save_checkpoint"
        ), "missing TD3 critic saving functionality"
        assert hasattr(
            CriticNetwork_td3, "load_checkpoint"
        ), "missing TD3 critic load functionality"
        assert hasattr(
            Agent_td3, "store_transistion"
        ), "missing TD3 transition storage functionality"
        assert hasattr(
            Agent_td3, "select_next_action"
        ), "missing TD3 agent action selection"
        assert hasattr(
            Agent_td3, "eval_next_action"
        ), "missing TD3 agent evaluation action selection"
        assert hasattr(
            Agent_td3, "_mini_batch"
        ), "missing TD3 mini-batch sampling functionality"
        assert hasattr(
            Agent_td3, "_multi_step_target"
        ), "missing TD3 target Q-value generation functionality"
        assert hasattr(Agent_td3, "learn"), "missing TD3 agent learning functionality"
        assert hasattr(
            Agent_td3, "_update_critic_parameters"
        ), "missing TD3 network update functionality"
        assert hasattr(Agent_td3, "save_models"), "missing TD3 agent save functionality"
        assert hasattr(Agent_td3, "load_models"), "missing TD3 agent load functionality"

    # replay buffer method checks
    if inputs["buffer_gpu"] == False:
        assert hasattr(
            ReplayBuffer, "_episode_history"
        ), "missing episode storage functionality"
        assert hasattr(
            ReplayBuffer, "store_exp"
        ), "missing transition store functionality"
        assert hasattr(
            ReplayBuffer, "_construct_history"
        ), "missing history construction functionality"
        assert hasattr(
            ReplayBuffer, "_episode_rewards_states_actions"
        ), "missing multi-step episode variables functionality"
        assert hasattr(
            ReplayBuffer, "_multi_step_batch"
        ), "missing multi-step mini-batch functionality"
        assert hasattr(
            ReplayBuffer, "sample_exp"
        ), "missing uniform transition sampling functionality"
    else:
        assert hasattr(
            ReplayBufferTorch, "_episode_history"
        ), "missing episode storage functionality"
        assert hasattr(
            ReplayBufferTorch, "store_exp"
        ), "missing transition store functionality"
        assert hasattr(
            ReplayBufferTorch, "_construct_history"
        ), "missing history construction functionality"
        assert hasattr(
            ReplayBufferTorch, "_episode_rewards_states_actions"
        ), "missing multi-step episode variables functionality"
        assert hasattr(
            ReplayBufferTorch, "_multi_step_batch"
        ), "missing multi-step mini-batch functionality"
        assert hasattr(
            ReplayBufferTorch, "sample_exp"
        ), "missing uniform transition sampling functionality"

    print("Algorithm Method Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def agent_tests(
    TEST_PYBULLET: int,
    TEST_MULTI: int,
    TEST_MULTI_SH: int,
    TEST_MARKET: int,
    TEST_SAC: int,
    TEST_TD3: int,
    TEST_CRITICS_EXTRA: int,
    TEST_CRITICS_RARE: int,
    TEST_MULTI_STEPS: int,
) -> None:
    """
    Conduct tests for agent training script.

    Parameters:
        Refer to `./tools/agent_tests.py` for input details.
    """
    assert isinstance(TEST_PYBULLET, int), ti
    assert TEST_PYBULLET == 0 or TEST_PYBULLET == 1, "must be 0 or 1"
    assert isinstance(TEST_MULTI, int), ti
    assert TEST_MULTI == 0 or TEST_MULTI == 1, "must be 0 or 1"
    assert isinstance(TEST_MULTI_SH, int), ti
    assert TEST_MULTI_SH == 0 or TEST_MULTI_SH == 1, "must be 0 or 1"
    assert isinstance(TEST_MARKET, int), ti
    assert TEST_MARKET == 0 or TEST_MARKET == 1, "must be 0 or 1"

    assert isinstance(TEST_SAC, int), ti
    assert TEST_SAC == 0 or TEST_SAC == 1, "must be 0 or 1"
    assert isinstance(TEST_TD3, int), ti
    assert TEST_TD3 == 0 or TEST_TD3 == 1, "must be 0 or 1"
    assert TEST_SAC == 1 or TEST_TD3 == 1, "must select at least one algorithm"

    assert isinstance(TEST_CRITICS_EXTRA, int), ti
    assert TEST_CRITICS_EXTRA == 0 or TEST_CRITICS_EXTRA == 1, "must be 0 or 1"
    assert isinstance(TEST_CRITICS_RARE, int), ti
    assert TEST_CRITICS_RARE == 0 or TEST_CRITICS_RARE == 1, "must be 0 or 1"

    assert isinstance(TEST_MULTI_STEPS, int), ti
    assert TEST_MULTI_STEPS == 0 or TEST_MULTI_STEPS == 1, "must be 0 or 1"

    print(
        "--------------------------------------------------------------------------------"
    )

    print("Agent Training Tests: Passed")
