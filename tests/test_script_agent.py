"""
title:                  test_script_agent.py
usage:                  python tests/test_script_agent.py
python version:         3.10
torch verison:          1.11
gym version:            0.24
pybullet version:       3.2

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Conduct tests on reinforcement learning agent across a wide variety of default
    situations with comprehensiveness dependent on user selection.

Instructions:
    1. Select which items to include in tests.
    2. Running the file will provide live progress in the terminal.
"""

import sys

sys.path.append("./")

import os
import shutil
import time

from main import gym_envs
from tests.test_input_agent import (
    agent_tests,
    algo_method_checks,
    env_tests,
    learning_tests,
)
from tools.utils import (
    device_details,
    env_dynamics,
    input_initialisation,
    load_market_data,
)

# select environments and variables to be tests
# integer selection using False == 0 and True == 1

# environments
TEST_PYBULLET = 1
TEST_MULTI = 1
TEST_MULTI_SH = 1
TEST_MARKET = 1

# algorithms (must select at least one)
TEST_SAC = 1
TEST_TD3 = 1

# additional critic loss functions
TEST_CRITICS_EXTRA = 0
TEST_CRITICS_RARE = 0

# bootstraps
TEST_MULTI_STEPS = 1

# fmt: off

inputs = {
    # LEARNING PARAMETERS
        # additive environment execution parameters
            "n_trials_add": 2,                  # number of training trials
            "n_cumsteps_add": 3e3,              # training steps per trial (must be greater than environment warm-up)
            "eval_freq_add": 1e3,               # interval (steps) between evaluation episodes
            "n_eval_add": 1e1,                  # number of evaluation episodes
            "max_eval_reward": 1e4,             # maximum score per evaluation episode
            "actor_percentile_add": 100,        # bottom percentile of actor mini-batch to be maximised (>0, <=100)
            "critic_percentile_add": 100,       # top percentile of critic mini-batch to be minimised (>0, <=100)

        # multiplicative environment execution parameters
            "n_trials_mul": 2,                  # ibid.
            "n_cumsteps_mul": 3e3,              # ibid.
            "eval_freq_mul": 1e3,               # ibid.
            "n_eval_mul": 1e1,                  # ibid.
            "max_eval_steps_mul": 1e2,          # maximum steps per evaluation episode
            "smoothing_window_mul": 2e3,        # training steps up to which action smoothing window is applied
            "actor_percentile_mul": 50,         # ibid.
            "critic_percentile_mul": 50,        # ibid.

            "n_gambles":                        # number of simultaneous gambles (List[int] >0)
                [1, 5],

        # market environment execution parameters
            "market_dir":                       # directory containing historical market data
                "./tools/market_data/",

            "n_trials_mkt": 2,                  # ibid.
            "n_cumsteps_mkt": 3e3,              # ibid.
            "eval_freq_mkt": 1e3,               # ibid.
            "n_eval_mkt": 1e1,                  # ibid.
            "smoothing_window_mkt": 2e3,        # ibid.
            "actor_percentile_mkt": 50,         # ibid.
            "critic_percentile_mkt": 50,        # ibid.

            "action_days": 1,                   # number of days between agent trading actions (252 day years)
            "train_days": 1e3,                  # length of each training period
            "test_days": 250,                   # length of each evaluation period
            "train_shuffle_days": 5,            # interval size (>=1) to be shuffled for training
            "test_shuffle_days": 3,             # interval size (>=1) to be shuffled for inference
            "gap_days_min": 5,                  # minimum spacing (>=0) between training and testing windows
            "gap_days_max": 20,                 # maximum spacing (>=gap_days_min) between training-testing windows

            "past_days":                        # number of previous observed days (POMDP if =1) (List[int] >0)
                [1, 5],

        # guidance environment execution parameters
            "n_trials_gud": 2,                  # ibid.
            "n_cumsteps_gud": 3e4,              # ibid.
            "eval_freq_gud": 1e3,               # ibid.
            "n_eval_gud": 1e1,                  # ibid.
            "max_eval_steps_gud": 1e3,          # ibid.
            "actor_percentile_gud": 50,         # ibid.
            "critic_percentile_gud": 50,        # ibid.

            "targets":                          # number of unique targets for two-stage and countermeasures
                [3, 5],

        # learning variables
            "gpu": "cuda:0",                    # CUDA-based GPU to be used by PyTorch or use CPU ("cpu")
            "buffer_gpu": False,                # GPU-based replay buffer (may be faster for single-step, much slower for multi-step)

            "buffer": 1e6,                      # maximum transitions in experience replay buffer
            "discount": 0.99,                   # discount factor for successive steps
            "trail": 50,                        # moving average of training episode scores used for model saving
            "cauchy_scale": 1,                  # Cauchy scale parameter initialisation value
            "r_abs_zero": None,                 # defined absolute zero value for rewards added to buffer
            "continue": False,                  # whether to continue learning with same parameters across trials

        # critic loss aggregation
            "critic_mean_type": "E",            # critic learning using either empirical "E" or shadow "S" means (only E)
            "shadow_low_mul": 1e0,              # lower bound multiplier of minimum for critic power law
            "shadow_high_mul": 1e1,             # upper bound multiplier of maximum for critic power law

    # MODEL HYPERPARAMETERS
        # SAC hyperparameters (https://arxiv.org/pdf/1812.05905.pdf)
            "sac_actor_learn_rate": 3e-4,       # actor learning rate (Adam optimiser)
            "sac_critic_learn_rate": 3e-4,      # critic learning rate (Adam optimiser)
            "sac_temp_learn_rate": 3e-4,        # log temperature learning rate (Adam optimiser)
            "sac_layer_1_units": 256,           # nodes in first fully connected layer
            "sac_layer_2_units": 256,           # nodes in second fully connected layer
            "sac_actor_step_update": 1,         # actor policy network update frequency (steps)
            "sac_temp_step_update": 1,          # temperature update frequency (steps)
            "sac_target_critic_update": 1,      # target critic networks update frequency (steps)
            "sac_target_update_rate": 5e-3,     # Polyak averaging rate for target network parameter updates

            "initial_logtemp": 0,               # initial log weighting given to entropy maximisation
            "reward_scale": 1,                  # constant scaling factor of next reward ("inverse temperature")
            "log_scale_min": -20,               # minimum log scale for stochastic policy
            "log_scale_max": 2,                 # maximum log scale for stochastic policy
            "reparam_noise": 1e-6,              # miniscule constant to keep logarithm of actions bounded

        # TD3 hyperparameters (https://arxiv.org/pdf/1802.09477.pdf)
            "td3_actor_learn_rate": 1e-3,       # ibid.
            "td3_critic_learn_rate": 1e-3,      # ibid.
            "td3_layer_1_units": 400,           # ibid.
            "td3_layer_2_units": 300,           # ibid.
            "td3_actor_step_update": 2,         # ibid.
            "td3_target_actor_update": 2,       # target actor network update frequency (steps)
            "td3_target_critic_update": 2,      # ibid.
            "td3_target_update_rate": 5e-3,     # ibid.

            "policy_noise": 0.1,                # exploration noise added to next actions
            "target_policy_noise": 0.2,         # noise added to next target actions acting as a regulariser
            "target_policy_clip": 0.5,          # clipping limit of noise added to next target actions

        # shared parameters
            "sample_dist":                      # policy distribution (normal "N", Laplace "L", or multi-variate normal "MVN")
                {"SAC": "N", "TD3": "N"},
            "batch_size":                       # mini-batch size for actor-critic neural networks
                {"SAC": 256, "TD3": 100},
            "grad_step":                        # standard gradient update frequency (steps)
                {"SAC": 1, "TD3": 1},
            "log_noise": 1e-6,                  # miniscule constant to keep tail estimation logarithm bounded
    }

# fmt: on

if __name__ == "__main__":

    # CONDUCT TESTS
    agent_tests(
        TEST_PYBULLET,
        TEST_MULTI,
        TEST_MULTI_SH,
        TEST_MARKET,
        TEST_SAC,
        TEST_TD3,
        TEST_CRITICS_EXTRA,
        TEST_CRITICS_RARE,
        TEST_MULTI_STEPS,
    )

    envs = []
    # arbitrarily selected environments to be tested
    if TEST_PYBULLET:
        envs += [1, 4]
    if TEST_MULTI:
        envs += [8, 13, 15]
    if TEST_MULTI_SH:
        envs += [17, 19]
    if TEST_MARKET:
        envs += [21, 31, 41]

    algo = []
    if TEST_SAC:
        algo.append("SAC")
    if TEST_TD3:
        algo.append("TD3")

    critic = ["MSE"]
    if TEST_CRITICS_EXTRA:
        critic += ["HUB", "MAE", "HSC"]
    if TEST_CRITICS_RARE:
        critic += ["CAU", "TCAU", "CIM", "MSE2", "MSE4", "MSE6"]

    multi_steps = [1]
    if TEST_MULTI_STEPS:
        multi_steps += [5]

    # clean-up test files from earlier uncompleted tests
    if os.path.exists("./results/additive-test/"):
        shutil.rmtree("./results/additive-test/")
    if os.path.exists("./results/multiplicative-test/"):
        shutil.rmtree("./results/multiplicative-test/")
    if os.path.exists("./results/market-test/"):
        shutil.rmtree("./results/market-test/")

    device_details(inputs)

    inputs = input_initialisation(inputs, envs, algo, critic, multi_steps)

    learning_tests(inputs)  # learning algorithm checks
    env_tests(gym_envs, inputs)  # environment setup tests
    algo_method_checks(inputs)  # class method existence checks

    from scripts.rl_additive import additive_env
    from scripts.rl_market import market_env
    from scripts.rl_multiplicative import multiplicative_env

    (
        multi_key,
        sh_key,
        market_key,
        market_env_keys,
        gud_key,
        two_key,
        counter_key,
    ) = env_dynamics(gym_envs)

    inputs["test_agent"] = True

    start_time = time.perf_counter()

    for env_key in envs:
        begin_time = time.perf_counter()

        inputs["ENV_KEY"] = env_key

        # additive environments
        if env_key < multi_key:
            additive_env(gym_envs, inputs)

        # multiplicative environments
        elif env_key < sh_key:
            for gambles in inputs["n_gambles"]:
                multiplicative_env(gym_envs, inputs, n_gambles=gambles)

        # multiplicative insurance safe haven environments
        elif env_key < market_key:
            multiplicative_env(gym_envs, inputs, n_gambles=1)

        # market environments
        elif env_key < gud_key:
            data = load_market_data(env_key, market_env_keys, inputs)

            for days in inputs["past_days"]:
                market_env(gym_envs, inputs, market_data=data, obs_days=days)

        finish_time = time.perf_counter()
        env_time = finish_time - begin_time

        print(
            "ENV_KEY {} TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
                env_key, env_time, env_time / 60, env_time / 3600
            )
        )

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(
        "TOTAL TIME: {:1.0f}s = {:1.1f}m = {:1.2f}h".format(
            total_time, total_time / 60, total_time / 3600
        )
    )

    # CLEAN UP TEST FILES

    if TEST_PYBULLET == 1:
        shutil.rmtree("./results/additive-test/")
    if TEST_MULTI == 1 or TEST_MULTI_SH == 1:
        shutil.rmtree("./results/multiplicative-test/")
    if TEST_MARKET == 1:
        shutil.rmtree("./results/market-test/")

    print(
        "--------------------------------------------------------------------------------"
    )

    print("All Selected Agent Training Tests: Passed")
