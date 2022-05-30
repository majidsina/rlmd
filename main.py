"""
title:                  main.py
usage:                  python main.py
python version:         3.10
torch verison:          1.11
gym version:            0.24
pybullet version:       3.2

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Responsible for executing all agent training and conducting tests on provided inputs.

Instructions:
    1. Select environments, algorithms, critic loss functions, and multi-steps
       using available options and enter them into the provided first four lists
       (all must be non-empty).
    2. Modify inputs dictionary containing agent training/learning parameters and
       reinforcement learning model hyperparameters if required.
    3. Running python file will provide live progress of learning details in the terminal.
    4. Upon completion, all learned PyTorch parameters, data, and plots will be
       placed within the ./results/ directory. In all cases, these will be organised
       by the reward dynamic (additive or multiplicative or market). Inside each of
       these there will exist directories /data and /models directory containing
       sub-directories titled by full environment ID (env_id) containing all output
       data and summary plots.
"""

import time
from typing import Dict, List

# fmt: off

gym_envs: Dict[str, list] = {
    # ENV_KEY: [env_id, state_dim, action_dim, initial warm-up steps to generate random seed]

    # ADDITIVE ENVIRONMENTS
        # OpenAI Box2D continuous control tasks
            "0": ["LunarLanderContinuous", 8, 2, 1e3],
            "1": ["BipedalWalker", 24, 4, 1e3],
            "2": ["BipedalWalkerHardcore", 24, 4, 1e3],
        # Roboschool environments ported to PyBullet
            "3": ["CartPoleContinuousBulletEnv-v0", 4, 1, 1e3],
            "4": ["InvertedPendulumBulletEnv-v0", 5, 1, 1e3],
            "5": ["InvertedDoublePendulumBulletEnv-v0", 9, 1, 1e3],
            "6": ["HopperBulletEnv-v0", 15, 3, 1e3],
            "7": ["Walker2DBulletEnv-v0", 22, 6, 1e3],
            "8": ["HalfCheetahBulletEnv-v0", 26, 6, 1e4],
            "9": ["AntBulletEnv-v0", 28, 8, 1e4],
            "10": ["HumanoidBulletEnv-v0", 44, 17, 1e4],
        # KOD*LAB quadruped direct-drive legged robots ported to PyBullet
            "11": ["MinitaurBulletEnv-v0", 28, 8, 1e4],
        # DeepMimic simulation of a imitating Humanoid mimic ported to PyBullet
            "12": ["HumanoidDeepMimicWalkBulletEnv-v1", 197, 36, 1e4],
            "13": ["HumanoidDeepMimicBackflipBulletEnv-v1", 197, 36, 1e4],

    # MULTIPLICATVE ENVIRONMENTS (MARKOV)
        # InvA = Investor A,  InvB = Investor B,  InvC = Investor C
        # assets following the binary coin flip
            "14": ["Coin_InvA", 5, 1, 1e3],
            "15": ["Coin_InvB", 5, 2, 1e3],
            "16": ["Coin_InvC", 5, 3, 1e3],
        # assets following the trinary dice roll
            "17": ["Dice_InvA", 5, 1, 1e3],
            "18": ["Dice_InvB", 5, 2, 1e3],
            "19": ["Dice_InvC", 5, 3, 1e3],
        # assets following GBM
            "20": ["GBM_InvA", 5, 1, 1e3],
            "21": ["GBM_InvB", 5, 2, 1e3],
            "22": ["GBM_InvC", 5, 3, 1e3],

    # MULTIPLICATVE INSURANCE SAFE HAVEN ENVIRONMENTS (MARKOV)
        # single asset cost-effective risk mitigation with insurance
            "23": ["Dice_SH_INSURED", 6, 1, 1e3],
        # InvA = Investor A,  InvB = Investor B,  InvC = Investor C
        # single asset following the dice roll with insurance safe havens
            "24": ["Dice_SH_InvA", 6, 2, 1e3],
            "25": ["Dice_SH_InvB", 6, 3, 1e3],
            "26": ["Dice_SH_InvC", 6, 4, 1e3],

    # MARKET ENVIRONMENTS (NON-MARKOV)
        # InvA = Investor A,  InvB = Investor B,  InvC = Investor C
        # SNP: S&P500 index (^SPX)
            "27": ["SNP_InvA", 5, 1, 1e3],
            "28": ["SNP_InvB", 5, 2, 1e3],
            "29": ["SNP_InvC", 5, 3, 1e3],
        # USEI: US-listed equity indicies (^SPX, ^NDX, ^DJIA)
            "30": ["EI_InvA", 7, 3, 1e3],
            "31": ["EI_InvB", 7, 4, 1e3],
            "32": ["EI_InvC", 7, 5, 1e3],
        # Minor: USEI + Gold, Silver, WTI
            "33": ["Minor_InvA", 10, 6, 1e3],
            "34": ["Minor_InvB", 10, 7, 1e3],
            "35": ["Minor_InvC", 10, 8, 1e3],
        # Medium: Minor + Cooper, Platinum, Lumber
            "36": ["Medium_InvA", 13, 9, 1e3],
            "37": ["Medium_InvB", 13, 10, 1e3],
            "38": ["Medium_InvC", 13, 11, 1e3],
        # Major: Medium + Palladium, RBOB, Cattle, Coffee, OJ
            "39": ["Major_InvA", 18, 14, 1e3],
            "40": ["Major_InvB", 18, 15, 1e3],
            "41": ["Major_InvC", 18, 16, 1e3],
        # DJI: USEI + 25/30 Dow Jones (^DJIA) components
            "42": ["DJI_InvA", 33, 29, 1e3],
            "43": ["DJI_InvB", 33, 30, 1e3],
            "44": ["DJI_InvC", 33, 31, 1e3],
        # Full: Major + 26/30 Dow Jones (^DJIA) components
            "45": ["Full_InvA", 44, 40, 1e3],
            "46": ["Full_InvB", 44, 41, 1e3],
            "47": ["Full_InvC", 44, 42, 1e3],

    # GUIDANCE ENVIRONMENTS (NON-MARKOV) ... ARE NOT PUBLIC!
        # NW = no wind,  CW = constant wind,  VW = variable wind
        # point projectile targeting (2D)
            "48": ["Laminar_2D_NW", 10, 1, 1e0],
            "49": ["Laminar_2D_CW", 13, 1, 1e3],
            "50": ["Laminar_2D_VW", 13, 1, 1e3],
        # point projectile targeting (3D)
            "51": ["Laminar_3D_NW", 14, 2, 1e3],
            "52": ["Laminar_3D_CW", 17, 2, 1e3],
            "53": ["Laminar_3D_VW", 17, 2, 1e3],
        # two-stage targeting and delivery of payloads
            "54": ["2Stage_NW", 59, 8, 1e3],
            "55": ["2Stage_CW", 62, 8, 1e3],
            "56": ["2Stage_VW", 62, 8, 1e3],
        # countermeasures for delivery of payloads
            "57": ["Counter_NW", 75, 8, 1e3],
            "58": ["Counter_CW", 78, 8, 1e3],
            "59": ["Counter_VW", 78, 8, 1e3],
    }

# fmt: on

# environments to train agent: [integer ENV_KEY] from gym_envs dict
envs: List[int] = [27]

# model-free off-policy (continous action space) agents: ["SAC", "TD3"]
algo: List[str] = ["SAC"]

# critic loss functions: ["MSE", "HUB", "MAE", "HSC", "CAU", "TCAU", "CIM", "MSE2", "MSE4", "MSE6"]
critic: List[str] = ["MSE"]

# bootstrapping of target critic values and discounted rewards: [integer > 0]
multi_steps: List[int] = [1]

# fmt: off

inputs: dict = {
    # LEARNING PARAMETERS
        # additive environment execution parameters
            "n_trials_add": 10,                 # number of training trials
            "n_cumsteps_add": 3e5,              # training steps per trial (must be greater than environment warm-up)
            "eval_freq_add": 1e3,               # interval (steps) between evaluation episodes
            "n_eval_add": 1e1,                  # number of evaluation episodes
            "max_eval_reward": 1e4,             # maximum score per evaluation episode
            "actor_percentile_add": 100,        # bottom percentile of actor mini-batch to be maximised (>0, <=100)
            "critic_percentile_add": 100,       # top percentile of critic mini-batch to be minimised (>0, <=100)

        # multiplicative environment execution parameters
            "n_trials_mul": 10,                 # ibid.
            "n_cumsteps_mul": 3e4,              # ibid.
            "eval_freq_mul": 1e3,               # ibid.
            "n_eval_mul": 1e2,                  # ibid.
            "max_eval_steps_mul": 1e2,          # maximum steps per evaluation episode
            "smoothing_window_mul": 2e3,        # training steps up to which action smoothing window is applied
            "actor_percentile_mul": 50,         # ibid.
            "critic_percentile_mul": 50,        # ibid.

            "n_gambles":                        # number of simultaneous gambles (List[int] >0)
                [1],

        # market environment execution parameters
            "market_dir":                       # directory containing historical market data
                "./tools/market_data/",

            "n_trials_mkt": 10,                 # ibid.
            "n_cumsteps_mkt": 1e5,              # ibid.
            "eval_freq_mkt": 1e3,               # ibid.
            "n_eval_mkt": 1e2,                  # ibid.
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
                [1],

        # guidance environment execution parameters
            "n_trials_gud": 10,                 # ibid.
            "n_cumsteps_gud": 3e4,              # ibid.
            "eval_freq_gud": 1e3,               # ibid.
            "n_eval_gud": 1e2,                  # ibid.
            "max_eval_steps_gud": 1e3,          # ibid.
            "actor_percentile_gud": 50,         # ibid.
            "critic_percentile_gud": 50,        # ibid.

            "targets":                          # number of unique targets for two-stage and countermeasures
                [3],

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

    from tests.input_tests import algo_method_checks, env_tests, learning_tests
    from tools.utils import (
        device_details,
        env_dynamics,
        input_initialisation,
        load_market_data,
    )

    device_details(inputs)

    inputs = input_initialisation(inputs, envs, algo, critic, multi_steps)

    # CONDUCT TESTS
    learning_tests(inputs)  # learning algorithm checks
    env_tests(gym_envs, inputs)  # environment setup tests
    algo_method_checks(inputs)  # class method existence checks

    # IMPORT SCRIPTS
    from scripts.rl_additive import additive_env
    from scripts.rl_market import market_env
    from scripts.rl_multiplicative import multiplicative_env

    # from scripts.rl_laminar import laminar_env
    # from scripts.rl_twostage import twostage_env
    # from scripts.rl_catch import counter_env

    # obtain environment key limits based on reward dynamics
    env_splits = env_dynamics(gym_envs)
    multi_key, sh_key, market_key, market_env_keys, gud_key, two_key, counter_key = (
        env_splits[0],
        env_splits[1],
        env_splits[2],
        env_splits[3],
        env_splits[4],
        env_splits[5],
        env_splits[6],
    )

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

        # guidance environments
        # elif env_key < two_key:
        #     laminar_env(gym_envs, inputs)
        # elif env_key < counter_key:
        #     twostage_env(gym_envs, inputs)
        # elif env_key >= counter_key:
        #     counter_env(gym_envs, inputs)

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
