"""
title:                  gen_figures.py
usage:                  python scripts/gen_figures.py
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
    Responsible for aggregating data and generating all final summary figures
    in the report for all experiments.

Instructions:
    1. Select additive environment aggregation inputs. Enter into the dictionary
       the common training hyperparameters and then into the lists the features
       that are varied.
    2. Select multiplicative environment aggregation inputs. Enter into each of the
       dictionaries the common training hyperparameters and into the lists the
       appropriate environments.
    3. Select reward dynamics to be plotted.
    4. Run python file and all figures will be placed inside the `./results/figures`
       directory with duplicate PNG and SVG images provided.
    5. If unsatisfied with graphically displayed growth and leverage values for
       aggerate figures, manual modification and adjustment of the plotted bounds is
       possible using the provided lists.
"""

import sys

sys.path.append("./")

import os
from typing import List

import plotting.plots_figures as plots
import tools.aggregate_data as aggregate_data
import tools.utils as utils
from main import gym_envs
from tests.input_tests import figure_tests

# ADDITIVE ENVIRONMENTS

# must select exactly four environments and two algorithms (can repeat selection)
# environment ENV_KEYS from main.gym_envs dictionary
add_envs: List[int] = [3, 4, 5, 7]
# title of plotted environment results
add_name: List[str] = ["Hopper", "Walker", "Cheetah", "Humanoid"]
add_algos: List[str] = ["SAC", "TD3"]
add_loss: List[str] = [
    "MSE",
    "HUB",
    "MAE",
    "HSC",
    "CAU",
    "TCAU",
    "MSE2",
    "MSE4",
    "MSE6",
]
add_multi: List[int] = [1, 3, 5, 7, 9]

add_inputs: dict = {
    "n_trials": 10,
    "n_cumsteps": 3e5,  # 300,000 training steps
    "eval_freq": 1e3,
    "n_eval": 1e1,
    "buffer": 1e6,
    "critic_mean_type": "E",
    "s_dist": "N",
    "multi_steps": 1,  # default for varying critic loss functions
}

# MULTIPLICATVE ENVIRONMENTS (MARKOV)

# must select exactly three (non-unique) number of simultaneous identical gambles
n_gambles: List[int] = [1, 3, 5]

# must select exactly three (non-unique) environments for each list
# assets following the coin flip
coin_keys: List[int] = [8, 9, 10]
# assets following the dice roll
dice_keys: List[int] = [11, 12, 13]
# assets following GBM
gbm_keys: List[int] = [14, 15, 16]

mul_inputs_td3: dict = {
    "n_trials": 10,
    "n_cumsteps": 5e4,  # 50,000 training steps
    "eval_freq": 1e3,
    "n_eval": 1e2,
    "buffer": 1e6,
    "critic_mean_type": "E",
    "s_dist": "N",
    "algo": "TD3",
    "loss_fn": "MSE",
    "multi_steps": 1,
}

mul_inputs_sac = mul_inputs_td3.copy()
mul_inputs_sac["algo"] = "SAC"
mul_inputs_sac["s_dist"] = "N"

# MULTIPLICATVE INSURANCE SAFE HAVEN ENVIRONMENTS (MARKOV)

# must select exactly two (non-unique) environments for each list
# single asset following the dice roll with safe haven
dice_sh_keys: List[int] = [11, 17]

spitz_inputs_td3: dict = {
    "n_trials": 10,
    "n_cumsteps": 5e4,  # 50,000 training steps
    "eval_freq": 1e3,
    "n_eval": 1e2,
    "buffer": 1e6,
    "critic_mean_type": "E",
    "s_dist": "N",
    "algo": "TD3",
    "loss_fn": "MSE",
    "multi_steps": 1,
}

spitz_inputs_sac = spitz_inputs_td3.copy()
spitz_inputs_sac["algo"] = "SAC"
spitz_inputs_sac["s_dist"] = "N"

dice_sh_a_keys: List[int] = [11, 18]
dice_sh_b_keys: List[int] = [12, 19]
dice_sh_c_keys: List[int] = [13, 20]

ins_inputs_td3: dict = {
    "n_trials": 10,
    "n_cumsteps": 5e4,  # 50,000 training steps
    "eval_freq": 1e3,
    "n_eval": 1e2,
    "buffer": 1e6,
    "critic_mean_type": "E",
    "s_dist": "N",
    "algo": "TD3",
    "loss_fn": "MSE",
    "multi_steps": 1,
}

ins_inputs_sac = ins_inputs_td3.copy()
ins_inputs_sac["algo"] = "SAC"
ins_inputs_sac["s_dist"] = "N"

# MARKERT ENVIRONMENTS (NON-MARKOV)

# must select exactly three (non-unique) number of observed days
obs_days: List[int] = [1, 3, 5]
# action spacing for all environments
action_days: int = 1

# market environment keys (non-unique)
mkt_name: List[str] = ["snp", "usei", "minor", "medium", "major", "dji", "full"]
mkt_invA_env: List[int] = [21, 24, 27, 30, 33, 36, 39]
mkt_invB_env: List[int] = [22, 25, 28, 31, 34, 37, 40]
mkt_invC_env: List[int] = [23, 26, 29, 32, 35, 38, 41]

# only obtain results for investor A
only_invA = True

mkt_inputs_td3: dict = {
    "n_trials": 10,
    "n_cumsteps": 1e5,  # 100,000 training steps
    "eval_freq": 1e3,
    "n_eval": 1e2,
    "buffer": 1e6,
    "critic_mean_type": "E",
    "s_dist": "N",
    "algo": "TD3",
    "loss_fn": "MSE",
    "multi_steps": 1,
}

mkt_inputs_sac = mkt_inputs_td3.copy()
mkt_inputs_sac["algo"] = "SAC"
mkt_inputs_sac["s_dist"] = "N"

# select reward dynamics to be plotted
# integer selection using False == 0 and True == 1
PLOT_ADDITIVE = 0
PLOT_MULTIPLICATIVE = 0
PLOT_MULTIPLICATIVE_SH = 0
PLOT_MARKET = 0
PLOT_GUIDANCE = 0

if __name__ == "__main__":

    # directory to save figures
    path = "./results/figs/"
    # directory containing historical market data
    market_data_path = "./tools/market_data/"

    # CONDUCT TESTS
    figure_tests(
        path,
        market_data_path,
        gym_envs,
        add_envs,
        add_name,
        add_algos,
        add_loss,
        add_multi,
        add_inputs,
        n_gambles,
        coin_keys,
        dice_keys,
        gbm_keys,
        mul_inputs_td3,
        mul_inputs_sac,
        dice_sh_keys,
        spitz_inputs_td3,
        spitz_inputs_sac,
        dice_sh_a_keys,
        dice_sh_b_keys,
        dice_sh_c_keys,
        ins_inputs_td3,
        ins_inputs_sac,
        obs_days,
        action_days,
        mkt_name,
        mkt_invA_env,
        mkt_invB_env,
        mkt_invC_env,
        only_invA,
        mkt_inputs_td3,
        mkt_inputs_sac,
        PLOT_ADDITIVE,
        PLOT_MULTIPLICATIVE,
        PLOT_MULTIPLICATIVE_SH,
        PLOT_MARKET,
        PLOT_GUIDANCE,
    )

    if not os.path.exists(path):
        os.makedirs(path)

    # ADDITIVE ENVIRONMENTS

    if PLOT_ADDITIVE:

        # critic loss function plots
        plots.loss_fn_plot(path + "critic_loss")

        # critic loss functions
        path_loss: str = "add_loss"

        loss_data = aggregate_data.add_loss_aggregate(
            add_envs, gym_envs, add_inputs, add_algos, add_loss
        )
        (
            reward,
            closs,
            scale,
            kernel,
            logtemp,
            tail,
            cshadow,
            cmax,
            keqv,
        ) = aggregate_data.add_summary(add_inputs, loss_data)

        plots.plot_add(
            add_inputs,
            add_name,
            add_loss,
            False,
            reward,
            closs,
            scale,
            kernel,
            tail,
            cshadow,
            keqv,
            path + path_loss,
        )

        plots.plot_add_temp(
            add_inputs, add_name, add_loss, False, logtemp, path + path_loss
        )

        # multi-step returns
        path_multi: str = "add_multi"

        multi_data = aggregate_data.add_multi_aggregate(
            add_envs, gym_envs, add_inputs, add_algos, add_multi
        )

        (
            reward,
            closs,
            scale,
            kernel,
            logtemp,
            tail,
            cshadow,
            cmax,
            keqv,
        ) = aggregate_data.add_summary(add_inputs, multi_data)

        plots.plot_add(
            add_inputs,
            add_name,
            add_multi,
            True,
            reward,
            closs,
            scale,
            kernel,
            tail,
            cshadow,
            keqv,
            path + path_multi,
        )

        plots.plot_add_temp(
            add_inputs, add_name, add_multi, True, logtemp, path + path_multi
        )

    # MULTIPLICATIVE ENVIRONMENTS (MARKOV)

    if PLOT_MULTIPLICATIVE:

        # action smoothing function plots
        plots.plot_smoothing_fn(path + "action_smoothing")

        # number of simultaneous gambles
        n_str = ["_" + str(n) for n in n_gambles]

        # environment: coin flip
        path_env: str = "mul_coin_inv"
        for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

            algo = "_td3" if mul_inputs == mul_inputs_td3 else "_sac"

            coin_inv_n1 = aggregate_data.mul_inv_aggregate(
                coin_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                ls1,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, coin_inv_n1)

            plots.plot_inv(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                path + path_env + algo + n_str[0],
            )

            coin_inv_n2 = aggregate_data.mul_inv_aggregate(
                coin_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                ls2,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, coin_inv_n2)

            plots.plot_inv(
                mul_inputs,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                path + path_env + algo + n_str[1],
            )

            coin_inv_n3 = aggregate_data.mul_inv_aggregate(
                coin_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                ls3,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, coin_inv_n3)

            plots.plot_inv(
                mul_inputs,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                path + path_env + algo + n_str[2],
            )

            g_min = [-5, -20, -40] if mul_inputs == mul_inputs_td3 else [-5, -8, -25]
            g_max = [1, 5, 10] if mul_inputs == mul_inputs_td3 else [1, 3, None]
            l_min = (
                [-1.5, -0.5, -0.5]
                if mul_inputs == mul_inputs_td3
                else [None, -0.4, None]
            )
            l_max = [2, 1, 1.5] if mul_inputs == mul_inputs_td3 else [1.5, None, None]

            plots.plot_inv_all_n_perf(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                path + path_env + algo + "_perf",
                n_gambles,
                g_min,
                g_max,
                l_min,
                l_max,
                T=1,
                V_0=1,
            )

            plots.plot_inv_all_n_train(
                mul_inputs,
                closs_1,
                ctail_1,
                cshadow_1,
                keqv_1,
                closs_2,
                ctail_2,
                cshadow_2,
                keqv_2,
                closs_3,
                ctail_3,
                cshadow_3,
                keqv_3,
                path + path_env + algo + "_train",
                n_gambles,
            )

        # environment: dice roll
        path_env: str = "mul_dice_inv"
        for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

            algo = "_td3" if mul_inputs == mul_inputs_td3 else "_sac"

            dice_inv_n1 = aggregate_data.mul_inv_aggregate(
                dice_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                ls1,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, dice_inv_n1)

            plots.plot_inv(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                path + path_env + algo + n_str[0],
            )

            dice_inv_n2 = aggregate_data.mul_inv_aggregate(
                dice_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                ls2,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, dice_inv_n2)

            plots.plot_inv(
                mul_inputs,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                path + path_env + algo + n_str[1],
            )

            dice_inv_n3 = aggregate_data.mul_inv_aggregate(
                dice_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                ls3,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, dice_inv_n3)

            plots.plot_inv(
                mul_inputs,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                path + path_env + algo + n_str[2],
            )

            g_min = [-3, -6, -25] if mul_inputs == mul_inputs_td3 else [-2, -4, -20]
            g_max = [2, 2, 1] if mul_inputs == mul_inputs_td3 else [1.5, None, None]
            l_min = (
                [-1.5, -0.5, -0.5] if mul_inputs == mul_inputs_td3 else [-1, None, None]
            )
            l_max = [1.5, 1, 1] if mul_inputs == mul_inputs_td3 else [1.5, None, None]

            plots.plot_inv_all_n_perf(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                path + path_env + algo + "_perf",
                n_gambles,
                g_min,
                g_max,
                l_min,
                l_max,
                T=1,
                V_0=1,
            )

            plots.plot_inv_all_n_train(
                mul_inputs,
                closs_1,
                ctail_1,
                cshadow_1,
                keqv_1,
                closs_2,
                ctail_2,
                cshadow_2,
                keqv_2,
                closs_3,
                ctail_3,
                cshadow_3,
                keqv_3,
                path + path_env + algo + "_train",
                n_gambles,
            )

        #  maximum leverage with GBM plot
        plots.plot_gbm_max_lev(path + "gbm_max_lev")

        # environment: GBM
        path_env: str = "mul_gbm_inv"
        for mul_inputs in [mul_inputs_td3, mul_inputs_sac]:

            algo = "_td3" if mul_inputs == mul_inputs_td3 else "_sac"

            gbm_inv_n1 = aggregate_data.mul_inv_aggregate(
                gbm_keys, n_gambles[0], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                ls1,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, gbm_inv_n1)

            plots.plot_inv(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                closs_1,
                ctail_1,
                cshadow_1,
                cmax_1,
                keqv_1,
                path + path_env + algo + n_str[0],
            )

            gbm_inv_n2 = aggregate_data.mul_inv_aggregate(
                gbm_keys, n_gambles[1], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                ls2,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, gbm_inv_n2)

            plots.plot_inv(
                mul_inputs,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                closs_2,
                ctail_2,
                cshadow_2,
                cmax_2,
                keqv_2,
                path + path_env + algo + n_str[1],
            )

            gbm_inv_n3 = aggregate_data.mul_inv_aggregate(
                gbm_keys, n_gambles[2], gym_envs, mul_inputs, safe_haven=False
            )

            (
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                ls3,
            ) = aggregate_data.mul_inv_n_summary(mul_inputs, gbm_inv_n3)

            plots.plot_inv(
                mul_inputs,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                closs_3,
                ctail_3,
                cshadow_3,
                cmax_3,
                keqv_3,
                path + path_env + algo + n_str[2],
            )

            g_min = (
                [None, None, None]
                if mul_inputs == mul_inputs_td3
                else [None, None, None]
            )
            g_max = (
                [None, None, None]
                if mul_inputs == mul_inputs_td3
                else [None, None, None]
            )
            l_min = (
                [None, None, None]
                if mul_inputs == mul_inputs_td3
                else [None, None, None]
            )
            l_max = (
                [None, None, None]
                if mul_inputs == mul_inputs_td3
                else [None, None, None]
            )

            plots.plot_inv_all_n_perf(
                mul_inputs,
                reward_1,
                lev_1,
                stop_1,
                reten_1,
                reward_2,
                lev_2,
                stop_2,
                reten_2,
                reward_3,
                lev_3,
                stop_3,
                reten_3,
                path + path_env + algo + "_perf",
                n_gambles,
                g_min,
                g_max,
                l_min,
                l_max,
                T=1,
                V_0=1,
            )

            plots.plot_inv_all_n_train(
                mul_inputs,
                closs_1,
                ctail_1,
                cshadow_1,
                keqv_1,
                closs_2,
                ctail_2,
                cshadow_2,
                keqv_2,
                closs_3,
                ctail_3,
                cshadow_3,
                keqv_3,
                path + path_env + algo + "_train",
                n_gambles,
            )

    # MULTIPLICATVE INSURANCE SAFE HAVEN ENVIRONMENTS (MARKOV)-

    if PLOT_MULTIPLICATIVE_SH:

        # environment: dice roll with insurance safe haven
        path_env: str = "mul_dice_sh"

        for spitz_inputs in [spitz_inputs_td3, spitz_inputs_sac]:

            algo = "_td3" if spitz_inputs == spitz_inputs_td3 else "_sac"
            g_min = -2 if spitz_inputs == spitz_inputs_td3 else -5
            g_max = 4 if spitz_inputs == spitz_inputs_td3 else 4
            l_min = -0.25 if spitz_inputs == spitz_inputs_td3 else None
            l_max = 12 / 11 if spitz_inputs == spitz_inputs_td3 else 12 / 11

            dice_sh = aggregate_data.mul_inv_aggregate(
                dice_sh_keys, 1, gym_envs, spitz_inputs, safe_haven=True
            )

            (
                reward_sh,
                lev_sh,
                stop_sh,
                reten_sh,
                closs_sh,
                ctail_sh,
                cshadow_sh,
                cmax_sh,
                keqv_sh,
                levsh_sh,
            ) = aggregate_data.mul_inv_n_summary(spitz_inputs, dice_sh)

            plots.plot_safe_haven(
                spitz_inputs,
                reward_sh,
                lev_sh,
                stop_sh,
                reten_sh,
                closs_sh,
                ctail_sh,
                cshadow_sh,
                cmax_sh,
                keqv_sh,
                levsh_sh,
                path + path_env + algo,
                g_min,
                g_max,
                l_min,
                l_max,
                inv="a",
                T=1,
                V_0=1,
            )

        dice_sh_a1 = aggregate_data.mul_inv_aggregate(
            dice_sh_keys, 1, gym_envs, spitz_inputs_sac, safe_haven=True
        )

        dice_sh_a2 = aggregate_data.mul_inv_aggregate(
            dice_sh_keys, 1, gym_envs, spitz_inputs_td3, safe_haven=True
        )

        (
            reward_sh1,
            lev_sh1,
            stop_sh1,
            reten_sh1,
            closs_sh1,
            ctail_1,
            cshadow_sh1,
            cmax_sh1,
            keqv_sh1,
            levsh_sh1,
        ) = aggregate_data.mul_inv_n_summary(spitz_inputs_sac, dice_sh_a1)

        (
            reward_sh2,
            lev_sh2,
            stop_sh2,
            reten_sh2,
            closs_sh2,
            ctail_2,
            cshadow_sh2,
            cmax_sh2,
            keqv_sh2,
            levsh_sh2,
        ) = aggregate_data.mul_inv_n_summary(spitz_inputs_td3, dice_sh_a2)

        g_min, g_max = [-3, -3], [3, 3]
        l_min, l_max = [-0.25, -0.25], [12 / 11, 12 / 11]

        plots.plot_sh_perf(
            spitz_inputs_sac,
            reward_sh1,
            lev_sh1,
            reward_sh2,
            lev_sh2,
            path + path_env + "_perf",
            g_min,
            g_max,
            l_min,
            l_max,
            inv="a",
            T=1,
            V_0=1,
        )

        plots.plot_sh_train(
            spitz_inputs_sac,
            closs_sh1,
            ctail_1,
            cshadow_sh1,
            cmax_sh1,
            keqv_sh1,
            closs_sh2,
            ctail_2,
            cshadow_sh2,
            cmax_sh2,
            keqv_sh2,
            path + path_env + "_train",
            inv="a",
        )

        for ins_inputs in [ins_inputs_td3, ins_inputs_sac]:

            algo = "_td3" if ins_inputs == ins_inputs_td3 else "_sac"

            dice_inv_a = aggregate_data.mul_inv_aggregate(
                dice_sh_a_keys, 1, gym_envs, ins_inputs, safe_haven=True
            )

            (
                reward_a,
                lev_a,
                stop_a,
                reten_a,
                closs_a,
                ctail_a,
                cshadow_a,
                cmax_a,
                keqv_a,
                levsh_a,
            ) = aggregate_data.mul_inv_n_summary(
                ins_inputs, dice_inv_a, safe_haven=True
            )

            plots.plot_safe_haven(
                ins_inputs,
                reward_a,
                lev_a,
                stop_a,
                reten_a,
                closs_a,
                ctail_a,
                cshadow_a,
                cmax_a,
                keqv_a,
                levsh_a,
                path + path_env + algo + "_a",
                inv="a",
            )

            dice_inv_b = aggregate_data.mul_inv_aggregate(
                dice_sh_b_keys, 1, gym_envs, ins_inputs, safe_haven=True
            )

            (
                reward_b,
                lev_b,
                stop_b,
                reten_b,
                closs_b,
                ctail_b,
                cshadow_b,
                ctail_b,
                keqv_b,
                levsh_b,
            ) = aggregate_data.mul_inv_n_summary(
                ins_inputs, dice_inv_b, safe_haven=True
            )

            plots.plot_safe_haven(
                ins_inputs,
                reward_b,
                lev_b,
                stop_b,
                reten_b,
                closs_b,
                ctail_b,
                cshadow_b,
                ctail_b,
                keqv_b,
                levsh_b,
                path + path_env + algo + "_b",
                inv="b",
            )

            dice_inv_c = aggregate_data.mul_inv_aggregate(
                dice_sh_c_keys, 1, gym_envs, ins_inputs, safe_haven=True
            )

            (
                reward_c,
                lev_c,
                stop_c,
                reten_c,
                closs_c,
                ctail_c,
                cshadow_c,
                ctail_c,
                keqv_c,
                levsh_c,
            ) = aggregate_data.mul_inv_n_summary(
                ins_inputs, dice_inv_c, safe_haven=True
            )

            plots.plot_safe_haven(
                ins_inputs,
                reward_c,
                lev_c,
                stop_c,
                reten_c,
                closs_c,
                ctail_c,
                cshadow_c,
                ctail_c,
                keqv_c,
                levsh_c,
                path + path_env + algo + "_c",
                inv="c",
            )

            g_min = [-5, -5, -3] if ins_inputs == ins_inputs_td3 else [-4, -5, -3]
            g_max = [3, 2, 1] if ins_inputs == ins_inputs_td3 else [2.5, 1, 1]
            l_min = (
                [-0.5, -1, -1] if ins_inputs == ins_inputs_td3 else [None, None, None]
            )
            l_max = [2, 2, 2] if ins_inputs == ins_inputs_td3 else [None, None, None]

            plots.plot_inv_sh_perf(
                ins_inputs,
                reward_a,
                lev_a,
                stop_a,
                reten_a,
                levsh_a,
                reward_b,
                lev_b,
                stop_b,
                reten_b,
                levsh_b,
                reward_c,
                lev_c,
                stop_c,
                reten_c,
                levsh_c,
                path + path_env + algo + "_perf",
                g_min,
                g_max,
                l_min,
                l_max,
                T=1,
                V_0=1,
            )

            plots.plot_inv_sh_train(
                ins_inputs,
                closs_a,
                ctail_a,
                cshadow_a,
                keqv_a,
                closs_b,
                ctail_b,
                cshadow_b,
                keqv_b,
                closs_c,
                ctail_c,
                cshadow_c,
                keqv_c,
                path + path_env + algo + "_train",
            )

    # MARKET ENVIRONMENTS (NON-MARKOV)

    if PLOT_MARKET:

        # unique shuffled histories count
        plots.plot_shuffled_histories(path + "shuffled_histories")

        # plot market prices of several assets
        plots.market_prices(market_data_path, path)

        # number of observed days
        o_str = ["_" + str(o) for o in obs_days]
        inv_str = ["_A", "_B", "_C"]

        # environment: historical market agent training
        mkt_evs = (
            [mkt_invA_env] if only_invA else [mkt_invA_env, mkt_invB_env, mkt_invC_env]
        )

        i = 0
        for invs in mkt_evs:
            for mkt_inputs in [mkt_inputs_td3, mkt_inputs_sac]:

                algo = "_td3" if mkt_inputs == mkt_inputs_td3 else "_sac"

                e = 0
                for env in invs:

                    equity_inv_o = aggregate_data.mkt_obs_aggregate(
                        env, obs_days, action_days, gym_envs, mkt_inputs
                    )

                    (
                        reward,
                        closs,
                        ctail,
                        cshadow,
                        cmax,
                        keqv,
                        mstart,
                        elength,
                    ) = aggregate_data.mkt_obs_summary(mkt_inputs, equity_inv_o)

                    plots.plot_mkt_inv(
                        mkt_inputs,
                        reward,
                        closs,
                        ctail,
                        cshadow,
                        cmax,
                        keqv,
                        mstart,
                        elength,
                        obs_days,
                        path + "mkt_" + mkt_name[e] + algo + inv_str[i],
                        bin_size=250,
                    )
                    e += 1
            i += 1

        i = 0
        for invs in mkt_evs:
            e = 0
            for env in invs:

                equity_inv_a1 = aggregate_data.mkt_obs_aggregate(
                    env, obs_days, action_days, gym_envs, mkt_inputs_sac
                )

                equity_inv_a2 = aggregate_data.mkt_obs_aggregate(
                    env, obs_days, action_days, gym_envs, mkt_inputs_td3
                )

                (
                    reward_1,
                    closs_1,
                    ctail_1,
                    cshadow_1,
                    cmax_1,
                    keqv_1,
                    mstart_1,
                    elength_1,
                ) = aggregate_data.mkt_obs_summary(mkt_inputs_sac, equity_inv_a1)

                (
                    reward_2,
                    closs_2,
                    ctail_2,
                    cshadow_2,
                    cmax_2,
                    keqv_2,
                    mstart_2,
                    elength_2,
                ) = aggregate_data.mkt_obs_summary(mkt_inputs_td3, equity_inv_a2)

                plots.plot_mkt_inv_perf(
                    mkt_inputs_sac,
                    reward_1,
                    mstart_1,
                    elength_1,
                    reward_2,
                    mstart_2,
                    elength_2,
                    obs_days,
                    mkt_name[e],
                    path + "mkt_" + mkt_name[e] + "_perf" + inv_str[i],
                    bin_size=120,
                )

                plots.plot_mkt_inv_train(
                    mkt_inputs_sac,
                    closs_1,
                    ctail_1,
                    cshadow_1,
                    cmax_1,
                    keqv_1,
                    closs_2,
                    ctail_2,
                    cshadow_2,
                    cmax_2,
                    keqv_2,
                    obs_days,
                    mkt_name[e],
                    path + "mkt_" + mkt_name[e] + "_train" + inv_str[i],
                )
                e += 1

            i += 1

    # GUIDANCE ENVIRONMENTS (NON-MARKOV)

    if PLOT_GUIDANCE:
        pass
