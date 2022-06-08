"""
title:                  test_input_extra.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Responsible for conducting tests on user inputs for figure generation and
    market data acquisition.
"""

import sys

sys.path.append("./")

import os
from os import PathLike
from typing import Dict, List, Union

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


def figure_tests(
    path: Union[str, bytes, PathLike],
    market_data_path: Union[str, bytes, PathLike],
    gym_envs: Dict[str, list],
    add_envs: List[int],
    add_name: List[str],
    add_algos: List[int],
    add_loss: List[str],
    add_multi: List[int],
    add_inputs: dict,
    n_gambles: List[int],
    coin_keys: List[int],
    dice_keys: List[int],
    gbm_keys: List[int],
    mul_inputs_td3: dict,
    mul_inputs_sac: dict,
    dice_sh_keys: List[int],
    spitz_inputs_td3: dict,
    spitz_inputs_sac: dict,
    dice_sh_a_keys: List[int],
    dice_sh_b_keys: List[int],
    dice_sh_c_keys: List[int],
    ins_inputs_td3: dict,
    ins_inputs_sac: dict,
    obs_days: List[int],
    action_days: int,
    mkt_name: List[str],
    mkt_invA_env: List[int],
    mkt_invB_env: List[int],
    mkt_invC_env: List[int],
    only_invA: bool,
    mkt_inputs_td3: dict,
    mkt_inputs_sac: dict,
    PLOT_ADDITIVE: bool,
    PLOT_MULTIPLICATIVE: bool,
    PLOT_MULTIPLICATIVE_SH: bool,
    PLOT_MARKET: bool,
    PLOT_GUIDANCE: bool,
) -> None:
    """
    Conduct tests for final figure generation summarising agent training across
    all environments.

    Parameters:
        Refer to `./scripts/gen_figures.py` for input details.
    """
    assert isinstance(path, Union[str, bytes, PathLike]), ts
    assert (
        path[0:2] == "./" and path[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(market_data_path, Union[str, bytes, PathLike]), ts
    assert (
        market_data_path[0:2] == "./" and market_data_path[-1] == "/"
    ), "market_data_path file path must be in a sub-directory relative to main.py"

    multi_key, sh_key, market_key, _, _, _, _ = env_dynamics(gym_envs)

    assert isinstance(add_envs, list), tl
    assert (
        len(add_envs) == 4
    ), "add_envs must select exactly four (non-unique) environments"
    for env in add_envs:
        assert isinstance(env, int), ti
        assert (
            0 <= env < multi_key
        ), "additive environments keys must be between 0 and {}".format(multi_key - 1)
    assert isinstance(add_name, list), tl
    assert (
        len(add_name) == 4
    ), "add_name must select exactly four (non-unique) environments"
    for env in add_name:
        assert isinstance(env, str), ts
    assert isinstance(add_algos, list), tl
    assert (
        len(add_algos) == 2
    ), "add_algos must select exactly two (non-unique) algorithms"
    assert set(add_algos).issubset(
        set(["SAC", "TD3"])
    ), "add_algos algorithms must be a list containing only 'SAC' and/or 'TD3'"
    assert isinstance(add_loss, list), tl
    assert set(add_loss).issubset(
        set(["MSE", "HUB", "MAE", "HSC", "CAU", "TCAU", "CIM", "MSE2", "MSE4", "MSE6"])
    ), "add_loss critic losses must be a list containing 'MSE', 'HUB', 'MAE', 'HSC', 'CAU', 'TCAU', 'CIM', 'MSE2', 'MSE4', and/or 'MSE6'"
    assert isinstance(add_multi, list), tl
    for mstep in add_multi:
        assert isinstance(mstep, int), ti
        assert (
            mstep > 0
        ), "add_multi (multi-steps) must be at least single-step boostrap"

    assert isinstance(n_gambles, list), tl
    assert (
        len(n_gambles) == 3
    ), "n_gambles must select exactly three (non-unique) number of gambles"
    for n in n_gambles:
        assert isinstance(n, int), ti
        assert n > 0, "number of gambles must exceed 0"
    assert isinstance(coin_keys, list), tl
    assert (
        len(coin_keys) == 3
    ), "coin_keys must select exactly three (non-unique) number of coin envs"
    assert isinstance(dice_keys, list), tl
    assert (
        len(dice_keys) == 3
    ), "dice_keys must select exactly three (non-unique) number of dice envs"
    assert isinstance(gbm_keys, list), tl
    assert (
        len(gbm_keys) == 3
    ), "gbm_keys must select exactly three (non-unique) number of GBM envs"

    multi_envs = [coin_keys, dice_keys, gbm_keys]
    multi_name = ["coin", "dice", "GBM"]

    i = 0
    for envs in multi_envs:
        for env in envs:
            assert isinstance(env, int), ti
            assert (
                multi_key + 3 * i <= env < multi_key + 3 * (i + 1)
            ), "{} environment {} keys must be between {} and {}".format(
                multi_name[i], env, multi_key + 3 * i - 1, multi_key + 3 * (i + 1) - 1
            )
        i += 1

    assert isinstance(dice_sh_keys, list), tl
    assert (
        len(dice_sh_keys) == 2
    ), "dice_sh_keys must select exactly two (non-unique) safe haven environments"
    assert isinstance(dice_sh_keys[0], int), ti
    assert (
        dice_sh_keys[0] == multi_key + 3
    ), "dice_sh_keys environment 0 must be {}".format(multi_key + 3)
    assert isinstance(dice_sh_keys[1], int), ti
    assert dice_sh_keys[1] == sh_key, "dice_sh_keys environment 1 must be {}".format(
        sh_key
    )

    assert isinstance(dice_sh_a_keys, list), tl
    assert (
        len(dice_sh_a_keys) == 2
    ), "dice_sh_a_keys must select exactly two (non-unique) safe haven environments"
    assert isinstance(dice_sh_b_keys, list), tl
    assert (
        len(dice_sh_b_keys) == 2
    ), "dice_sh_b_keys must select exactly two (non-unique) safe haven environments"
    assert isinstance(dice_sh_c_keys, list), tl
    assert (
        len(dice_sh_c_keys) == 2
    ), "dice_sh_c_keys must select exactly two (non-unique) safe haven environments"

    dice_sh_envs = [dice_sh_a_keys, dice_sh_b_keys, dice_sh_c_keys]
    dice_sh_name = ["dice_sh_a", "dice_sh_b", "dice_sh_c"]

    i = 0
    for env in dice_sh_envs:
        assert isinstance(env[0], int), ti
        assert (
            multi_key + 3 <= env[0] < multi_key + 3 * 2
        ), "{} environments keys must be between {} and {}".format(
            dice_sh_name[i], multi_key + 3 - 1, multi_key + 3 * 2 - 1
        )
        assert isinstance(env[1], int), ti
        assert (
            sh_key + 1 <= env[1] < market_key
        ), "{} environments keys must be between {} and {}".format(
            dice_sh_name[i], sh_key + 1, market_key - 1
        )
        i += 1

    assert isinstance(obs_days, list), tl
    assert (
        len(obs_days) == 3
    ), "obs_days must select exactly three (non-unique) number of days"
    for od in obs_days:
        assert isinstance(od, int), ti
        assert od > 0, "number of observed days must exceed 0"
    assert isinstance(action_days, int), ti
    assert action_days > 0, gt0

    assert isinstance(mkt_name, list), tl
    assert len(mkt_name) >= 1, "mkt_name must contain at least one market name"
    assert len(mkt_name) == len(
        set(mkt_name)
    ), "mkt_name must contain only unique elements"
    for name in mkt_name:
        assert isinstance(name, str), ts

    assert isinstance(mkt_invA_env, list), tl
    assert (
        len(mkt_invA_env) >= 1
    ), "mkt_invA_env must contain at least one market environments"
    assert len(mkt_invA_env) == len(
        set(mkt_invA_env)
    ), "mkt_invA_env must contain only unique elements"
    assert isinstance(mkt_invB_env, list), tl
    assert (
        len(mkt_invB_env) >= 1
    ), "mkt_invB_env must contain at least one market environments"
    assert len(mkt_invB_env) == len(
        set(mkt_invB_env)
    ), "mkt_invB_env must contain only unique elements"
    assert isinstance(mkt_invC_env, list), tl
    assert (
        len(mkt_invC_env) >= 1
    ), "mkt_invC_env must contain at least one market environments"
    assert len(mkt_invC_env) == len(
        set(mkt_invC_env)
    ), "mkt_invC_env must contain only unique elements"

    mkt_envs = [mkt_invA_env, mkt_invB_env, mkt_invC_env]
    mkt_envs_list = ["mkt_a", "mkt_b", "mkt_c"]

    i = 0
    for envs in mkt_envs:
        for env in envs:
            assert isinstance(env, int), ti
            assert (
                market_key <= env
            ), "{} environment {} key must be greater than or equal to {}".format(
                mkt_envs_list[i], env, market_key
            )
        i += 1

    assert len(mkt_name) >= len(
        mkt_invA_env
    ), "must have at least as many names as environments"
    assert len(mkt_name) >= len(
        mkt_invB_env
    ), "must have at least as many names as environments"
    assert len(mkt_name) >= len(
        mkt_invC_env
    ), "must have at least as many names as environments"

    assert isinstance(only_invA, bool), tb

    assert isinstance(add_inputs, dict), td
    assert isinstance(mul_inputs_td3, dict), td
    assert isinstance(mul_inputs_sac, dict), td
    assert isinstance(spitz_inputs_td3, dict), td
    assert isinstance(spitz_inputs_sac, dict), td
    assert isinstance(ins_inputs_td3, dict), td
    assert isinstance(ins_inputs_sac, dict), td
    assert isinstance(mkt_inputs_td3, dict), td
    assert isinstance(mkt_inputs_sac, dict), td

    inputs = [
        add_inputs,
        mul_inputs_td3,
        mul_inputs_sac,
        spitz_inputs_td3,
        spitz_inputs_sac,
        ins_inputs_td3,
        ins_inputs_sac,
        mkt_inputs_td3,
        mkt_inputs_sac,
    ]

    for params in inputs:
        if params != add_inputs:
            assert isinstance(params["algo"], str), ts
            assert (
                params["algo"] == "SAC" or params["algo"] == "TD3"
            ), "algorithm must be either 'SAC' or 'TD3'"

        assert isinstance(params["n_trials"], (float, int)), tfi
        assert int(params["n_trials"]) >= 1, gte1
        assert isinstance(params["n_cumsteps"], (float, int)), tfi
        assert set(list(str(params["n_cumsteps"])[2:])).issubset(
            set(["0", "."])
        ), "n_cumsteps must consist of only 2 leading non-zero digits"
        assert int(params["n_cumsteps"]) >= 1, gte1
        assert isinstance(params["eval_freq"], (float, int)), tfi
        assert int(params["eval_freq"]) >= 1, gte1
        assert int(params["eval_freq"]) <= int(
            params["n_cumsteps"]
        ), "eval_freq must be less than or equal to n_cumsteps"
        assert isinstance(params["n_eval"], (float, int)), tfi
        assert int(params["n_eval"]) >= 1, gte1
        assert isinstance(params["buffer"], (float, int)), tfi
        assert set(list(str(params["buffer"])[2:])).issubset(
            set(["0", "."])
        ), "buffer must consist of only 2 leading non-zero digits"
        assert int(params["buffer"]) >= 1, gte1
        assert (
            params["buffer"] >= params["n_cumsteps"]
        ), "buffer must be greater than or equal to n_cumsteps"
        assert (
            params["critic_mean_type"] == "E"
        ), "critic_mean_type must be 'E' ('S' not currently possible)"
        assert (
            params["s_dist"] == "N"
            or params["s_dist"] == "L"
            or (params["algo"] == "SAC" and params["s_dist"] == "MVN")
        ), "s_dist must be either 'N' (normal=Gaussian), 'L' (2x exponential=Laplace) and for SAC only 'MVN' (multi-variate normal)"

    assert spitz_inputs_td3["n_trials"] == spitz_inputs_sac["n_trials"]
    assert spitz_inputs_td3["n_cumsteps"] == spitz_inputs_sac["n_cumsteps"]
    assert spitz_inputs_td3["eval_freq"] == spitz_inputs_sac["eval_freq"]
    assert spitz_inputs_td3["n_eval"] == spitz_inputs_sac["n_eval"]

    assert mkt_inputs_td3["n_trials"] == mkt_inputs_sac["n_trials"]
    assert mkt_inputs_td3["n_cumsteps"] == mkt_inputs_sac["n_cumsteps"]
    assert mkt_inputs_td3["eval_freq"] == mkt_inputs_sac["eval_freq"]
    assert mkt_inputs_td3["n_eval"] == mkt_inputs_sac["n_eval"]

    assert os.path.isfile(
        market_data_path + "stooq_usei.pkl"
    ), "stooq_usei.pkl not generated or found in {}".format(market_data_path)
    assert os.path.isfile(
        market_data_path + "stooq_dji.pkl"
    ), "stooq_dji.pkl not generated or found in {}".format(market_data_path)
    assert os.path.isfile(
        market_data_path + "stooq_major.pkl"
    ), "stooq_major.pkl not generated or found in {}".format(market_data_path)

    assert isinstance(PLOT_ADDITIVE, int), ti
    assert PLOT_ADDITIVE == 0 or PLOT_ADDITIVE == 1, "must be 0 or 1"
    assert isinstance(PLOT_MULTIPLICATIVE, int), ti
    assert PLOT_MULTIPLICATIVE == 0 or PLOT_MULTIPLICATIVE == 1, "must be 0 or 1"
    assert isinstance(PLOT_MULTIPLICATIVE_SH, int), ti
    assert PLOT_MULTIPLICATIVE_SH == 0 or PLOT_MULTIPLICATIVE_SH == 1, "must be 0 or 1"
    assert isinstance(PLOT_MARKET, int), ti
    assert PLOT_MARKET == 0 or PLOT_MARKET == 1, "must be 0 or 1"
    assert isinstance(PLOT_GUIDANCE, int), ti
    assert PLOT_GUIDANCE == 0 or PLOT_GUIDANCE == 1, "must be 0 or 1"

    print("Figure Generation Tests: Passed")


def market_data_tests(
    start: str,
    end: str,
    stooq: Dict[str, list],
    path: Union[str, bytes, PathLike],
    price_type: str,
) -> None:
    """
    Conduct tests prior to scrapping historical financial data from Stooq.

    Parameters:
        star: start data
        end: end date
        stooq: market data to download from Stooq
        path: directory to save data
        price_type: type of market price to utilise
    """
    assert isinstance(start, str), ts
    assert start[4] == start[7] == "-", "data format must be YYYY-MM-DD"
    assert isinstance(end, str), ts
    assert end[4] == end[7] == "-", "data format must be YYYY-MM-DD"

    y_s, m_s, d_s = start[:4], start[5:7], start[-2:]
    y_e, m_e, d_e = end[:4], end[5:7], end[-2:]

    assert isinstance(int(y_s), int), ti
    assert isinstance(int(m_s), int), ti
    assert isinstance(int(d_s), int), ti
    assert isinstance(int(y_e), int), ti
    assert isinstance(int(m_e), int), ti
    assert isinstance(int(d_e), int), ti

    assert 1900 < int(y_s), "start year should be post 1900"
    assert 0 < int(m_s) <= 12, "only 12 months in year"
    assert 0 < int(d_s) <= 31, "maximum 31 days per month"
    assert 1900 < int(y_s), "end year should be post 1900"
    assert 0 < int(m_e) <= 12, "only 12 months in year"
    assert 0 < int(d_e) <= 31, "maximum 31 days per month"

    assert int(y_e + m_e + d_e) > int(
        y_s + m_s + d_s
    ), "end date must exceed start date"

    assert isinstance(stooq, dict), td

    for x in stooq:
        mkt = stooq[str(x)]
        assert isinstance(mkt, list), tl
        assert len(mkt) == 2, "mkt must have at least one number symbol"
        assert isinstance(mkt[0], str), ts
        assert isinstance(mkt[1], list), tl
        assert len(mkt[1]) == len(set(mkt[1])), "mkt must contain only unique elements"
        assert all(isinstance(a, str) for a in mkt[1]), ts

    assert isinstance(path, Union[str, bytes, PathLike])
    assert (
        path[0:2] == "./" and path[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert isinstance(price_type, str)
    assert (
        price_type.capitalize() == "Open" or "High" or "Low" or "Close"
    ), "price_type must be one of Open, High, Low, or Close"

    print("Market Import Tests: Passed")
