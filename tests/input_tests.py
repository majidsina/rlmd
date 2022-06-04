"""
title:                  input_tests.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Responsible for conducting tests on all user inputs throughout the codebase.
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


def coin_flip_tests(
    INVESTORS: float,
    HORIZON: float,
    TOP: float,
    VALUE_0: float,
    UP_PROB: float,
    UP_R: float,
    DOWN_R: float,
    ASYM_LIM: float,
    VRAM: bool,
    path_results: Union[str, bytes, PathLike],
    path_figs: Union[str, bytes, PathLike],
    l0_l: float,
    l0_h: float,
    l0_i: float,
    l1_l: float,
    l1_h: float,
    l1_i: float,
    s2_l: float,
    s2_h: float,
    s2_i: float,
    r2_l: float,
    r2_h: float,
    r2_i: float,
    s3_l: float,
    s3_h: float,
    s3_i: float,
    r3_l: float,
    r3_h: float,
    r3_i: float,
    ru_l: float,
    ru_h: float,
    ru_i: float,
    rd_l: float,
    rd_h: float,
    rd_i: float,
    pu_l: float,
    pu_h: float,
    pu_i: float,
) -> None:
    """
    Conduct tests on coin flip optimal leverage experiment.

    Parameters:
        Refer to `./lev/coin_flip.py` for input details.
    """
    assert isinstance(INVESTORS, (float, int)), tfi
    assert INVESTORS > 0
    assert isinstance(HORIZON, (float, int)), tfi
    assert HORIZON > 0
    assert isinstance(TOP, (float, int)), tfi
    assert TOP > 0
    assert isinstance(VALUE_0, (float, int)), tfi
    assert VALUE_0 > 0
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert -1 <= DOWN_R < 0
    assert isinstance(ASYM_LIM, float), tfi
    assert 0 < ASYM_LIM < 1e-3
    assert isinstance(VRAM, bool), tb

    assert isinstance(path_results, Union[str, bytes, PathLike]), ts
    assert (
        path_results[0:2] == "./" and path_results[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(path_figs, Union[str, bytes, PathLike]), ts
    assert (
        path_figs[0:2] == "./" and path_figs[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert isinstance(l0_i, (float, int)), tfi
    assert isinstance(l0_l, (float, int)), tfi
    assert isinstance(l0_h, (float, int)), tfi
    assert l0_i > 0, "increment must by positive"
    assert l0_l <= l0_h

    assert isinstance(l1_i, (float, int)), tfi
    assert isinstance(l1_l, (float, int)), tfi
    assert isinstance(l1_h, (float, int)), tfi
    assert l1_i > 0, "increment must by positive"
    assert l1_l <= l1_h

    assert isinstance(s2_i, float), tfi
    assert isinstance(s2_l, float), tfi
    assert isinstance(s2_h, float), tfi
    assert s2_i > 0, "increment must by positive"
    assert 0 <= s2_l <= s2_h < 1
    assert isinstance(r2_i, float), tfi
    assert isinstance(r2_l, float), tfi
    assert isinstance(r2_h, float), tfi
    assert r2_i > 0, "increment must by positive"
    assert 0 <= r2_l <= r2_h < 1

    assert isinstance(s3_i, float), tfi
    assert isinstance(s3_l, float), tfi
    assert isinstance(s3_h, float), tfi
    assert s3_i > 0, "increment must by positive"
    assert 0 <= s3_l <= s3_h < 1
    assert isinstance(r3_i, float), tfi
    assert isinstance(r3_l, float), tfi
    assert isinstance(r3_h, float), tfi
    assert r3_i > 0, "increment must by positive"
    assert 0 <= r3_l <= r3_h < 1

    assert isinstance(ru_i, (float, int)), tfi
    assert isinstance(ru_l, (float, int)), tfi
    assert isinstance(ru_h, (float, int)), tfi
    assert ru_i > 0, "increment must by positive"
    assert 0 < ru_l <= ru_h
    assert isinstance(rd_i, (float, int)), tfi
    assert isinstance(rd_l, (float, int)), tfi
    assert isinstance(rd_h, (float, int)), tfi
    assert rd_i > 0, "increment must by positive"
    assert 0 < rd_l <= rd_h
    assert isinstance(pu_i, float), tf
    assert isinstance(pu_l, float), tf
    assert isinstance(pu_h, float), tf
    assert pu_i > 0, "increment must by positive"
    assert 0 < pu_l <= pu_h < 1
    assert (
        int(pu_h / pu_i + 1) - int(pu_l / pu_i) == 3
    ), "3 unique probability increments are required"

    print("Coin Flip Experiment Tests: Passed")


def dice_roll_tests(
    INVESTORS: float,
    HORIZON: float,
    TOP: float,
    VALUE_0: float,
    UP_PROB: float,
    DOWN_PROB: float,
    UP_R: float,
    DOWN_R: float,
    MID_R: float,
    ASYM_LIM: float,
    VRAM: bool,
    path_results: Union[str, bytes, PathLike],
    path_figs: Union[str, bytes, PathLike],
    l0_l: float,
    l0_h: float,
    l0_i: float,
    l1_l: float,
    l1_h: float,
    l1_i: float,
    s2_l: float,
    s2_h: float,
    s2_i: float,
    r2_l: float,
    r2_h: float,
    r2_i: float,
    s3_l: float,
    s3_h: float,
    s3_i: float,
    r3_l: float,
    r3_h: float,
    r3_i: float,
) -> None:
    """
    Conduct tests on dice roll optimal leverage experiment.

    Parameters:
        Refer to `./lev/dice_roll.py` for input details.
    """
    assert isinstance(INVESTORS, (float, int)), tfi
    assert INVESTORS > 0
    assert isinstance(HORIZON, (float, int)), tfi
    assert HORIZON > 0
    assert isinstance(TOP, (float, int)), tfi
    assert TOP > 0
    assert isinstance(VALUE_0, (float, int)), tfi
    assert VALUE_0 > 0
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(DOWN_PROB, (float, int)), tfi
    assert 0 < DOWN_PROB < 1
    assert UP_PROB + DOWN_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert -1 <= DOWN_R < 0
    assert isinstance(MID_R, (float, int)), tfi
    assert UP_R > MID_R > DOWN_R
    assert isinstance(ASYM_LIM, float), tfi
    assert 0 < ASYM_LIM < 1e-3
    assert isinstance(VRAM, bool), tb

    assert isinstance(path_results, Union[str, bytes, PathLike]), ts
    assert (
        path_results[0:2] == "./" and path_results[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(path_figs, Union[str, bytes, PathLike]), ts
    assert (
        path_figs[0:2] == "./" and path_figs[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert isinstance(l0_i, (float, int)), tfi
    assert isinstance(l0_l, (float, int)), tfi
    assert isinstance(l0_h, (float, int)), tfi
    assert l0_i > 0, "increment must by positive"
    assert l0_l <= l0_h

    assert isinstance(l1_i, (float, int)), tfi
    assert isinstance(l1_l, (float, int)), tfi
    assert isinstance(l1_h, (float, int)), tfi
    assert l1_i > 0, "increment must by positive"
    assert l1_l <= l1_h

    assert isinstance(s2_i, float), tfi
    assert isinstance(s2_l, float), tfi
    assert isinstance(s2_h, float), tfi
    assert s2_i > 0, "increment must by positive"
    assert 0 <= s2_l <= s2_h < 1
    assert isinstance(r2_i, float), tfi
    assert isinstance(r2_l, float), tfi
    assert isinstance(r2_h, float), tfi
    assert r2_i > 0, "increment must by positive"
    assert 0 <= r2_l <= r2_h < 1

    assert isinstance(s3_i, float), tfi
    assert isinstance(s3_l, float), tfi
    assert isinstance(s3_h, float), tfi
    assert s3_i > 0, "increment must by positive"
    assert 0 <= s3_l <= s3_h < 1
    assert isinstance(r3_i, float), tfi
    assert isinstance(r3_l, float), tfi
    assert isinstance(r3_h, float), tfi
    assert r3_i > 0, "increment must by positive"
    assert 0 <= r3_l <= r3_h < 1

    print("Dice Roll Experiment Tests: Passed")


def dice_roll_sh_tests(
    INVESTORS: float,
    HORIZON: float,
    TOP: float,
    VALUE_0: float,
    UP_PROB: float,
    DOWN_PROB: float,
    UP_R: float,
    DOWN_R: float,
    MID_R: float,
    SH_UP_R: float,
    SH_DOWN_R: float,
    SH_MID_R: float,
    VRAM: bool,
    path_results: Union[str, bytes, PathLike],
    path_figs: Union[str, bytes, PathLike],
    l0_l: float,
    l0_h: float,
    l0_i: float,
    l1_l: float,
    l1_h: float,
    l1_i: float,
) -> None:
    """
    Conduct tests on dice roll (safe haven) optimal leverage experiment.

    Parameters:
        Refer to `./lev/dice_roll_sh.py` for input details.
    """
    assert isinstance(INVESTORS, (float, int)), tfi
    assert INVESTORS > 0
    assert isinstance(HORIZON, (float, int)), tfi
    assert HORIZON > 0
    assert isinstance(TOP, (float, int)), tfi
    assert TOP > 0
    assert isinstance(VALUE_0, (float, int)), tfi
    assert VALUE_0 > 0
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(DOWN_PROB, (float, int)), tfi
    assert 0 < DOWN_PROB < 1
    assert UP_PROB + DOWN_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert -1 <= DOWN_R < 0
    assert isinstance(MID_R, (float, int)), tfi
    assert UP_R > MID_R > DOWN_R
    assert isinstance(SH_UP_R, (float, int)), tfi
    assert -1 <= SH_UP_R < 0
    assert isinstance(SH_DOWN_R, (float, int)), tfi
    assert SH_DOWN_R > 0
    assert isinstance(SH_MID_R, (float, int)), tfi
    assert -1 <= SH_MID_R < SH_DOWN_R
    assert SH_MID_R >= SH_UP_R
    assert isinstance(VRAM, bool), tb

    assert isinstance(path_results, Union[str, bytes, PathLike]), ts
    assert (
        path_results[0:2] == "./" and path_results[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(path_figs, Union[str, bytes, PathLike]), ts
    assert (
        path_figs[0:2] == "./" and path_figs[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert isinstance(l0_i, (float, int)), tfi
    assert isinstance(l0_l, (float, int)), tfi
    assert isinstance(l0_h, (float, int)), tfi
    assert l0_i > 0, "increment must by positive"
    assert l0_l <= l0_h

    assert isinstance(l1_i, (float, int)), tfi
    assert isinstance(l1_l, (float, int)), tfi
    assert isinstance(l1_h, (float, int)), tfi
    assert l1_i > 0, "increment must by positive"
    assert l1_l <= l1_h

    print("Dice Roll (Safe Haven) Experiment Tests: Passed")


def gbm_tests(
    INVESTORS: float,
    HORIZON: float,
    TOP: float,
    VALUE_0: float,
    drift: List[float],
    vol: List[float],
    name: List[str],
    VRAM: bool,
    path_results: Union[str, bytes, PathLike],
    path_figs: Union[str, bytes, PathLike],
    l0_l: List[float],
    l0_h: List[float],
    l0_i: List[float],
    l1_l: List[float],
    l1_h: List[float],
    l1_i: List[float],
) -> None:
    """
    Conduct tests on GBM optimal leverage experiment.

    Parameters:
        Refer to `./lev/gbm.py` for input details.
    """
    assert isinstance(INVESTORS, (float, int)), tfi
    assert INVESTORS > 0
    assert isinstance(HORIZON, (float, int)), tfi
    assert HORIZON > 0
    assert isinstance(TOP, (float, int)), tfi
    assert TOP > 0
    assert isinstance(VALUE_0, (float, int)), tfi
    assert VALUE_0 > 0
    assert isinstance(VRAM, bool), tb

    assert isinstance(path_results, Union[str, bytes, PathLike]), ts
    assert (
        path_results[0:2] == "./" and path_results[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(path_figs, Union[str, bytes, PathLike]), ts
    assert (
        path_figs[0:2] == "./" and path_figs[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert isinstance(drift, list), tl
    assert isinstance(vol, list), tl
    assert isinstance(name, list), tl
    assert len(name) == len(set(name)), "must contain only unique elements"

    assert isinstance(l0_i, list), tl
    assert isinstance(l0_l, list), tl
    assert isinstance(l0_h, list), tl

    assert isinstance(l1_i, list), tl
    assert isinstance(l1_l, list), tl
    assert isinstance(l1_h, list), tl

    assert (
        len(drift)
        == len(vol)
        == len(name)
        == len(l0_i)
        == len(l0_l)
        == len(l0_h)
        == len(l1_i)
        == len(l1_l)
        == len(l1_h)
    ), "all input lists must be of equal length"

    assert len(drift) > 0, "must contain at least one environment"

    for x in range(len(drift)):
        assert isinstance(drift[x], (float, int)), tfi
        assert isinstance(vol[x], (float, int)), tfi
        assert vol[x] > 0, "volatility {} must be non-zero and non-negative".format(
            vol[x]
        )
        assert isinstance(name[x], str), ts

        if drift[x] < vol[x] ** 2 / 2:
            print(
                "Optimal leverage for situations where u < s^2 / 2 (u={}, s={}) are not stable.".format(
                    drift[x], vol[x]
                )
            )

        assert isinstance(l0_i[x], (float, int)), tfi
        assert isinstance(l0_l[x], (float, int)), tfi
        assert isinstance(l0_h[x], (float, int)), tfi
        assert l0_i[x] > 0, "increment must by positive"
        assert l0_l[x] <= l0_h[x]

        assert isinstance(l1_i[x], (float, int)), tfi
        assert isinstance(l1_l[x], (float, int)), tfi
        assert isinstance(l1_h[x], (float, int)), tfi
        assert l1_i[x] > 0, "increment must by positive"
        assert l1_l[x] <= l1_h[x]

    print("GBM Experiment Tests: Passed")


def levarge_tests(
    INVESTORS: float,
    HORIZON: float,
    TOP: float,
    VALUE_0: float,
    ASYM_LIM: float,
    VRAM: bool,
    path_results: Union[str, bytes, PathLike],
    path_figs: Union[str, bytes, PathLike],
    COIN_UP_PROB: float,
    COIN_UP_R: float,
    COIN_DOWN_R: float,
    DICE_UP_PROB: float,
    DICE_DOWN_PROB: float,
    DICE_UP_R: float,
    DICE_DOWN_R: float,
    DICE_MID_R: float,
    DICESH_UP_PROB: float,
    DICESH_DOWN_PROB: float,
    DICESH_UP_R: float,
    DICESH_DOWN_R: float,
    DICESH_MID_R: float,
    DICESH_SH_UP_R: float,
    DICESH_SH_DOWN_R: float,
    DICESH_SH_MID_R: float,
    l0_l: float,
    l0_h: float,
    l0_i: float,
    l1_l: float,
    l1_h: float,
    l1_i: float,
    s2_l: float,
    s2_h: float,
    s2_i: float,
    r2_l: float,
    r2_h: float,
    r2_i: float,
    s3_l: float,
    s3_h: float,
    s3_i: float,
    r3_l: float,
    r3_h: float,
    r3_i: float,
    ru_l: float,
    ru_h: float,
    ru_i: float,
    rd_l: float,
    rd_h: float,
    rd_i: float,
    pu_l: float,
    pu_h: float,
    pu_i: float,
    GBM_DRIFT: List[float],
    GBM_VOL: List[float],
    GBM_NAME: List[str],
    GBM_l0_l: List[float],
    GBM_l0_h: List[float],
    GBM_l0_i: List[float],
    GBM_l1_l: List[float],
    GBM_l1_h: List[float],
    GBM_l1_i: List[float],
) -> None:
    """
    Conduct tests on leverage test experiments.

    Parameters:
        Refer to `./tests/lev_tests.py` for input details.
    """
    assert isinstance(INVESTORS, (float, int)), tfi
    assert INVESTORS > 0
    assert isinstance(HORIZON, (float, int)), tfi
    assert HORIZON > 0
    assert isinstance(TOP, (float, int)), tfi
    assert TOP > 0
    assert isinstance(VALUE_0, (float, int)), tfi
    assert VALUE_0 > 0
    assert isinstance(ASYM_LIM, float), tfi
    assert 0 < ASYM_LIM < 1e-3
    assert isinstance(VRAM, bool), tb

    assert isinstance(path_results, Union[str, bytes, PathLike]), ts
    assert (
        path_results[0:2] == "./" and path_results[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"
    assert isinstance(path_figs, Union[str, bytes, PathLike]), ts
    assert (
        path_figs[0:2] == "./" and path_figs[-1] == "/"
    ), "file path must be in a sub-directory relative to main.py"

    assert 0 < COIN_UP_PROB < 1
    assert isinstance(COIN_UP_R, (float, int)), tfi
    assert COIN_UP_R > 0
    assert isinstance(COIN_DOWN_R, (float, int)), tfi
    assert -1 <= COIN_DOWN_R < 0

    assert isinstance(DICE_UP_PROB, (float, int)), tfi
    assert 0 < DICE_UP_PROB < 1
    assert isinstance(DICE_DOWN_PROB, (float, int)), tfi
    assert 0 < DICE_DOWN_PROB < 1
    assert DICE_UP_PROB + DICE_DOWN_PROB < 1
    assert isinstance(DICE_UP_R, (float, int)), tfi
    assert DICE_UP_R > 0
    assert isinstance(DICE_DOWN_R, (float, int)), tfi
    assert -1 <= DICE_DOWN_R < 0
    assert isinstance(DICE_MID_R, (float, int)), tfi
    assert DICE_UP_R > DICE_MID_R > DICE_DOWN_R

    assert isinstance(DICESH_UP_PROB, (float, int)), tfi
    assert 0 < DICESH_UP_PROB < 1
    assert isinstance(DICESH_DOWN_PROB, (float, int)), tfi
    assert 0 < DICESH_DOWN_PROB < 1
    assert DICESH_UP_PROB + DICESH_DOWN_PROB < 1
    assert isinstance(DICESH_UP_R, (float, int)), tfi
    assert DICESH_UP_R > 0
    assert isinstance(DICESH_DOWN_R, (float, int)), tfi
    assert -1 <= DICESH_DOWN_R < 0
    assert isinstance(DICESH_MID_R, (float, int)), tfi
    assert DICESH_UP_R > DICESH_MID_R > DICESH_DOWN_R
    assert isinstance(DICESH_SH_UP_R, (float, int)), tfi
    assert -1 <= DICESH_SH_UP_R < 0
    assert isinstance(DICESH_SH_DOWN_R, (float, int)), tfi
    assert DICESH_SH_DOWN_R > 0
    assert isinstance(DICESH_SH_MID_R, (float, int)), tfi
    assert -1 <= DICESH_SH_MID_R < DICESH_SH_DOWN_R
    assert DICESH_SH_MID_R >= DICESH_SH_UP_R

    assert isinstance(l0_i, (float, int)), tfi
    assert isinstance(l0_l, (float, int)), tfi
    assert isinstance(l0_h, (float, int)), tfi
    assert l0_i > 0, "increment must by positive"
    assert l0_l <= l0_h

    assert isinstance(l1_i, (float, int)), tfi
    assert isinstance(l1_l, (float, int)), tfi
    assert isinstance(l1_h, (float, int)), tfi
    assert l1_i > 0, "increment must by positive"
    assert l1_l <= l1_h

    assert isinstance(s2_i, float), tfi
    assert isinstance(s2_l, float), tfi
    assert isinstance(s2_h, float), tfi
    assert s2_i > 0, "increment must by positive"
    assert 0 <= s2_l <= s2_h < 1
    assert isinstance(r2_i, float), tfi
    assert isinstance(r2_l, float), tfi
    assert isinstance(r2_h, float), tfi
    assert r2_i > 0, "increment must by positive"
    assert 0 <= r2_l <= r2_h < 1

    assert isinstance(s3_i, float), tfi
    assert isinstance(s3_l, float), tfi
    assert isinstance(s3_h, float), tfi
    assert s3_i > 0, "increment must by positive"
    assert 0 <= s3_l <= s3_h < 1
    assert isinstance(r3_i, float), tfi
    assert isinstance(r3_l, float), tfi
    assert isinstance(r3_h, float), tfi
    assert r3_i > 0, "increment must by positive"
    assert 0 <= r3_l <= r3_h < 1

    assert isinstance(ru_i, (float, int)), tfi
    assert isinstance(ru_l, (float, int)), tfi
    assert isinstance(ru_h, (float, int)), tfi
    assert ru_i > 0, "increment must by positive"
    assert 0 < ru_l <= ru_h
    assert isinstance(rd_i, (float, int)), tfi
    assert isinstance(rd_l, (float, int)), tfi
    assert isinstance(rd_h, (float, int)), tfi
    assert rd_i > 0, "increment must by positive"
    assert 0 < rd_l <= rd_h
    assert isinstance(pu_i, float), tf
    assert isinstance(pu_l, float), tf
    assert isinstance(pu_h, float), tf
    assert pu_i > 0, "increment must by positive"
    assert 0 < pu_l <= pu_h < 1
    assert (
        int(pu_h / pu_i + 1) - int(pu_l / pu_i) == 3
    ), "3 unique probability increments are required"

    assert isinstance(GBM_DRIFT, list), tl
    assert isinstance(GBM_VOL, list), tl
    assert isinstance(GBM_NAME, list), tl
    assert len(GBM_NAME) == len(set(GBM_NAME)), "must contain only unique elements"

    assert isinstance(GBM_l0_i, list), tl
    assert isinstance(GBM_l0_l, list), tl
    assert isinstance(GBM_l0_h, list), tl

    assert isinstance(GBM_l1_i, list), tl
    assert isinstance(GBM_l1_l, list), tl
    assert isinstance(GBM_l1_h, list), tl

    assert (
        len(GBM_DRIFT)
        == len(GBM_VOL)
        == len(GBM_NAME)
        == len(GBM_l0_i)
        == len(GBM_l0_l)
        == len(GBM_l0_h)
        == len(GBM_l1_i)
        == len(GBM_l1_l)
        == len(GBM_l1_h)
    ), "all input lists must be of equal length"

    for x in range(len(GBM_DRIFT)):
        assert isinstance(GBM_DRIFT[x], (float, int)), tfi
        assert isinstance(GBM_VOL[x], (float, int)), tfi
        assert GBM_VOL[x] > 0, "volatility {} must be non-zero and non-negative".format(
            GBM_VOL[x]
        )
        assert isinstance(GBM_NAME[x], str), ts

        if GBM_DRIFT[x] < GBM_VOL[x] ** 2 / 2:
            print(
                "Optimal leverage for situations where u < s^2 / 2 (u={}, s={}) are not stable.".format(
                    GBM_DRIFT[x], GBM_VOL[x]
                )
            )

        assert isinstance(GBM_l0_i[x], (float, int)), tfi
        assert isinstance(GBM_l0_l[x], (float, int)), tfi
        assert isinstance(GBM_l0_h[x], (float, int)), tfi
        assert GBM_l0_i[x] > 0, "increment must by positive"
        assert GBM_l0_l[x] <= GBM_l0_h[x]

        assert isinstance(GBM_l1_i[x], (float, int)), tfi
        assert isinstance(GBM_l1_l[x], (float, int)), tfi
        assert isinstance(GBM_l1_h[x], (float, int)), tfi
        assert GBM_l1_i[x] > 0, "increment must by positive"
        assert GBM_l1_l[x] <= GBM_l1_h[x]

    print("Leverage Tests: Passed")


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


def multi_env_tests(
    name: str,
    MAX_VALUE: float,
    INITIAL_PRICE: float,
    INITIAL_VALUE: float,
    MIN_VALUE_RATIO: float,
    MAX_VALUE_RATIO: float,
    DISCRETE_RETURNS: bool,
    MAX_ABS_ACTION: float,
    MIN_REWARD: float,
    MIN_RETURN: float,
    MAX_RETURN: float,
    MIN_WEIGHT: float,
) -> None:
    """
    Conduct tests for multiplicative environment initialisation.

    Parameters:
        Refer to financial environments in `./envs` for input details.
    """
    assert isinstance(name, str), ts

    assert isinstance(MAX_VALUE, (float, int)), tfi
    assert (
        0 < MAX_VALUE < 1e37
    ), "MAX_VALUE must be less than float32 (single-precision) limit 1e38"
    assert isinstance(INITIAL_PRICE, (float, int)), tfi
    assert INITIAL_PRICE > 0
    assert isinstance(INITIAL_VALUE, (float, int)), tfi
    assert INITIAL_VALUE > 0
    assert isinstance(MIN_VALUE_RATIO, (float, int)), tfi
    assert 1 / MAX_VALUE <= MIN_VALUE_RATIO < 1
    assert isinstance(MAX_VALUE_RATIO, (float, int)), tfi
    assert MIN_VALUE_RATIO < MAX_VALUE_RATIO

    assert isinstance(DISCRETE_RETURNS, bool), tb

    assert isinstance(MAX_ABS_ACTION, (float, int)), tfi
    assert MAX_ABS_ACTION > 0
    assert isinstance(MIN_REWARD, (float, int)), tfi
    assert isinstance(MIN_RETURN, (float, int)), tfi

    if DISCRETE_RETURNS == True:
        assert MIN_REWARD > 0
        assert MIN_RETURN > -1
    else:
        assert (
            MIN_REWARD > 1e-4
        ), "MIN_REWARD must be sufficiently large to prevent critic loss divergence"
        assert (
            MIN_RETURN > -1e38
        ), "MIN_RETURN must be greater than float32 (single-precision) limit -1e38"

    assert isinstance(MAX_RETURN, (float, int)), tfi
    assert MAX_RETURN > MIN_RETURN
    assert isinstance(MIN_WEIGHT, (float, int)), tfi
    assert MIN_WEIGHT > 0

    MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
    MIN_VALUE_SPACE = min(
        MIN_VALUE / INITIAL_VALUE - 1, MIN_REWARD / MAX_VALUE, MIN_VALUE_RATIO
    )

    assert MAX_VALUE > MIN_VALUE
    assert INITIAL_PRICE > MIN_VALUE
    assert INITIAL_VALUE > MIN_VALUE
    assert MAX_VALUE_RATIO > MIN_VALUE_SPACE

    print("{} Environment Tests: Passed".format(name))


def coin_flip_env_tests(UP_PROB: float, UP_R: float, DOWN_R: float) -> None:
    """
    Conduct tests for coin flip environment initialisation.

    Parameters:
        Refer to `./envs/coin_flip_envs.py` for input details.
    """
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert DOWN_R < 0

    print("Coin Flip Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def dice_roll_env_tests(
    UP_PROB: float,
    DOWN_PROB: float,
    UP_R: float,
    DOWN_R: float,
    MID_R: float,
) -> None:
    """
    Conduct tests for dice roll environment initialisation.

    Parameters:
        Refer to `./envs/dice_roll_envs.py` for input details.
    """
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(DOWN_PROB, (float, int)), tfi
    assert 0 < DOWN_PROB < 1
    assert UP_PROB + DOWN_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert DOWN_R < 0
    assert isinstance(MID_R, (float, int)), tfi
    assert UP_R > MID_R > DOWN_R

    print("Dice Roll Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def dice_roll_sh_env_tests(
    UP_PROB: float,
    DOWN_PROB: float,
    UP_R: float,
    DOWN_R: float,
    MID_R: float,
    SH_UP_R: float,
    SH_DOWN_R: float,
    SH_MID_R: float,
    I_LEV_FACTOR: float,
    SH_LEV_FACTOR: float,
) -> None:
    """
    Conduct tests for dice roll (safe haven) environment initialisation.

    Parameters:
        Refer to `./envs/dice_roll_sh_envs.py` for input details.
    """
    assert isinstance(UP_PROB, (float, int)), tfi
    assert 0 < UP_PROB < 1
    assert isinstance(DOWN_PROB, (float, int)), tfi
    assert 0 < DOWN_PROB < 1
    assert UP_PROB + DOWN_PROB < 1
    assert isinstance(UP_R, (float, int)), tfi
    assert UP_R > 0
    assert isinstance(DOWN_R, (float, int)), tfi
    assert DOWN_R < 0
    assert isinstance(MID_R, (float, int)), tfi
    assert UP_R > MID_R > DOWN_R

    assert isinstance(SH_UP_R, (float, int)), tfi
    assert -1 < SH_UP_R < 0
    assert isinstance(SH_DOWN_R, (float, int)), tfi
    assert SH_DOWN_R > 0
    assert isinstance(SH_MID_R, (float, int)), tfi
    assert SH_DOWN_R > SH_MID_R >= SH_UP_R

    assert I_LEV_FACTOR != 0
    assert isinstance(SH_LEV_FACTOR, (float, int)), tfi
    assert SH_LEV_FACTOR != 0

    print("Dice Roll (Safe Haven) Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def gbm_env_tests(DRIFT: float, VOL: float, LEV_FACTOR: float) -> None:
    """
    Conduct tests for GBM environment initialisation.

    Parameters:
        Refer to `./envs/gbm_envs.py` for input details.
    """
    assert isinstance(DRIFT, (float, int)), tfi
    assert isinstance(VOL, (float, int)), tfi
    assert VOL > 0, "volatility must be non-zero and non-negative"

    assert (
        DRIFT - VOL**2 / 2 > 0
    ), "require positive growth rate to achieve optimal (positive) leverage solution"

    assert isinstance(LEV_FACTOR, (float, int)), tfi
    assert LEV_FACTOR != 0, "must have non-zero leverage"

    print("GBM Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def market_env_tests(
    MAX_VALUE: float,
    INITIAL_VALUE: float,
    MIN_VALUE_RATIO: float,
    MAX_VALUE_RATIO: float,
    MAX_ABS_ACTION: float,
    MIN_REWARD: float,
    MIN_RETURN: float,
    MAX_RETURN: float,
    MIN_WEIGHT: float,
    LEV_FACTOR: float,
) -> None:
    """
    Conduct tests for market environment initialisation.

    Parameters:
        Refer to `./envs/market_envs.py` for input details.
    """
    assert isinstance(MAX_VALUE, (float, int)), tfi
    assert (
        0 < MAX_VALUE < 1e37
    ), "MAX_VALUE must be less than float32 (single-precision) limit 1e38"
    assert isinstance(INITIAL_VALUE, (float, int)), tfi
    assert INITIAL_VALUE > 0
    assert isinstance(MIN_VALUE_RATIO, (float, int)), tfi
    assert 1 / MAX_VALUE <= MIN_VALUE_RATIO < 1
    assert isinstance(MAX_VALUE_RATIO, (float, int)), tfi
    assert MIN_VALUE_RATIO < MAX_VALUE_RATIO

    assert isinstance(MAX_ABS_ACTION, (float, int)), tfi
    assert MAX_ABS_ACTION > 0
    assert isinstance(MIN_REWARD, (float, int)), tfi
    assert MIN_REWARD > 0
    assert isinstance(MIN_RETURN, (float, int)), tfi
    assert MIN_RETURN > -1
    assert isinstance(MAX_RETURN, (float, int)), tfi
    assert MAX_RETURN > MIN_RETURN
    assert isinstance(MIN_WEIGHT, (float, int)), tfi
    assert MIN_WEIGHT > 0

    MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
    MIN_OBS_SPACE = min(
        MIN_VALUE / INITIAL_VALUE - 1, MIN_REWARD / MAX_VALUE, MIN_VALUE
    )

    assert MAX_VALUE > MIN_VALUE
    assert INITIAL_VALUE > MIN_VALUE
    assert MAX_VALUE_RATIO > MIN_OBS_SPACE

    assert isinstance(LEV_FACTOR, (float, int)), tfi
    assert LEV_FACTOR != 0

    print("Market Environment and Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def guidance_env_tests(
    name: str,
    MAX_ABS_ACTION: float,
    MIN_REWARD: float,
    MIN_RETURN: float,
    WIND_REDUCTION: float,
    MIN_TIME_RATIO: float,
    MAX_DIST_MUL: float,
    WIND_MAG_VOL: float,
    WIND_AZI_VOL: float,
    WIND_INC_VOL: float,
    ALPHA: float,
    BETA: float,
) -> None:
    """
    Conduct tests for guidance environment initialisation.

    Parameters:
        Refer to guidance environments in `./envs` for input details.
    """
    assert isinstance(name, str), ts

    assert isinstance(MAX_ABS_ACTION, (float, int)), tfi
    assert MAX_ABS_ACTION > 0
    assert isinstance(MIN_REWARD, (float, int)), tfi
    assert MIN_REWARD > 0
    assert isinstance(MIN_RETURN, (float, int)), tfi
    assert MIN_RETURN > -1
    assert isinstance(WIND_REDUCTION, (float, int)), tfi
    assert WIND_REDUCTION > 0
    assert isinstance(MIN_TIME_RATIO, (float, int)), tfi
    assert 0 < MIN_TIME_RATIO < 1
    assert isinstance(MAX_DIST_MUL, (float, int)), tfi
    assert MAX_DIST_MUL >= 1

    assert isinstance(WIND_MAG_VOL, (float, int)), tfi
    assert WIND_MAG_VOL > 0
    assert isinstance(WIND_AZI_VOL, (float, int)), tfi
    assert WIND_AZI_VOL > 0
    assert isinstance(WIND_INC_VOL, (float, int)), tfi
    assert WIND_INC_VOL > 0

    assert isinstance(ALPHA, (float, int)), tfi
    assert ALPHA >= 0
    assert isinstance(BETA, (float, int)), tfi
    assert BETA >= 0

    if BETA > 1:
        print(
            "beta = {} > 1 is unadvised due to constant increase in time steps.".format(
                BETA
            )
        )

    print("{} Environment Tests: Passed".format(name))


def laminar_env_tests(
    WIND_REDUCTION: float, LENGTH: float, VELOCITY: float, THRESHOLD_RAD: float
) -> None:
    """
    Conduct tests for guidance under laminar flow environment initialisation.

    Parameters:
        Refer to `./envs/laminar_envs.py` for input details.
    """
    assert isinstance(LENGTH, int), tfi
    assert LENGTH >= 2
    assert isinstance(VELOCITY, (float, int)), tfi
    assert VELOCITY > WIND_REDUCTION
    assert isinstance(THRESHOLD_RAD, (float, int)), tfi
    assert 0 < THRESHOLD_RAD < LENGTH

    print("Laminar Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def two_stage_env_tests(
    WIND_REDUCTION: float,
    LENGTH: float,
    VELOCITY_1: float,
    VELOCITY_2: float,
    THRESHOLD_RAD_1: float,
    THRESHOLD_RAD_2: float,
    HEIGHT_D: float,
    HEIGHT_M: float,
    MAX_RANGE: float,
) -> None:
    """
    Conduct tests for two-stage delivery environment initialisation.

    Parameters:
        Refer to `./envs/two_stage_envs.py` for input details.
    """
    assert isinstance(LENGTH, int), tfi
    assert LENGTH >= 2
    assert isinstance(VELOCITY_1, (float, int)), tfi
    assert VELOCITY_1 > WIND_REDUCTION
    assert isinstance(VELOCITY_2, (float, int)), tfi
    assert VELOCITY_2 > WIND_REDUCTION
    assert isinstance(THRESHOLD_RAD_1, (float, int)), tfi
    assert 0 < THRESHOLD_RAD_1 < LENGTH
    assert isinstance(THRESHOLD_RAD_1, (float, int)), tfi
    assert 0 < THRESHOLD_RAD_2 < LENGTH

    assert isinstance(HEIGHT_D, (float, int)), tfi
    assert isinstance(HEIGHT_M, (float, int)), tfi
    assert HEIGHT_D > HEIGHT_M > THRESHOLD_RAD_1
    assert HEIGHT_D > HEIGHT_M > THRESHOLD_RAD_2
    assert isinstance(MAX_RANGE, (float, int)), tfi
    assert MAX_RANGE > HEIGHT_M > THRESHOLD_RAD_2 > 0

    print("Two-Stage Parameter Tests: Passed")

    print(
        "--------------------------------------------------------------------------------"
    )


def countermeasure_env_tests(
    WIND_REDUCTION: float,
    LENGTH: float,
    VELOCITY_1: float,
    VELOCITY_2: float,
    THRESHOLD_RAD: float,
    MAX_OBS_R: float,
    MAX_TARGET_R: float,
    HEIGHT_E: float,
) -> None:
    """
    Conduct tests for countermeasure environment initialisation.

    Parameters:
        Refer to `./envs/counter_envs.py` for input details.
    """
    assert isinstance(LENGTH, int), tfi
    assert LENGTH >= 2
    assert isinstance(VELOCITY_1, (float, int)), tfi
    assert VELOCITY_1 > WIND_REDUCTION
    assert isinstance(VELOCITY_2, (float, int)), tfi
    assert VELOCITY_2 > WIND_REDUCTION
    assert VELOCITY_2 > VELOCITY_1
    assert isinstance(THRESHOLD_RAD, (float, int)), tfi
    assert 0 < THRESHOLD_RAD < LENGTH

    assert isinstance(MAX_OBS_R, (float, int)), tfi
    assert isinstance(MAX_TARGET_R, (float, int)), tfi
    assert MAX_OBS_R >= MAX_TARGET_R > THRESHOLD_RAD
    assert isinstance(HEIGHT_E, (float, int)), tfi
    assert MAX_TARGET_R > HEIGHT_E >= 0

    print("Catching Fire Parameter Tests: Passed")

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
