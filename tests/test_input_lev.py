"""
title:                  input_tests.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Responsible for conducting tests on all user inputs for empirical
    leverage experiments.
"""

import sys

sys.path.append("./")

from os import PathLike
from typing import List, Union

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
    assert INVESTORS > 1
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
    assert INVESTORS > 1
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
    assert INVESTORS > 1
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
    assert INVESTORS > 1
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
