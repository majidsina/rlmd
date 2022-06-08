"""
title:                  test_input_envs.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Responsible for conducting tests on user inputs across all custom
    created environments.
"""

import sys

sys.path.append("./")

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
