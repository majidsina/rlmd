"""
title:                  env_resources.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    Collection of tools needed for various multiplicative environments.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]


def multi_dones(
    wealth: float,
    MIN_VALUE: float,
    reward: float,
    MIN_REWARD: float,
    step_return: float,
    MIN_RETURN: float,
    lev: NDArrayFloat,
    MIN_WEIGHT: float,
    MAX_ABS_ACTION: float,
    LEV_FACTOR: float,
    next_state: NDArrayFloat,
    MAX_VALUE_RATIO: float,
    active: float = None,
) -> List[bool]:
    """
    Agent done flags for multiplicative environments controlling episode termination
    and Q-value estimation for learning (i.e. whether genuine or forced).

    Parameters:
        wealth: portfolio value
        reward: time-average growth rate
        step_return: single step return
        lev: asset leverages
        next_state: normalised state values
        active: bet size for investors B and C

    Returns:
        done: episode termination
        learn_done: agent learning done flag
    """
    done_wealth = wealth == MIN_VALUE
    done_reward = reward < MIN_REWARD
    done_return = step_return == MIN_RETURN

    done_lev_max = np.any(np.abs(lev) == MAX_ABS_ACTION * LEV_FACTOR)
    done_lev_min = np.all(np.abs(lev) < MIN_WEIGHT)

    done_state = np.any(next_state >= MAX_VALUE_RATIO)

    done_active = False if active == None else active == 0

    done = bool(
        done_wealth
        or done_reward
        or done_return
        or done_lev_max
        or done_lev_min
        or done_state
        or done_active
    )

    learn_done = done and not done_state

    return [done, learn_done]


def multi_gbm_dones(
    wealth: float,
    MIN_VALUE: float,
    reward: float,
    MIN_REWARD: float,
    step_return: float,
    MIN_RETURN: float,
    lev: NDArrayFloat,
    MIN_WEIGHT: float,
    MAX_ABS_ACTION: float,
    LEV_FACTOR: float,
    next_state: NDArrayFloat,
    MAX_VALUE_RATIO: float,
    active: float = None,
) -> List[bool]:
    """
    Agent done flags for multiplicative GBM environments controlling episode
    termination and Q-value estimation for learning (i.e. whether genuine or forced).

    Parameters:
        wealth: portfolio value
        reward: time-average growth rate
        step_return: single step return
        lev: asset leverages
        next_state: normalised state values
        active: bet size for investors B and C

    Returns:
        done: episode termination
        learn_done: agent learning done flag
    """
    done_wealth = wealth == MIN_VALUE
    done_reward = reward < MIN_REWARD
    done_return = step_return == MIN_RETURN

    done_lev_max = np.all(np.abs(lev) == MAX_ABS_ACTION * LEV_FACTOR)
    done_lev_min = np.all(np.abs(lev) < MIN_WEIGHT)

    done_state = np.any(next_state >= MAX_VALUE_RATIO)

    done_active = False if active == None else active == 0

    done = bool(
        done_wealth
        or done_reward
        or done_return
        or done_lev_max
        or done_lev_min
        or done_state
        or done_active
    )

    learn_done = done and not done_state

    return [done, learn_done]


def market_dones(
    time: int,
    TIME_LENGTH: int,
    wealth: float,
    MIN_VALUE: float,
    reward: float,
    MIN_REWARD: float,
    step_return: float,
    MIN_RETURN: float,
    lev: NDArrayFloat,
    MIN_WEIGHT: float,
    MAX_ABS_ACTION: float,
    LEV_FACTOR: float,
    next_state: NDArrayFloat,
    MAX_VALUE_RATIO: float,
    active: float = None,
) -> List[bool]:
    """
    Agent done flags for market environments controlling episode termination and
    Q-value estimation for learning (i.e. whether genuine or forced).

    Parameters:
        time: episode time step
        wealth: portfolio value
        reward: time-average growth rate
        step_return: single step return
        lev: asset leverages
        next_state: normalised state values
        active: bet size for investors B and C

    Returns:
        done: episode termination
        learn_done: agent learning done flag
    """
    done_time = time == TIME_LENGTH

    done_wealth = wealth == MIN_VALUE
    done_reward = reward < MIN_REWARD
    done_return = step_return == MIN_RETURN

    done_lev_max = np.all(np.abs(lev) == MAX_ABS_ACTION * LEV_FACTOR)
    done_lev_min = np.all(np.abs(lev) < MIN_WEIGHT)

    done_state = np.any(next_state >= MAX_VALUE_RATIO)

    done_active = False if active == None else active == 0

    done = bool(
        done_time
        or done_wealth
        or done_reward
        or done_return
        or done_lev_max
        or done_lev_min
        or done_state
        or done_active
    )

    learn_done = done and not (done_time or done_state)

    return [done, learn_done]


def observed_market_state(
    market_extract: NDArrayFloat, time_step: int, action_days: int, obs_days: int
) -> NDArrayFloat:
    """
    Flattened and ordered observed historical market prices used for agent decision-making.

    Parameters:
        market_extract: shuffled time series extract of history used for training/inference
        time_step: current time step in market_extract
        action_days: number of days between agent trading actions
        obs_days: number of previous days agent uses for decision-making

    Return:
        obs_state: current observed past states used for next decision
    """
    if obs_days == 1:
        return market_extract[time_step * action_days]

    if time_step > 0:
        return market_extract[
            time_step * action_days : time_step * action_days + obs_days
        ].reshape(-1)[::-1]
    else:
        return market_extract[time_step * action_days : obs_days].reshape(-1)[::-1]


def time_slice(
    prices: NDArrayFloat, extract_days: int, action_days: int, sample_days: int
) -> Tuple[NDArrayFloat, int]:
    """
    Extract sequential slice of time series preserving the non-i.i.d. nature of
    the data keeping heteroscedasticity and serial correlation relatively unchanged
    compared to random sampling.

    Parameters:
        prices: array of all assets prices across a shared time period
        extract_days: length of period to be extracted
        action_days: number of days between agent trading actions
        sample_days: maximum buffer of days restricting starting index

    Returns:
        market_extract: extracted time sliced data from complete time series
        start_idx: first index of sampled training time series
    """
    max_train = prices.shape[0] - sample_days

    start_idx = np.random.randint(0, max_train)
    end_idx = start_idx + extract_days * action_days + 1

    market_extract = prices[start_idx:end_idx]

    return market_extract, start_idx


def shuffle_data(prices: NDArrayFloat, interval_days: int) -> NDArrayFloat:
    """
    Split data into identical subset intervals and randomly shuffle data within each
    interval. Purpose is to generate a fairly random (non-parametric) bootstrap (or seed)
    for known historical data while preserving overall long-term trends and structure.

    Parameters:
        prices: array of all historical assets prices across a shared time period
        interval_days: size of ordered subsets

    Returns:
        shuffled_prices: prices randomly shuffled within each interval
    """
    length = prices.shape[0]
    mod = length % interval_days
    intervals = int((length - mod) / interval_days)

    shuffled_prices = np.empty((length, prices.shape[1]))

    if mod != 0:
        split_prices = np.split(prices[:-mod], indices_or_sections=intervals, axis=0)
    else:
        split_prices = np.split(prices, indices_or_sections=intervals, axis=0)

    for x in range(intervals):
        shuffled_prices[
            x * interval_days : (x + 1) * interval_days
        ] = np.random.permutation(split_prices[x])

    if mod != 0:
        shuffled_prices[intervals * interval_days :] = np.random.permutation(
            prices[-mod:]
        )

    return shuffled_prices


def train_test_split(
    prices: NDArrayFloat, train_years: float, test_years: float, gap_years: float
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Generate sequential train/test split of time series data with fixed gap between sets.
    This formulation preserves the non-i.i.d. nature of the data keeping heteroscedasticity
    and serial correlation relatively unchanged compared to random sampling.

    Parameters:
        prices: array of all assets prices across a shared time period
        train_years: length of training period
        test_years: length of testing period
        gap_years: fixed length between end of training and start of testing periods

    Returns:
        train: array of training period
        test: array of evaluation period
    """
    train_days = int(252 * train_years)
    test_days = int(252 * test_years)
    gap_days = int(252 * gap_years)

    max_train = prices.shape[0] - (1 + train_days + gap_days + test_days)

    start_train = np.random.randint(train_days - 1, max_train)
    end_train = start_train + train_days + 1
    start_test = end_train + gap_days
    end_test = start_test + test_days + 1

    train, test = prices[start_train:end_train], prices[start_test:end_test]

    return train, test
