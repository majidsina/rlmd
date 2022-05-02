"""
title:                  lev_exp.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    Contains an assortment of functions for conducting optimal leverage
    experiments across a variety of Markovian environments.
"""

import sys

sys.path.append("./")

from typing import Tuple

import torch as T


def param_range(low: float, high: float, increment: float):
    """
    Create list of increments.

    Parameters:
        lows: start
        high: end
        increment: step size

    Returns:
        params: range of parameters
    """
    min = int(low / increment)
    max = int(high / increment + 1)
    mod = low / increment - min

    params = [(x + mod) * increment for x in range(min, max, 1)]

    if 0 in params:
        params.remove(0)

    return params


def coin_fixed_final_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
):
    """
    Simple printing of end result of fixed leverage valuations.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    for lev in lev_range:

        gambles = T.where(outcomes == 1, 1 + lev * up_r, 1 + lev * down_r)

        value_T = value_0 * gambles.prod(dim=1)

        sort_value = value_T.sort(descending=True)[0]
        top_value = sort_value[0:top]
        adj_value = sort_value[top:]

        # summary statistics
        std, mean = T.std_mean(value_T, unbiased=False)
        med = T.median(value_T)
        mad = T.mean(T.abs(value_T - mean))

        std_top, mean_top = T.std_mean(top_value, unbiased=False)
        med_top = T.median(top_value)
        mad_top = T.mean(T.abs(top_value - mean_top))

        std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
        med_adj = T.median(adj_value)
        mad_adj = T.mean(T.abs(adj_value - mean_adj))

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )


def coin_smart_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Valuations across all time for fixed leverages.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes

    Returns:
        data: valuation summary statistics for each time step
        data_T: valuations at maturity
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    data = T.zeros((len(lev_range), 3 * 4 + 1, horizon - 1), device=device)
    data_T = T.zeros((len(lev_range), investors), device=device)

    i = 0
    for lev in lev_range:

        gambles = T.where(outcomes == 1, 1 + lev * up_r, 1 + lev * down_r)
        initial = value_0 * gambles[:, 0]

        for t in range(horizon - 1):

            value_t = initial * gambles[:, t + 1]
            initial = value_t

            sort_value = value_t.sort(descending=True)[0]
            top_value = sort_value[0:top]
            adj_value = sort_value[top:]

            # summary statistics
            std, mean = T.std_mean(value_t, unbiased=False)
            med = T.median(value_t)
            mad = T.mean(T.abs(value_t - mean))

            std_top, mean_top = T.std_mean(top_value, unbiased=False)
            med_top = T.median(top_value)
            mad_top = T.mean(T.abs(top_value - mean_top))

            std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
            med_adj = T.median(adj_value)
            mad_adj = T.mean(T.abs(adj_value - mean_adj))

            data[i, :, t] = T.tensor(
                [
                    mean,
                    mean_top,
                    mean_adj,
                    mad,
                    mad_top,
                    mad_adj,
                    std,
                    std_top,
                    std_adj,
                    med,
                    med_top,
                    med_adj,
                    lev,
                ]
            )

        data_T[i, :] = value_t

        i += 1

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_top,
                mad_adj,
                std_adj,
            )
        )

    return data, data_T


def coin_optimal_lev(
    value_t: float, value_0: float, value_min: float, lev_factor: float, roll: float
) -> float:
    """
    Calculate optimal leverage besed on either rolling or fixed stop loss at each
    time step.

    Parameters:
        value_t: current portfolio values
        value_0: intitial portfolio value
        value_min: global stop-loss
        lev_factor: maximum leverage to not be stopped out by a single move
        roll: retention ratio

    Returns:
        opt_lev: optimal leverage
    """
    # prevent leverage from exceeding the theoritical maximum
    if roll == 0:
        rolling_loss = value_min
        value_roll = T.maximum(value_min, rolling_loss)
        return lev_factor * (1 - value_roll / value_t)

    else:
        rolling_loss = T.where(
            value_t <= value_0, value_min, value_0 + roll * (value_t - value_0)
        )
        return lev_factor * (1 - rolling_loss / value_t)


def coin_big_brain_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    lev_factor: float,
    stop_min: float,
    stop_max: float,
    stop_incr: float,
    roll_min: float,
    roll_max: float,
    roll_incr: float,
) -> T.FloatTensor:
    """
    Valuations across all time for variable stop-losses and retention ratios that
    calculates optimal leverage at each time step for each investor.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        lev_factor: maximum leverage to not be stopped out by a single move
        stop_low: starting stop-loss
        stop_high: ending stop-loss
        stop_incr: stop-loss step sizes
        roll_low: starting retention ratio
        roll_high: ending retention ratio
        roll_incr: retention ratio step sizes

    Returns:
        data: valuation summary statistics for each time step
    """
    stop_range = T.tensor(param_range(stop_min, stop_max, stop_incr), device=device)
    roll_range = T.tensor(param_range(roll_min, roll_max, roll_incr), device=device)

    data = T.zeros(
        (len(roll_range), len(stop_range), 3 * 4 * 2 + 2, horizon - 1), device=device
    )

    j = 0
    for roll_level in roll_range:

        i = 0
        for stop_level in stop_range:

            gambles = T.where(outcomes == 1, up_r, down_r)

            value_min = stop_level * value_0

            lev = coin_optimal_lev(value_0, value_0, value_min, lev_factor, roll_level)
            sample_lev = T.ones((1, investors), device=device) * lev

            initial = value_0 * (1 + lev * gambles[:, 0])

            sample_lev = coin_optimal_lev(
                initial, value_0, value_min, lev_factor, roll_level
            )

            for t in range(horizon - 1):
                sort_lev = sample_lev.sort(descending=True)[0]
                top_lev = sort_lev[0:top]
                adj_lev = sort_lev[top:]

                # leverage summary statistics
                lstd, lmean = T.std_mean(sample_lev, unbiased=False)
                lmed = T.median(sample_lev)
                lmad = T.mean(T.abs(sample_lev - lmean))

                lstd_top, lmean_top = T.std_mean(top_lev, unbiased=False)
                lmed_top = T.median(top_lev)
                lmad_top = T.mean(T.abs(top_lev - lmean_top))

                lstd_adj, lmean_adj = T.std_mean(adj_lev, unbiased=False)
                lmed_adj = T.median(adj_lev)
                lmad_adj = T.mean(T.abs(adj_lev - lmean_adj))

                data[j, i, 12:26, t] = T.tensor(
                    [
                        lmean,
                        lmean_top,
                        lmean_adj,
                        lmad,
                        lmad_top,
                        lmad_adj,
                        lstd,
                        lstd_top,
                        lstd_adj,
                        lmed,
                        lmed_top,
                        lmed_adj,
                        stop_level,
                        roll_level,
                    ]
                )

                # calculate one-period change in valuations
                value_t = initial * (1 + sample_lev * gambles[:, t + 1])
                initial = value_t

                sample_lev = coin_optimal_lev(
                    initial, value_0, value_min, lev_factor, roll_level
                )

                sort_value = value_t.sort(descending=True)[0]
                top_value = sort_value[0:top]
                adj_value = sort_value[top:]

                # valuation summary statistics
                std, mean = T.std_mean(value_t, unbiased=False)
                med = T.median(value_t)
                mad = T.mean(T.abs(value_t - mean))

                std_top, mean_top = T.std_mean(top_value, unbiased=False)
                med_top = T.median(top_value)
                mad_top = T.mean(T.abs(top_value - mean_top))

                std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
                med_adj = T.median(adj_value)
                mad_adj = T.mean(T.abs(adj_value - mean_adj))

                data[j, i, 0:12, t] = T.tensor(
                    [
                        mean,
                        mean_top,
                        mean_adj,
                        mad,
                        mad_top,
                        mad_adj,
                        std,
                        std_top,
                        std_adj,
                        med,
                        med_top,
                        med_adj,
                    ]
                )

            i += 1

            print(
                """stop/roll {:1.0f}/{:1.0f}%:
                    avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}""".format(
                    stop_level * 100,
                    roll_level * 100,
                    mean,
                    med,
                    mad,
                    std,
                    lmean,
                    lmed,
                    lmad,
                    lstd,
                    mean_top,
                    med_top,
                    mad_top,
                    std_top,
                    lmean_top,
                    lmed_top,
                    lmad_top,
                    lstd_top,
                    mean_adj,
                    med_adj,
                    mad_adj,
                    std_adj,
                    lmean_adj,
                    lmed_adj,
                    lmad_adj,
                    lstd_adj,
                )
            )
        j += 1

    return data


def coin_galaxy_brain_lev(
    device: T.device,
    ru_min: float,
    ru_max: float,
    ru_incr: float,
    rd_min: float,
    rd_max: float,
    rd_incr: float,
    pu_min: float,
    pu_max: float,
    pu_incr: float,
) -> T.FloatTensor:
    """
    Optimal leverage determined using the Kelly criterion to maximise the geometric
    return of a binary payout valid across all time steps.

    Parameters:
        ru_min: starting up return
        ru_max: ending up return
        ru_incr: up return step sizes
        rd_min: starting down return (absolute)
        rd_max: ending down return (absolute)
        rd_incr: down return step sizes (absolute)
        pu_min: starting up probability
        pu_max: ending up probability
        pu_incr: up probability step sizes
    """
    ru_range = param_range(ru_min, ru_max, ru_incr)
    rd_range = param_range(rd_min, rd_max, rd_incr)
    pu_range = param_range(pu_min, pu_max, pu_incr)

    data = T.zeros((len(pu_range), len(ru_range), len(ru_range), 3 + 1), device=device)

    i = 0
    for pu in pu_range:

        j = 0
        for ru in ru_range:

            k = 0
            for rd in rd_range:
                kelly = pu / rd - (1 - pu) / ru
                data[i, j, k, :] = T.tensor([pu, ru, rd, kelly], device=device)

                k += 1

            j += 1

        i += 1

    return data


def dice_fixed_final_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    mid_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
):
    """
    Simple printing of end result of fixed leverage valuations.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    outcomes = T.tensor(outcomes, dtype=T.float, device=device)

    for lev in lev_range:

        gambles = T.where(outcomes == 0, 1 + lev * up_r, outcomes)
        gambles = T.where(outcomes == 1, 1 + lev * down_r, gambles)
        gambles = T.where(outcomes == 2, 1 + lev * mid_r, gambles)

        value_T = value_0 * gambles.prod(dim=1)

        sort_value = value_T.sort(descending=True)[0]
        top_value = sort_value[0:top]
        adj_value = sort_value[top:]

        # summary statistics
        std, mean = T.std_mean(value_T, unbiased=False)
        med = T.median(value_T)
        mad = T.mean(T.abs(value_T - mean))

        std_top, mean_top = T.std_mean(top_value, unbiased=False)
        med_top = T.median(top_value)
        mad_top = T.mean(T.abs(top_value - mean_top))

        std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
        med_adj = T.median(adj_value)
        mad_adj = T.mean(T.abs(adj_value - mean_adj))

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )


def dice_smart_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    mid_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Valuations across all time for fixed leverages.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes

    Returns:
        data: valuation summary statistics for each time step
        data_T: valuations at maturity
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    data = T.zeros((len(lev_range), 3 * 4 + 1, horizon - 1), device=device)
    data_T = T.zeros((len(lev_range), investors), device=device)

    outcomes = T.tensor(outcomes, dtype=T.float, device=device)

    i = 0
    for lev in lev_range:

        gambles = T.where(outcomes == 0, 1 + lev * up_r, outcomes)
        gambles = T.where(outcomes == 1, 1 + lev * down_r, gambles)
        gambles = T.where(outcomes == 2, 1 + lev * mid_r, gambles)
        initial = value_0 * gambles[:, 0]

        for t in range(horizon - 1):

            value_t = initial * gambles[:, t + 1]
            initial = value_t

            sort_value = value_t.sort(descending=True)[0]
            top_value = sort_value[0:top]
            adj_value = sort_value[top:]

            # summary statistics
            std, mean = T.std_mean(value_t, unbiased=False)
            med = T.median(value_t)
            mad = T.mean(T.abs(value_t - mean))

            std_top, mean_top = T.std_mean(top_value, unbiased=False)
            med_top = T.median(top_value)
            mad_top = T.mean(T.abs(top_value - mean_top))

            std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
            med_adj = T.median(adj_value)
            mad_adj = T.mean(T.abs(adj_value - mean_adj))

            data[i, :, t] = T.tensor(
                [
                    mean,
                    mean_top,
                    mean_adj,
                    mad,
                    mad_top,
                    mad_adj,
                    std,
                    std_top,
                    std_adj,
                    med,
                    med_top,
                    med_adj,
                    lev,
                ]
            )

        data_T[i, :] = value_t

        i += 1

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )

    return data, data_T


def dice_optimal_lev(
    device: T.device,
    value_t: float,
    value_0: float,
    value_min: float,
    lev_factor: float,
    roll: float,
) -> float:
    """
    Calculate optimal leverage besed on either rolling or fixed stop loss at each
    time step.

    Parameters:
        device: PyTorch device
        value_t: current portfolio values
        value_0: intitial portfolio value
        value_min: global stop-loss
        lev_factor: maximum leverage to not be stopped out by a single move
        roll: retention ratio

    Returns:
        opt_lev: optimal leverage
    """
    # prevent leverage from exceeding the theoritical maximum
    if roll == 0:
        rolling_loss = value_min
        value_roll = T.maximum(value_min, rolling_loss)
        return lev_factor * (1 - value_roll / value_t)

    else:
        value_t = T.tensor(value_t, dtype=T.float, device=device)
        rolling_loss = T.where(
            value_t <= value_0, value_min, value_0 + roll * (value_t - value_0)
        )
        return lev_factor * (1 - rolling_loss / value_t)


def dice_big_brain_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    mid_r: float,
    lev_factor: float,
    stop_min: float,
    stop_max: float,
    stop_incr: float,
    roll_min: float,
    roll_max: float,
    roll_incr: float,
) -> T.FloatTensor:
    """
    Valuations across all time for variable stop-losses and retention ratios that
    calculates optimal leverage at each time step for each investor.

    Parameters:
        device: PyTorch device
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        lev_factor: maximum leverage to not be stopped out by a single move
        stop_low: starting stop-loss
        stop_high: ending stop-loss
        stop_incr: stop-loss step sizes
        roll_low: starting retention ratio
        roll_high: ending retention ratio
        roll_incr: retention ratio step sizes

    Returns:
        data: valuation summary statistics for each time step
    """
    stop_range = T.tensor(param_range(stop_min, stop_max, stop_incr), device=device)
    roll_range = T.tensor(param_range(roll_min, roll_max, roll_incr), device=device)

    data = T.zeros(
        (len(roll_range), len(stop_range), 3 * 4 * 2 + 2, horizon - 1), device=device
    )

    outcomes = T.tensor(outcomes, dtype=T.double, device=device)

    j = 0
    for roll_level in roll_range:

        i = 0
        for stop_level in stop_range:

            gambles = T.where(outcomes == 0, up_r, outcomes)
            gambles = T.where(outcomes == 1, down_r, gambles)
            gambles = T.where(outcomes == 2, mid_r, gambles)

            value_min = stop_level * value_0

            lev = dice_optimal_lev(
                device, value_0, value_0, value_min, lev_factor, roll_level
            )
            sample_lev = T.ones((1, investors), device=device) * lev

            initial = value_0 * (1 + lev * gambles[:, 0])

            sample_lev = dice_optimal_lev(
                device, initial, value_0, value_min, lev_factor, roll_level
            )

            for t in range(horizon - 1):
                sort_lev = sample_lev.sort(descending=True)[0]
                top_lev = sort_lev[0:top]
                adj_lev = sort_lev[top:]

                # leverage summary statistics
                lstd, lmean = T.std_mean(sample_lev, unbiased=False)
                lmed = T.median(sample_lev)
                lmad = T.mean(T.abs(sample_lev - lmean))

                lstd_top, lmean_top = T.std_mean(top_lev, unbiased=False)
                lmed_top = T.median(top_lev)
                lmad_top = T.mean(T.abs(top_lev - lmean_top))

                lstd_adj, lmean_adj = T.std_mean(adj_lev, unbiased=False)
                lmed_adj = T.median(adj_lev)
                lmad_adj = T.mean(T.abs(adj_lev - lmean_adj))

                data[j, i, 12:26, t] = T.tensor(
                    [
                        lmean,
                        lmean_top,
                        lmean_adj,
                        lmad,
                        lmad_top,
                        lmad_adj,
                        lstd,
                        lstd_top,
                        lstd_adj,
                        lmed,
                        lmed_top,
                        lmed_adj,
                        stop_level,
                        roll_level,
                    ]
                )

                # calculate one-period change in valuations
                value_t = initial * (1 + sample_lev * gambles[:, t + 1])
                initial = value_t

                sample_lev = dice_optimal_lev(
                    device, initial, value_0, value_min, lev_factor, roll_level
                )

                sort_value = value_t.sort(descending=True)[0]
                top_value = sort_value[0:top]
                adj_value = sort_value[top:]

                # valuation summary statistics
                std, mean = T.std_mean(value_t, unbiased=False)
                med = T.median(value_t)
                mad = T.mean(T.abs(value_t - mean))

                std_top, mean_top = T.std_mean(top_value, unbiased=False)
                med_top = T.median(top_value)
                mad_top = T.mean(T.abs(top_value - mean_top))

                std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
                med_adj = T.median(adj_value)
                mad_adj = T.mean(T.abs(adj_value - mean_adj))

                data[j, i, 0:12, t] = T.tensor(
                    [
                        mean,
                        mean_top,
                        mean_adj,
                        mad,
                        mad_top,
                        mad_adj,
                        std,
                        std_top,
                        std_adj,
                        med,
                        med_top,
                        med_adj,
                    ]
                )

            i += 1

            print(
                """stop/roll {:1.0f}/{:1.0f}%:
                    avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}
                    adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}  l {:1.2f} / {:1.2f} / {:1.1f} / {:1.1f}""".format(
                    stop_level * 100,
                    roll_level * 100,
                    mean,
                    med,
                    mad,
                    std,
                    lmean,
                    lmed,
                    lmad,
                    lstd,
                    mean_top,
                    med_top,
                    mad_top,
                    std_top,
                    lmean_top,
                    lmed_top,
                    lmad_top,
                    lstd_top,
                    mean_adj,
                    med_adj,
                    mad_adj,
                    std_adj,
                    lmean_adj,
                    lmed_adj,
                    lmad_adj,
                    lstd_adj,
                )
            )
        j += 1

    return data


def gbm_fixed_final_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    top: int,
    value_0: T.FloatTensor,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
):
    """
    Simple printing of end result of fixed leverage valuations.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        sh_up_r: safe haven return if up move
        sh_down_r: safe haven return if down move
        sh_mid_r: safe haven return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)

    for lev in lev_range:

        gambles = T.exp(lev * outcomes)

        value_T = value_0 * gambles.prod(dim=1)

        sort_value = value_T.sort(descending=True)[0]
        top_value = sort_value[0:top]
        adj_value = sort_value[top:]

        # summary statistics
        std, mean = T.std_mean(value_T, unbiased=False)
        med = T.median(value_T)
        mad = T.mean(T.abs(value_T - mean))

        std_top, mean_top = T.std_mean(top_value, unbiased=False)
        med_top = T.median(top_value)
        mad_top = T.mean(T.abs(top_value - mean_top))

        std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
        med_adj = T.median(adj_value)
        mad_adj = T.mean(T.abs(adj_value - mean_adj))

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )


def gbm_smart_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Valuations across all time for fixed leverages.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        sh_up_r: safe haven return if up move
        sh_down_r: safe haven return if down move
        sh_mid_r: safe haven return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes

    Returns:
        data: valuation summary statistics for each time step
        data_T: valuations at maturity
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)

    data = T.zeros((len(lev_range), 3 * 4 + 1, horizon - 1), device=device)
    data_T = T.zeros((len(lev_range), investors), device=device)

    i = 0
    for lev in lev_range:

        gambles = T.exp(lev * outcomes)
        initial = value_0 * gambles[:, 0]

        for t in range(horizon - 1):

            value_t = initial * gambles[:, t + 1]
            initial = value_t

            sort_value = value_t.sort(descending=True)[0]
            top_value = sort_value[0:top]
            adj_value = sort_value[top:]

            # summary statistics
            std, mean = T.std_mean(value_t, unbiased=False)
            med = T.median(value_t)
            mad = T.mean(T.abs(value_t - mean))

            std_top, mean_top = T.std_mean(top_value, unbiased=False)
            med_top = T.median(top_value)
            mad_top = T.mean(T.abs(top_value - mean_top))

            std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
            med_adj = T.median(adj_value)
            mad_adj = T.mean(T.abs(adj_value - mean_adj))

            data[i, :, t] = T.tensor(
                [
                    mean,
                    mean_top,
                    mean_adj,
                    mad,
                    mad_top,
                    mad_adj,
                    std,
                    std_top,
                    std_adj,
                    med,
                    med_top,
                    med_adj,
                    lev,
                ]
            )

        data_T[i, :] = value_t

        i += 1

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )

    return data, data_T


def dice_sh_fixed_final_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    mid_r: float,
    sh_up_r: float,
    sh_down_r: float,
    sh_mid_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
):
    """
    Simple printing of end result of fixed leverage valuations.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        sh_up_r: safe haven return if up move
        sh_down_r: safe haven return if down move
        sh_mid_r: safe haven return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    outcomes = T.tensor(outcomes, dtype=T.float, device=device)

    for lev in lev_range:

        gambles = T.where(outcomes == 0, 1 + lev * up_r + (1 - lev) * sh_up_r, outcomes)
        gambles = T.where(
            outcomes == 1, 1 + lev * down_r + (1 - lev) * sh_down_r, gambles
        )
        gambles = T.where(
            outcomes == 2, 1 + lev * mid_r + (1 - lev) * sh_mid_r, gambles
        )

        value_T = value_0 * gambles.prod(dim=1)

        sort_value = value_T.sort(descending=True)[0]
        top_value = sort_value[0:top]
        adj_value = sort_value[top:]

        # summary statistics
        std, mean = T.std_mean(value_T, unbiased=False)
        med = T.median(value_T)
        mad = T.mean(T.abs(value_T - mean))

        std_top, mean_top = T.std_mean(top_value, unbiased=False)
        med_top = T.median(top_value)
        mad_top = T.mean(T.abs(top_value - mean_top))

        std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
        med_adj = T.median(adj_value)
        mad_adj = T.mean(T.abs(adj_value - mean_adj))

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )


def dice_sh_smart_lev(
    device: T.device,
    outcomes: T.FloatTensor,
    investors: T.IntTensor,
    horizon: T.IntTensor,
    top: int,
    value_0: T.FloatTensor,
    up_r: float,
    down_r: float,
    mid_r: float,
    sh_up_r: float,
    sh_down_r: float,
    sh_mid_r: float,
    lev_low: float,
    lev_high: float,
    lev_incr: float,
) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Valuations across all time for fixed leverages.

    Parameters:
        outcomes: matrix of up or down payoffs for each investor for all time
        investor: number of investors
        horizon: number of time steps
        top: number of investors in top sub-group
        value_0: intitial portfolio value
        up_r: return if up move
        down_r: return if down move
        mid_r: return if mid move
        sh_up_r: safe haven return if up move
        sh_down_r: safe haven return if down move
        sh_mid_r: safe haven return if mid move
        lev_low: starting leverage
        lev_high: ending leverage
        lev_incr: leverage step sizes

    Returns:
        data: valuation summary statistics for each time step
        data_T: valuations at maturity
    """
    lev_range = T.tensor(param_range(lev_low, lev_high, lev_incr), device=device)
    lev_range = -lev_range if -down_r > up_r else lev_range

    data = T.zeros((len(lev_range), 3 * 4 + 1, horizon - 1), device=device)
    data_T = T.zeros((len(lev_range), investors), device=device)

    outcomes = T.tensor(outcomes, dtype=T.float, device=device)

    i = 0
    for lev in lev_range:

        gambles = T.where(outcomes == 0, 1 + lev * up_r + (1 - lev) * sh_up_r, outcomes)
        gambles = T.where(
            outcomes == 1, 1 + lev * down_r + (1 - lev) * sh_down_r, gambles
        )
        gambles = T.where(
            outcomes == 2, 1 + lev * mid_r + (1 - lev) * sh_mid_r, gambles
        )
        initial = value_0 * gambles[:, 0]

        for t in range(horizon - 1):

            value_t = initial * gambles[:, t + 1]
            initial = value_t

            sort_value = value_t.sort(descending=True)[0]
            top_value = sort_value[0:top]
            adj_value = sort_value[top:]

            # summary statistics
            std, mean = T.std_mean(value_t, unbiased=False)
            med = T.median(value_t)
            mad = T.mean(T.abs(value_t - mean))

            std_top, mean_top = T.std_mean(top_value, unbiased=False)
            med_top = T.median(top_value)
            mad_top = T.mean(T.abs(top_value - mean_top))

            std_adj, mean_adj = T.std_mean(adj_value, unbiased=False)
            med_adj = T.median(adj_value)
            mad_adj = T.mean(T.abs(adj_value - mean_adj))

            data[i, :, t] = T.tensor(
                [
                    mean,
                    mean_top,
                    mean_adj,
                    mad,
                    mad_top,
                    mad_adj,
                    std,
                    std_top,
                    std_adj,
                    med,
                    med_top,
                    med_adj,
                    lev,
                ]
            )

        data_T[i, :] = value_t

        i += 1

        print(
            """       lev {:1.0f}%:
                 avg mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 top mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}
                 adj mean/med/mad/std:  $ {:1.2e} / {:1.2e} / {:1.1e} / {:1.1e}""".format(
                lev * 100,
                mean,
                med,
                mad,
                std,
                mean_top,
                med_top,
                mad_top,
                std_top,
                mean_adj,
                med_adj,
                mad_adj,
                std_adj,
            )
        )

    return data, data_T
