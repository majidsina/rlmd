"""
title:                  critic_loss.py
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
    Responsible for aggregating mini-batch of critic current and target critic
    values into empirical (arithmetic) mean losses for neural network backpropagation
    to learn Q-values. Also generates shadow mean estimates for the mini-batch.
"""

from typing import Tuple

import torch as T


@T.jit.script
def truncation(
    estimated: T.FloatTensor, target: T.FloatTensor
) -> Tuple[T.FloatTensor, T.FloatTensor]:
    """
    Elements to be truncated based on Gaussian distribution assumption based on a
    correction of Section 3.3 in https://arxiv.org/pdf/1906.00495.pdf.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        estimated: truncated current Q-values
        target: truncated Q-values from mini-batch
    """
    sigma1, mean1 = T.std_mean(estimated, unbiased=False)
    sigma2, mean2 = T.std_mean(target, unbiased=False)
    zero1, zero2 = estimated - estimated, target - target

    # 3-sigma rejection rule
    estimated = T.where(T.abs(estimated - mean1) > 3 * sigma1, zero1, estimated)
    target = T.where(T.abs(target - mean2) > 3 * sigma2, zero2, target)

    return estimated, target


@T.jit.script
def cauchy(
    estimated: T.FloatTensor, target: T.FloatTensor, scale: float
) -> T.FloatTensor:
    """
    Cauchy loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        scale: Cauchy scale parameter

    Returns:
        loss (float): loss values
    """
    scale = T.tensor(scale)
    arg = ((target - estimated) / scale) ** 2

    return T.log(1 + arg)


@T.jit.script
def nagy_algo(
    estimated: T.FloatTensor, target: T.FloatTensor, scale: float
) -> T.FloatTensor:
    """
    Use the Nagy alogrithm to estimate the Cauchy scale paramter based on residual
    errors in Eq. 18 in
    http://www.jucs.org/jucs_12_9/parameter_estimation_of_the/jucs_12_09_1332_1344_nagy.pdf
    and Section 3.2 in https://arxiv.org/pdf/1906.00495.pdf.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        scale: current Cauchy scale parameter [step: t]

    Returns:
        scale_new: updated scale parameter > 0 [step: t + 1]
    """
    scale = T.tensor(scale)
    estimated, target = estimated.detach().clone(), target.detach().clone()

    arg = ((target - estimated) / scale) ** 2
    inv_error = 1 / T.mean(1 / (1 + arg))

    if inv_error > 1:
        return scale * T.sqrt(inv_error - 1)
    else:
        return scale


@T.jit.script
def correntropy(
    estimated: T.FloatTensor, target: T.FloatTensor, kernel: float
) -> T.FloatTensor:
    """
    Correntropy-induced metric (CIM) loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        kernel: width of Gaussain

    Returns:
        loss (float): loss values
    """
    kernel = T.tensor(kernel)
    arg = (target - estimated) ** 2

    return 1 - T.exp(-arg / (2 * kernel**2)) / T.sqrt(2 * T.pi * kernel)


@T.jit.script
def cim_size(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    Empirically estimated kernel size for CIM taken as the average reconstruction error
    based on Eq. 25 in https://lcs.ios.ac.cn/~ydshen/ICDM-12.pdf.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        kernel: standard deviation
    """
    arg = (target - estimated) ** 2

    return T.std(arg.detach().clone(), unbiased=False)


@T.jit.script
def hypersurface(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    Hypersurface cost based loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss: loss values
    """
    arg = (target - estimated) ** 2

    return T.sqrt(1 + arg) - 1


@T.jit.script
def mse(estimated: T.FloatTensor, target: T.FloatTensor, exp: int = 0) -> T.FloatTensor:
    """
    MSE loss function and higher even orders.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch
        exp (even int): exponent in addition to MSE

    Returns:
        loss: loss values
    """
    return (target - estimated) ** (int(2 + exp))


@T.jit.script
def mae(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    MAE loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss: loss values
    """
    return T.abs(target - estimated)


@T.jit.script
def huber(estimated: T.FloatTensor, target: T.FloatTensor) -> T.FloatTensor:
    """
    Huber loss function.

    Parameters:
        estimated: current Q-values
        target: Q-values from mini-batch

    Returns:
        loss: loss values
    """
    arg = T.abs(target - estimated)

    return T.where(arg < 1, 0.5 * arg**2, arg - 0.5)


@T.jit.script
def hill_est(values: T.FloatTensor, log_noise: T.FloatTensor) -> T.FloatTensor:
    """
    Calculates using extreme value thoery the Hill estimator as a proxy for the
    tail index of a power law provided alpha > 0. Treats all values as extreme.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        log_noise: miniscule constant to tail estimation logarithm bounded

    Returns:
        alpha: tail index of power law
    """
    values = T.abs(values.view(-1))

    order_stats = values.sort(descending=True)[0]
    min_val = order_stats[0]
    geo_mean = T.prod(order_stats[1:])
    geo_mean = geo_mean ** (1 / geo_mean.shape[0])

    hill_1 = T.log(geo_mean / min_val + log_noise)
    gamma = hill_1

    # method of moments estimator
    # hill_2 = ((vals - min_val)**2).mean()
    # gamma += 1 - 1 / 2 * (1 - hill_1**2 / hill_2)**(-1)

    return 1 / gamma


@T.jit.script
def zipf_plot(
    values: T.FloatTensor,
    log_noise: T.FloatTensor,
    zipf_x: T.FloatTensor,
    zipf_x2: T.FloatTensor,
) -> T.FloatTensor:
    """
    Obtain gradient of Zipf (or Pareto Q-Q) plot using ordered statistics.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        log_noise: miniscule constant to tail estimation logarithm bounded
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis

    Returns:
        alpha (>=0): tail index estimated using plot gradient
    """
    values = values.view(-1)

    order_stats = values.sort(descending=True)[0]
    order_stats = T.log(order_stats + log_noise)
    diff_stats = order_stats - T.mean(order_stats)

    # standard linear regression coefficient
    gamma = T.sum(zipf_x * diff_stats) / zipf_x2

    return 1 / gamma


@T.jit.script
def aggregator(
    values: T.FloatTensor,
    shadow_low_mul: float,
    shadow_high_mul: float,
    log_noise: T.FloatTensor,
    zipf_x: T.FloatTensor,
    zipf_x2: T.FloatTensor,
) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor, T.FloatTensor, T.FloatTensor]:
    """
    Aggregates several mini-batch summary statistics: "empirical" mean (strong LLN
    approach), min/max, uses power law heuristics to estimate the shadow mean, and
    the tail exponent.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        shadow_low_mul: lower bound multiplier of sample minimum to form minimum threshold of interest
        shadow_high_mul: upper bound multiplier of sample maximum to form upper limit
        log_noise: miniscule constant to tail estimation logarithm bounded
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis

    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: shadow mean
        alpha: tail index of power law
    """
    mean, min, max = T.mean(values), T.min(values), T.max(values)

    low, high = T.min(values) * shadow_low_mul, T.max(values) * shadow_high_mul
    alpha = zipf_plot(values, log_noise, zipf_x, zipf_x2)

    # upper incomplete gamma function valid only for alpha, high > 0
    up_gamma = T.exp(T.lgamma(1 - alpha)) * (1 - T.igamma(1 - alpha, alpha / high))

    # shadow mean estimate
    shadow = (
        low + (high - low) * T.exp(alpha / high) * (alpha / high) ** alpha * up_gamma
    )

    return mean, min, max, shadow, alpha


@T.jit.script
def aggregator_fast(
    values: T.FloatTensor,
    log_noise: T.FloatTensor,
    zipf_x: T.FloatTensor,
    zipf_x2: T.FloatTensor,
) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor, None, T.FloatTensor]:
    """
    Aggregates several mini-batch summary statistics: "empirical" mean (strong LLN
    approach), min/max, and the tail exponent. Shadow means are not calcauted.

    Parameters:
        values: critic loss per sample in the mini-batch without aggregation
        log_noise: miniscule constant to tail estimation logarithm bounded
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis

    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: temporary placeholder for shadow mean
        alpha: tail index of power law
    """
    mean, min, max = T.mean(values), T.min(values), T.max(values)
    alpha = zipf_plot(values, log_noise, zipf_x, zipf_x2)

    return mean, min, max, None, alpha


@T.jit.script
def loss_function(
    estimated_1: T.FloatTensor,
    scale_1: float,
    kernel_1: float,
    estimated_2: T.FloatTensor,
    scale_2: float,
    kernel_2: float,
    target: T.FloatTensor,
    batch_size: int,
    optimise_count: int,
    log_noise: T.FloatTensor,
    zipf_x: T.FloatTensor,
    zipf_x2: T.FloatTensor,
    loss_type: str,
) -> Tuple[
    T.FloatTensor,
    T.FloatTensor,
    T.FloatTensor,
    None,
    T.FloatTensor,
    T.FloatTensor,
    T.FloatTensor,
    T.FloatTensor,
    None,
    T.FloatTensor,
]:
    """
    Gives scalar critic loss value (retaining graph) for network backpropagation.

    Parameters:
        estimated: current Q-values from mini-batch
        scale (float>0): current Cauchy scale parameter
        kernel (float>0): standard deviation for CIM
        target: raget Q-values from mini-batch
        batch_size: mini-batch size
        optimise_count: number of largest critic losses to minimise
        log_noise: miniscule constant to tail estimation logarithm bounded
        zipf_x: array for Zipf plot x-axis
        zipf_x2: sum of squared deviations form the mean for Zipf plot x-axis
        loss_type: loss function title

    Returns:
        mean: empirical mean
        min: minimum critic loss
        max: maximum critic loss
        shadow: temporary placeholder for shadow mean
        alpha: tail index
    """
    if loss_type == "MSE":
        loss_1 = mse(estimated_1, target, 0)
        loss_2 = mse(estimated_2, target, 0)

    elif loss_type == "HUB":
        loss_1 = huber(estimated_1, target)
        loss_2 = huber(estimated_2, target)

    elif loss_type == "MAE":
        loss_1 = mae(estimated_1, target)
        loss_2 = mae(estimated_2, target)

    elif loss_type == "HSC":
        loss_1 = hypersurface(estimated_1, target)
        loss_2 = hypersurface(estimated_2, target)

    elif loss_type == "CAU":
        loss_1 = cauchy(estimated_1, target, scale_1)
        loss_2 = cauchy(estimated_2, target, scale_2)

    elif loss_type == "TCAU":
        estimated_1, target_1 = truncation(estimated_1, target)
        estimated_2, target_2 = truncation(estimated_2, target)
        loss_1 = cauchy(estimated_1, target_1, scale_1)
        loss_2 = cauchy(estimated_2, target_2, scale_2)

    elif loss_type == "CIM":
        loss_1 = correntropy(estimated_1, target, kernel_1)
        loss_2 = correntropy(estimated_2, target, kernel_2)

    else:
        # higher even orders of MSE: MSE2, MSE4, MSE6
        loss_1 = mse(estimated_1, target, int(loss_type[3:]))
        loss_2 = mse(estimated_2, target, int(loss_type[3:]))

    # fractional Kelly i.e. emphasis placed on improving largest errors
    if batch_size > optimise_count:
        q_loss = loss_1 + loss_2
        _, sort = q_loss.sort(descending=True, dim=0)
        sort = sort[:optimise_count]
        loss_1, loss_2 = loss_1[sort], loss_2[sort]

    mean_1, min_1, max_1, shadow_1, alpha_1 = aggregator_fast(
        loss_1, log_noise, zipf_x, zipf_x2
    )
    mean_2, min_2, max_2, shadow_2, alpha_2 = aggregator_fast(
        loss_2, log_noise, zipf_x, zipf_x2
    )

    return (
        mean_1,
        min_1,
        max_1,
        shadow_1,
        alpha_1,
        mean_2,
        min_2,
        max_2,
        shadow_2,
        alpha_2,
    )
