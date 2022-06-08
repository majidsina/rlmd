"""
title:                  gud_resources.py
python version:         3.10

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://github.com/rajabinks

Description:
    Collection of tools needed for various guidance environments.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float_]


def guidance_dones(
    dist: float,
    THRESHOLD_RAD: float,
    MAX_LENGTH: float,
    time: int,
    t_min: int,
    MIN_TIME_RATIO: float,
    reward: float,
    MIN_REWARD: float,
    step_return: float,
    MIN_RETURN: float,
) -> List[bool]:
    """
    Agent done flags for guidance environments controlling episode termination and
    Q-value estimation for learning (i.e. whether genuine or forced).

    Parameters:
        dist: distance from target
        time: episode time step
        t_min: minimum theoretical time to reach target from initialisation
        reward: guidance reward signal
        step_return: single step return

    Returns:
        done: episode termination
        learn_done: agent learning done flag
    """
    done_threshold = dist <= THRESHOLD_RAD
    done_time_initial = dist <= THRESHOLD_RAD and time == 1

    done_dist = dist > MAX_LENGTH

    done_time = t_min / time < MIN_TIME_RATIO

    done_reward = reward < MIN_REWARD
    done_return = step_return <= MIN_RETURN

    done = bool(
        done_threshold
        or done_time_initial
        or done_dist
        or done_time
        or done_reward
        or done_return
    )

    learn_done = done and not (done_threshold or done_time_initial)

    return [
        done,
        learn_done,
        # temporary: all dones are logged for diagnostic purposes
        done_threshold,
        done_time_initial,
        done_dist,
        done_time,
        done_reward,
        done_return,
    ]


def laminar_flow(
    velocity: float, wind_reduction: float, three_d: bool
) -> Tuple[float, float, float]:
    """
    Generate constant external vector field acting as laminar flow.

    Parameters:
        velocity: projectile velocity
        wind_reduction: offset for laminar flow magnitude
        three_d: whether 3D environment

    Return:
        rho: magnitude of field
        azimuth: azimuth of field
        incline: polar inclination of field
    """
    rho = np.random.uniform(low=0, high=velocity - wind_reduction)
    azimuth = np.random.uniform(low=0, high=2 * np.pi)

    if three_d == False:
        return rho, azimuth, None

    incline = np.random.uniform(low=0, high=np.pi)

    return rho, azimuth, incline


def laminar_flow_peturb(
    velocity: float,
    wind_reduction: float,
    rho: float,
    rho_i: float,
    rho_vol: float,
    azimuth: float,
    azimuth_vol: float,
    incline: float,
    incline_vol: float,
    three_d: bool,
) -> Tuple[float, float, float]:
    """
    Generate a new field acting as a varying laminar flow by perturbating
    the previous field.

    Parameters:
        velocity: projectile velocity
        wind_reduction: offset for laminar flow magnitude
        rho: current magnitude of field
        rho_i: initial magnitude of field
        rho_vol: volatility of rho
        azimuth: current azimuth of field
        azimuth_vol: volatility of azimuth
        incline: current polar inclination of field
        incline_vol: volatility of incline
        three_d: whether 3D environment

    Return:
        rho_new: new perturbated magnitude of field
        azimuth_new: new perturbated azimuth of field
        incline_new: new perturbated polar inclination of field
    """
    rho_new = np.random.normal(loc=rho, scale=rho_i * rho_vol)
    rho_new = np.clip(rho_new, 0, velocity - wind_reduction)

    azimuth_new = np.random.normal(loc=azimuth, scale=2 * np.pi * azimuth_vol)

    if three_d == False:
        return rho_new, azimuth_new, None

    incline_new = np.random.normal(loc=incline, scale=np.pi * incline_vol)

    return rho_new, azimuth_new, incline_new


def distance(target: NDArrayFloat, position: NDArrayFloat, order: int = 2) -> float:
    """
    Calculates the Minkowski distance of order p between two points. Euclidean
    distance (p=2) and the Manhattan distance (p=1) are most common.

    Parameters:
        target: location of target
        position: current position
        order: distance order p

    Returns:
        distance: Minkowski distance of order p
    """
    dist_p = np.abs(target - position) ** order

    return (dist_p.sum()) ** (1 / order)


def distance_multiple(
    target: NDArrayFloat, position: NDArrayFloat, order: int = 2
) -> NDArrayFloat:
    """
    Calculates the Minkowski distances of order p between multiple sets of two points.
    Euclidean distance (p=2) and the Manhattan distance (p=1) are most common.

    Parameters:
        target: location of target
        position: current position
        order: distance order p

    Returns:
        distances: Minkowski distances of order p
    """
    dist_p = np.abs(target - position) ** order

    return (dist_p.sum(axis=1)) ** (1 / order)


def reward_components(
    dist: float, dist_thr: float, time: int, t_min: int, div_offset: float
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Calculate components of guidance reward for generating state history.

    Parameters:
        dist: distance between current position and target
        dist_thr: minimum threshold distance for success
        time: current time step
        t_min: minimum theoretical time to reach target from initialisation
        div_offset: offset to prevent divergence (or division by zero) for distance

    Returns:
        dist_multiplier: distance component of reward
        time_multiplier: time component of reward
    """
    dist_multiplier = dist_thr / (dist + div_offset)

    time_multiplier = t_min / time

    return dist_multiplier, time_multiplier


def reward_signal(
    dist: float,
    dist_thr: float,
    time: int,
    t_min: int,
    alpha: float,
    beta: float,
    harsh_r: bool,
) -> float:
    """
    Calculate guidance reward signal used for agent learning.

    Parameters:
        dist: distance between current position and target
        dist_thr: minimum threshold distance for success
        time: current time step
        t_min: minimum theoretical time to reach target from initialisation
        alpha: index for distance component
        beta: index for time component
        harsh_r: whether harsh or lenient reward aggregation

    Return:
        reward: reward signal
    """
    #  unity if dist < dist_thr
    dist_multiplier = np.minimum(1, (dist_thr / dist) ** alpha)

    #  unity if time < t_min
    time_multiplier = np.minimum(1, (t_min / time) ** beta)

    if harsh_r:
        return np.sqrt(dist_multiplier * time_multiplier)

    else:
        return (dist_multiplier + time_multiplier) / 2


def reward_components_stage_2(
    targets: NDArrayFloat,
    positions: NDArrayFloat,
    dist_thr: float,
    time: int,
    time_stage_2: int,
    t_min: NDArrayFloat,
    div_offset: float,
    alpha: float,
    beta: float,
    harsh_r: bool,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Calculates all the stage 2 reward signal components.

    Parameters:
        target: location of targets
        position: current positions
        dist_thr: minimum threshold distance for success
        time: current time step
        time_stage_2: time step for transition to second stage
        t_min: minimum theoretical time to reach target from initialisation
        div_offset: offset to prevent divergence (or division by zero) for distance
        alpha: index for distance component
        beta: index for time component
        harsh_r: whether harsh or lenient reward aggregation

    Returns:
        dist_multiplier: distance component of rewards
        time_multiplier: time component of rewards
        rewards_stage_2: stage 2 reward signals
    """
    dists = distance_multiple(targets, positions)
    dist_multipliers = dist_thr / (dists + div_offset)

    time_multipliers = t_min / (time - time_stage_2)

    dist_components = np.minimum(1, dist_multipliers**alpha)
    time_components = np.minimum(1, time_multipliers**beta)

    if harsh_r:
        rewards_stage_2 = np.sqrt(dist_components * time_components)

        return dist_multipliers, time_multipliers, rewards_stage_2

    else:
        rewards_stage_2 = (dist_components + time_components) / 2

        return dist_multipliers, time_multipliers, rewards_stage_2


def reward_signal_two_stage(
    reward_stage_1: float,
    reward_stage_2: NDArrayFloat,
    time_stage_2: int,
    n_targets: int,
    harsh_r: bool,
) -> float:
    """
    Calculate the reward signal used for agent learning in two-stage environments.

    Parameters:
        reward_stage_1: stage 1 reward
        reward_stage_2: stage 2 rewards for each projectile
        time_stage_2: time step for transition to second stage
        n_targets: number of targets for second stage
        harsh_r: whether harsh or lenient reward aggregation

    Return:
        reward: reward signal
    """
    reward = reward_stage_1 / (n_targets + 1)

    if time_stage_2 != None:

        if harsh_r:
            reward_2 = np.prod(reward_stage_2) ** (1 / n_targets)
        else:
            reward_2 = np.mean(reward_stage_2)

        reward += n_targets / (n_targets + 1) * reward_2

    return reward


def reward_components_counter(
    targets: NDArrayFloat,
    positions: NDArrayFloat,
    dist_thr: float,
    time: int,
    t_min: NDArrayFloat,
    div_offset: float,
    alpha: float,
    beta: float,
    harsh_r: bool,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Calculates the countermeasure reward signal components for offensive actor.

    Parameters:
        target: location of targets
        position: current positions
        dist_thr: minimum threshold distance for success
        time: current time step
        t_min: minimum theoretical time to reach target from initialisation
        div_offset: offset to prevent divergence (or division by zero) for distance
        alpha: index for distance component
        beta: index for time component
        harsh_r: whether harsh or lenient reward aggregation

    Returns:
        dist_multiplier: distance component of rewards
        time_multiplier: time component of rewards
        reward_offense: offensive reward signal
    """
    dists = distance_multiple(targets, positions)
    dist_multipliers = dist_thr / (dists + div_offset)

    time_multipliers = t_min / time

    dist_components = np.minimum(1, dist_multipliers**alpha)
    time_components = np.minimum(1, time_multipliers**beta)

    if not harsh_r:
        reward_offense = (dist_components + time_components) / 2

        return dist_multipliers, time_multipliers, reward_offense

    else:
        reward_offense = np.sqrt(dist_components * time_components)

        return dist_multipliers, time_multipliers, reward_offense


def reward_signal_counter(reward_offense: NDArrayFloat, harsh_r: bool) -> float:
    """
    Calculate the reward signal used for agent learning in countermeasure environments.

    Parameters:
        reward_offense: lenient rewards for each offensive agent projectile

    Return:
        reward: reward signal
        harsh_r: whether harsh or lenient reward aggregation
    """
    if not harsh_r:
        return 1 - np.mean(reward_offense)

    else:
        return 1 - np.prod(reward_offense) ** (1 / reward_offense.shape[0])


def create_2d_grid(
    length: int, velocity: float
) -> Tuple[NDArrayFloat, NDArrayFloat, float, int]:
    """
    Generate square grid array for initial and target positions.

    Parameters:
        length: dimension of 2D square grid
        velocity: constant speed of projectile

    Returns:
        initial: initial location of projectile
        target: location of target
        d_min: minimum distance to target
        t_min: minimum time steps to target
    """
    points = length**2

    start_end = np.random.choice(points, size=2, replace=False)
    start, end = start_end[0], start_end[1]

    start_x = int(np.floor(start / length))
    end_x = int(np.floor(end / length))

    start_y = start % length
    end_y = end % length

    initial = np.array([start_x, start_y], dtype=np.float64)
    target = np.array([end_x, end_y], dtype=np.float64)

    d_min = distance(target, initial)
    t_min = int(np.ceil(d_min / velocity))

    return initial, target, d_min, t_min


def create_3d_grid(
    length: int, velocity: float
) -> Tuple[NDArrayFloat, NDArrayFloat, float, int]:
    """
    Generate cube grid array for initial and target positions.

    Parameters:
        length: dimension of 3D cube grid
        velocity: constant speed of projectile

    Returns:
        initial: initial location of projectile
        target: location of target
        d_min: minimum distance to target
        t_min: minimum time steps to target
    """
    points = length**3

    start_end = np.random.choice(points, size=2, replace=False)
    start, end = start_end[0], start_end[1]

    start_x = int(np.floor(start / (length**2)))
    end_x = int(np.floor(end / (length**2)))

    start_2d = start - start_x * length**2
    end_2d = end - end_x * length**2

    start_y = int(np.floor(start_2d / length))
    end_y = int(np.floor(end_2d / length))

    start_z = start_2d % length
    end_z = end_2d % length

    initial = np.array([start_x, start_y, start_z], dtype=np.float64)
    target = np.array([end_x, end_y, end_z], dtype=np.float64)

    d_min = distance(target, initial)
    t_min = int(np.ceil(d_min / velocity))

    return initial, target, d_min, t_min


def create_two_stage(
    length: int,
    height: float,
    midpoint: float,
    cone_max: float,
    velocity_1: float,
    velocity_2: float,
    n_targets: int,
) -> Tuple[
    NDArrayFloat,
    NDArrayFloat,
    float,
    int,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
    NDArrayFloat,
]:
    """
    Generate two-stage payload delivery environment initialisation conditions.

    Parameters:
        length: dimension of 2D square grid at launch height
        height: initial launch height for stage 1
        midpoint: transition height for start of stage 2
        cone_max: maximum range of stage 2 projectiles(s)
        velocity_1: constant speed of stage 1 projectile
        velocity_2: constant speed of stage 2 projectiles(s)
        n_targets: number of projectiles for stage 2

    Returns:
        launch: initial position of stage 1
        mid: transition position for start of stage 2
        d_min_1: minimum distance to mid
        t_min_1: minimum time steps to mid
        mid_launches: array of initial positions of all stage 2 projectiles
        targets: array of all targets for stage 2
        d_min_2: minimum distances to targets
        t_min_2: minimum time steps to targets
    """
    # target locations for incoming offensive projectiles (stage 2)
    max_radius = np.sqrt(cone_max**2 - midpoint**2)

    target_r = np.random.uniform(low=0, high=max_radius, size=n_targets)

    target_azimuth = np.random.uniform(low=0, high=2 * np.pi, size=n_targets)

    targets = np.empty((n_targets, 3), dtype=np.float64)

    for n in range(n_targets):
        x_unit_vec = np.cos(target_azimuth[n])
        y_unit_vec = np.sin(target_azimuth[n])
        targets[n] = target_r[n] * np.array([x_unit_vec, y_unit_vec, 0])

    # stage 1 -> stage 2 transition location
    mid = np.array([0, 0, midpoint], dtype=np.float64)

    n_count = ["mid" for x in range(n_targets)]
    n_count = str(n_count)[1:-1]
    n_count = n_count.replace("'", "")

    mid_launches = np.vstack((eval(n_count)))

    d_min_2 = distance_multiple(targets, mid)
    t_min_2 = np.ceil(d_min_2 / velocity_2)

    # initial launch location (stage 1)
    top = np.random.choice(length, size=2, replace=True)

    launch = np.array([top[0], top[1], height], dtype=np.float64)

    d_min_1 = distance(launch, mid)
    t_min_1 = np.ceil(d_min_1 / velocity_1)

    return launch, mid, d_min_1, t_min_1, mid_launches, targets, d_min_2, t_min_2


def create_countermeasure(
    obs_radius: float,
    target_radius: float,
    min_height: float,
    velocity: float,
    n_targets: int,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Generate countermeasure environment initialisation conditions.

    Parameters:
        obs_radius: initial start radius of incoming projectile(s)
        target_radius: target radius for incoming projectile(s)
        min_height: minimum start height of incoming projectile(s)
        velocity: constant speed of incoming projectile(s)
        n_targets: number of projectiles

    Returns:
        starts: initial positions of incoming projectile(s)
        targets: targets of all incoming projectile(s)
        d_min: minimum distances to targets
        t_min: minimum time steps to targets
    """
    # initial conditions for incoming offensive projectiles
    start_r = np.random.uniform(low=0, high=obs_radius, size=n_targets)

    start_azimuth = np.random.uniform(low=0, high=2 * np.pi, size=n_targets)

    max_incline = np.pi / 2 - np.arctan(min_height / obs_radius)
    start_incline = np.random.uniform(low=0, high=max_incline, size=n_targets)

    starts = np.empty((n_targets, 3), dtype=np.float64)

    for n in range(n_targets):
        x_unit_vec = np.cos(start_azimuth[n]) * np.sin(start_incline[n])
        y_unit_vec = np.sin(start_azimuth[n]) * np.cos(start_incline[n])
        z_unit_vec = np.cos(start_incline[n])

        starts[n] = start_r[n] * np.array([x_unit_vec, y_unit_vec, z_unit_vec])

    # target locations for incoming offensive projectiles
    target_r = np.random.uniform(low=0, high=target_radius, size=n_targets)

    target_azimuth = np.random.uniform(low=0, high=2 * np.pi, size=n_targets)

    targets = np.empty((n_targets, 3), dtype=np.float64)

    for n in range(n_targets):
        x_unit_vec = np.cos(target_azimuth[n])
        y_unit_vec = np.sin(target_azimuth[n])
        targets[n] = target_r[n] * np.array([x_unit_vec, y_unit_vec, 0])

    d_min = distance_multiple(targets, starts)
    t_min = np.ceil(d_min / velocity)

    return starts, targets, d_min, t_min
