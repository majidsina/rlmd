"""
title:                  dice_flip_sh_envs.py
python version:         3.10
gym version:            0.23

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    OpenAI Gym compatible environments for training an agent on various three-state
    dice roll gambles with insurance safe havens based on
    https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797.
"""

import sys

sys.path.append("./")

from typing import List, Tuple

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

NDArrayFloat = npt.NDArray[np.float_]

from tests.input_tests import dice_roll_sh_env_tests, multi_env_tests
from tools.env_resources import multi_dones

# fmt: off

MAX_VALUE = 1e18            # maximium potfolio value for normalisation
INITIAL_PRICE = 1e3         # initial price of all assets
INITIAL_VALUE = 1e4         # initial portfolio value
MIN_VALUE_RATIO = 1e-2      # minimum portfolio value ratio (psi) relative to MAX_VALUE
MAX_VALUE_RATIO = 1         # maximum possible value relative to MAX_VALUE

MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-6           # minimum step reward (epsilon_2)
MIN_RETURN = -0.99          # minimum step return (epsilon_3)
MAX_RETURN = 1e10           # maximum step return
MIN_WEIGHT = 1e-5           # minimum all asset weights (epsilon_4)

# hyperparameters for the dice roll gamble for investors A-C
UP_PROB = 1 / 6             # probability of up move
DOWN_PROB = 1 / 6           # probability of down move
UP_R = 0.5                  # upside return (UP_R>MID_R>=0)
DOWN_R = -0.5               # downside return (DOWN_R<=0)
MID_R = 0.05                # mid return

# hyperparameters for the insurance safe haven
SH_UP_R = -1                # safe haven upside return (<0)
SH_DOWN_R = 5               # safe haven downside return (>0)
SH_MID_R = -1               # safe haven mid return (SH_DOWN_R>SH_MID_R>=SH_UP_R)

# fmt: on

MID_PROB = 1 - (UP_PROB + DOWN_PROB)
SH_UP_R = max(SH_UP_R, MIN_RETURN)
SH_MID_R = max(SH_MID_R, MIN_RETURN)

# -100% = SH_DOWN_R - (SH_DOWN_R - DOWN_R) * l_max for DOWN_R < 0 and SH_DOWN_R > 0
I_LEV_FACTOR = (-1 - SH_DOWN_R) / (DOWN_R - SH_DOWN_R)
SH_LEV_FACTOR = 1

# conduct tests
DISCRETE_RETURNS = True
multi_env_tests(
    "Dice Roll (Safe Haven)",
    MAX_VALUE,
    INITIAL_PRICE,
    INITIAL_VALUE,
    MIN_VALUE_RATIO,
    MAX_VALUE_RATIO,
    DISCRETE_RETURNS,
    MAX_ABS_ACTION,
    MIN_REWARD,
    MIN_RETURN,
    MAX_RETURN,
    MIN_WEIGHT,
)

dice_roll_sh_env_tests(
    UP_PROB,
    DOWN_PROB,
    UP_R,
    DOWN_R,
    MID_R,
    SH_UP_R,
    SH_DOWN_R,
    SH_MID_R,
    I_LEV_FACTOR,
    SH_LEV_FACTOR,
)

# maximum (absolute) leverage per assset type (eta)
if np.abs(UP_R) < np.abs(DOWN_R):
    LEV_FACTOR = 1 / np.abs(DOWN_R)
else:
    LEV_FACTOR = 1 / np.abs(UP_R)

# environment constraints
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
LOW_RETURN = min(DOWN_R, SH_UP_R, MIN_RETURN)
MIN_OBS_SPACE = min(
    LOW_RETURN / MAX_VALUE,
    MIN_VALUE / INITIAL_VALUE - 1,
    MIN_REWARD / MAX_VALUE,
    MIN_VALUE,
)
MAX_OBS_SPACE = max(UP_R, SH_DOWN_R, np.inf, 1)


class Dice_SH_INSURED(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step
    for the dice roll gamble with safe haven.

    Replicates Chapter 3 - Side Bets from
    https://www.wiley.com/en-us/Safe+Haven%3A+Investing+for+Financial+Storms-p-9781119401797

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_INSURED, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE, high=MAX_OBS_SPACE, shape=(4 + 2,), dtype=np.float64
        )

        # action space: [leverage 0]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, shape=(1,), dtype=np.float64
        )

        self.reset()

    def step(
        self, action: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of additional data retrieved from each step
        """
        initial_wealth = self.wealth

        # obtain leverage from neural network
        lev = action * I_LEV_FACTOR
        lev_sh = 1 - lev

        # sample returns
        r = np.random.choice(
            [UP_R, DOWN_R, MID_R], p=[UP_PROB, DOWN_PROB, MID_PROB], size=1
        )[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = np.clip(sum(lev * r + lev_sh * r_sh), MIN_RETURN, MAX_RETURN)

        # determine change in values
        self.wealth = initial_wealth * (1 + step_return)

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, MIN_VALUE, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        next_state = np.array(
            [self.wealth, step_return, growth, reward, r, r_sh], dtype=np.float64
        )
        next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_dones(
            self.wealth,
            MIN_VALUE,
            reward,
            MIN_REWARD,
            step_return,
            MIN_RETURN,
            lev,
            MIN_WEIGHT,
            MAX_ABS_ACTION,
            LEV_FACTOR,
            next_state,
            MAX_VALUE_RATIO,
            None,
        )

        risk = np.array(
            [reward, self.wealth, step_return, lev, np.nan, np.nan, lev_sh],
            dtype=np.float64,
        )

        self.time += 1

        return next_state, reward, done, risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + 2), dtype=np.float64)
        state[:] = [self.wealth, 0, 1, 1, 0, 0]
        state /= MAX_VALUE

        return state


class Dice_SH_InvA(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step
    for the dice roll gamble with safe haven.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_InvA, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE, high=MAX_OBS_SPACE, shape=(4 + 2,), dtype=np.float64
        )

        # action space: [leverage 0, safe haven leverage]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, shape=(2,), dtype=np.float64
        )

        self.reset()

    def step(
        self, action: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, bool, NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth

        # obtain leverage from neural network
        lev = action[0] * LEV_FACTOR
        lev_sh = (action[1] + MAX_ABS_ACTION) / 2 * SH_LEV_FACTOR

        # sample returns
        r = np.random.choice(
            [UP_R, DOWN_R, MID_R], p=[UP_PROB, DOWN_PROB, MID_PROB], size=1
        )[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = np.clip(lev * r + lev_sh * r_sh, MIN_RETURN, MAX_RETURN)

        # determine change in values
        self.wealth = initial_wealth * (1 + step_return)

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, MIN_VALUE, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        next_state = np.array(
            [self.wealth, step_return, growth, reward, r, r_sh], dtype=np.float64
        )
        next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_dones(
            self.wealth,
            MIN_VALUE,
            reward,
            MIN_REWARD,
            step_return,
            MIN_RETURN,
            lev,
            MIN_WEIGHT,
            MAX_ABS_ACTION,
            LEV_FACTOR,
            next_state,
            MAX_VALUE_RATIO,
            None,
        )

        risk = np.array(
            [reward, self.wealth, step_return, lev, np.nan, np.nan, lev_sh],
            dtype=np.float64,
        )

        self.time += 1

        return next_state, reward, done, risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + 2), dtype=np.float64)
        state[:] = [self.wealth, 0, 1, 1, 0, 0]
        state /= MAX_VALUE

        return state


class Dice_SH_InvB(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at each
    time step for the dice roll gamble with safe haven.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_InvB, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE, high=MAX_OBS_SPACE, shape=(4 + 2,), dtype=np.float64
        )

        # action space: [leverage 0, safe have leverage, stop-loss]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, shape=(3,), dtype=np.float64
        )

        self.reset()

    def step(
        self, action: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, bool, NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth

        # obtain leverages and stop-loss from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        lev = action[1] * LEV_FACTOR
        lev_sh = (action[2] + MAX_ABS_ACTION) / 2 * SH_LEV_FACTOR

        # sample returns
        r = np.random.choice(
            [UP_R, DOWN_R, MID_R], p=[UP_PROB, DOWN_PROB, MID_PROB], size=1
        )[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = np.clip(lev * r + lev_sh * r_sh, MIN_RETURN, MAX_RETURN)

        # amount of portoflio to bet and outcome
        min_wealth = np.maximum(INITIAL_VALUE * stop_loss, MIN_VALUE)
        active = np.maximum(initial_wealth - min_wealth, 0)
        change = active * (1 + step_return)

        # determine change in values

        self.wealth = min_wealth + change

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, min_wealth, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        next_state = np.array(
            [self.wealth, step_return, growth, reward, r, r_sh], dtype=np.float64
        )
        next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_dones(
            self.wealth,
            min_wealth,
            reward,
            MIN_REWARD,
            step_return,
            MIN_RETURN,
            lev,
            MIN_WEIGHT,
            MAX_ABS_ACTION,
            LEV_FACTOR,
            next_state,
            MAX_VALUE_RATIO,
            active,
        )

        risk = np.array(
            [reward, self.wealth, step_return, lev, stop_loss, np.nan, lev_sh],
            dtype=np.float64,
        )

        self.time += 1

        return next_state, reward, done, risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + 2), dtype=np.float64)
        state[:] = [self.wealth, 0, 1, 1, 0, 0]
        state /= MAX_VALUE

        return state


class Dice_SH_InvC(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss,
    retention ratio] at each time step for the dice roll gamble with safe haven.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.
        """
        super(Dice_SH_InvC, self).__init__()

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset 0, safe haven]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE, high=MAX_OBS_SPACE, shape=(4 + 2,), dtype=np.float64
        )

        # action space: [leverage 0, safe haven leverage, stop-loss, retention ratio]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION, high=MAX_ABS_ACTION, shape=(4,), dtype=np.float64
        )

        self.reset()

    def step(
        self, action: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, bool, NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth

        # obtain leverages, stop-loss, and retention ratio from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        retention = (action[1] + MAX_ABS_ACTION) / 2
        lev = action[2] * LEV_FACTOR
        lev_sh = (action[3] + MAX_ABS_ACTION) / 2 * SH_LEV_FACTOR

        # sample returns
        r = np.random.choice(
            [UP_R, DOWN_R, MID_R], p=[UP_PROB, DOWN_PROB, MID_PROB], size=1
        )[0]

        # one-step portfolio return
        if r == MID_R:
            r_sh = SH_MID_R
        elif r == UP_R:
            r_sh = SH_UP_R
        else:
            r_sh = SH_DOWN_R

        step_return = np.clip(lev * r + lev_sh * r_sh, MIN_RETURN, MAX_RETURN)

        # bet portion of existing profit at each step
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor B
            min_wealth = np.maximum(INITIAL_VALUE * stop_loss, MIN_VALUE)
        else:
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention

        active = np.maximum(initial_wealth - min_wealth, 0)
        change = active * (1 + step_return)

        # determine change in values

        self.wealth = min_wealth + change

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, min_wealth, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        next_state = np.array(
            [self.wealth, step_return, growth, reward, r, r_sh], dtype=np.float64
        )
        next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_dones(
            self.wealth,
            min_wealth,
            reward,
            MIN_REWARD,
            step_return,
            MIN_RETURN,
            lev,
            MIN_WEIGHT,
            MAX_ABS_ACTION,
            LEV_FACTOR,
            next_state,
            MAX_VALUE_RATIO,
            active,
        )

        risk = np.array(
            [reward, self.wealth, step_return, lev, stop_loss, retention, lev_sh],
            dtype=np.float64,
        )

        self.time += 1

        return next_state, reward, done, risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + 2), dtype=np.float64)
        state[:] = [self.wealth, 0, 1, 1, 0, 0]
        state /= MAX_VALUE

        return state
