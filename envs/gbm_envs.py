"""
title:                  gbm_envs.py
python version:         3.10
gym version:            0.24

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://www.github.com/rajabinks

Description:
    OpenAI Gym compatible environments for training an agent on various gambles
    for assets following geometric Brownian motion based on
    https://www.tandfonline.com/doi/pdf/10.1080/14697688.2010.513338?needAccess=true,
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.110.100603, and
    https://arxiv.org/pdf/1802.02939.pdf.

    Historical financial data is obtained from Stooq at https://stooq.com/.
"""

import sys

sys.path.append("./")

from typing import List, Tuple

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

NDArrayFloat = npt.NDArray[np.float_]

from tests.test_input_envs import gbm_env_tests, multi_env_tests
from tools.env_resources import multi_gbm_dones

# fmt: off

MAX_VALUE = 1e18            # maximium potfolio value for normalisation
INITIAL_PRICE = 1e3         # initial price of all assets
INITIAL_VALUE = 1e4         # initial portfolio value
MIN_VALUE_RATIO = 1e-2      # minimum portfolio value ratio (psi) relative to MAX_VALUE
MAX_VALUE_RATIO = 1         # maximum possible value relative to MAX_VALUE
MIN_PRICE = 1e-2            # minimum possible price of any asset

MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-3           # minimum step reward (epsilon_2)
MIN_RETURN = np.log(0.1)    # minimum step return (epsilon_3)
MAX_RETURN = 1e10           # maximum step return
MIN_WEIGHT = 1e-5           # minimum all asset weights (epsilon_4)

# hyperparameters for investors A-C GBM gamble
DRIFT = 0.0540025395205692  # S&P500 mean annual log return from 1900->2021 (121 years) ending December 31
VOL = 0.1897916175617430    # S&P500 sample standard deviation from 1900->2021 (121 years) ending December 31
LEV_FACTOR = 5              # maximum (absolute) leverage per assset (eta)

# fmt: on

LOG_MEAN = DRIFT - VOL**2 / 2  # mean of lognormal distribution for S&P500 prices

# conduct tests
DISCRETE_RETURNS = False
multi_env_tests(
    "GBM",
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

gbm_env_tests(DRIFT, VOL, LEV_FACTOR)

# environment constraints
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
LOW_RETURN = min(-np.inf, np.exp(MIN_RETURN) - 1)
MIN_OBS_SPACE = min(
    LOW_RETURN, MIN_VALUE / INITIAL_VALUE - 1, MIN_REWARD / MAX_VALUE, MIN_VALUE
)
MAX_OBS_SPACE = max(np.inf, 1)


class GBM_InvA(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step
    for the GBM approximation of the S&P500 index.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_InvA, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((3 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_gambles), dtype=np.float64)

        self.next_state = np.empty((4 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_gambles,),
            dtype=np.float64,
        )

        # action space: [leverages]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(n_gambles,),
            dtype=np.float64,
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

        # obtain actions from neural network
        lev = action * LEV_FACTOR

        # sample new price % change factor
        r = np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles)

        # one-step portfolio return
        step_return = np.maximum(np.sum(lev * r), MIN_RETURN)
        step_exp_return = np.minimum(np.exp(step_return), 1 + MAX_RETURN)

        # determine change in values
        self.wealth = initial_wealth * step_exp_return

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, MIN_VALUE, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_gbm_dones(
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
            self.next_state,
            MAX_VALUE_RATIO,
            None,
        )

        self.risk[0:4] = [reward, self.wealth, step_return, np.mean(lev)]

        if self.n_gambles > 1:
            self.risk[4:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + self.n_gambles), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_gambles), dtype=np.float64)
        state /= MAX_VALUE

        return state


class GBM_InvB(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss]
    at each time step for the GBM approximation of the S&P500 index.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_InvB, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((4 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_gambles), dtype=np.float64)

        self.next_state = np.empty((4 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_gambles,),
            dtype=np.float64,
        )

        # action space: [leverages]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(1 + n_gambles,),
            dtype=np.float64,
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

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        lev = action[1:] * LEV_FACTOR

        # sample new price % change factor
        r = np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles)

        # one-step portfolio return
        step_return = np.maximum(np.sum(lev * r), MIN_RETURN)
        step_exp_return = np.minimum(np.exp(step_return), 1 + MAX_RETURN)

        # amount of portoflio to bet and outcome
        min_wealth = np.maximum(INITIAL_VALUE * stop_loss, MIN_VALUE)
        active = np.maximum(initial_wealth - min_wealth, 0)
        change = active * step_exp_return

        # determine change in values
        self.wealth = min_wealth + change

        # calculate the step reward as 1 + time-average growth rated
        self.wealth = np.clip(self.wealth, min_wealth, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_gbm_dones(
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
            self.next_state,
            MAX_VALUE_RATIO,
            active,
        )

        self.risk[0:5] = [reward, self.wealth, step_return, np.mean(lev), stop_loss]

        if self.n_gambles > 1:
            self.risk[5:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + self.n_gambles), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_gambles), dtype=np.float64)
        state /= MAX_VALUE

        return state


class GBM_InvC(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss,
    retention ratio] at each time step for the GBM approximation of the S&P500 index.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_gambles: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_gambles: number of simultaneous identical gambles
        """
        super(GBM_InvC, self).__init__()

        self.n_gambles = n_gambles

        if n_gambles == 1:
            self.risk = np.empty((5 + n_gambles), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_gambles), dtype=np.float64)

        self.next_state = np.empty((4 + n_gambles), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_gambles,),
            dtype=np.float64,
        )

        # action space: [leverages]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(2 + n_gambles,),
            dtype=np.float64,
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

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        retention = (action[1] + MAX_ABS_ACTION) / 2
        lev = action[2:] * LEV_FACTOR

        # sample new price % change factor
        r = np.random.normal(loc=LOG_MEAN, scale=VOL, size=self.n_gambles)

        # one-step portfolio return
        step_return = np.maximum(np.sum(lev * r), MIN_RETURN)
        step_exp_return = np.minimum(np.exp(step_return), 1 + MAX_RETURN)

        # bet portion of existing profit at each step
        if initial_wealth <= INITIAL_VALUE:
            # revert to investor B
            min_wealth = np.maximum(INITIAL_VALUE * stop_loss, MIN_VALUE)
        else:
            min_wealth = INITIAL_VALUE + (initial_wealth - INITIAL_VALUE) * retention

        active = np.maximum(initial_wealth - min_wealth, 0)
        change = active * step_exp_return

        # determine change in values
        self.wealth = min_wealth + change

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, min_wealth, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = multi_gbm_dones(
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
            self.next_state,
            MAX_VALUE_RATIO,
            active,
        )

        self.risk[0:6] = [
            reward,
            self.wealth,
            step_return,
            np.mean(lev),
            stop_loss,
            retention,
        ]

        if self.n_gambles > 1:
            self.risk[6:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE

        state = np.empty((4 + self.n_gambles), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_gambles), dtype=np.float64)
        state /= MAX_VALUE

        return state
