"""
title:                  coin_flip_envs.py
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
    OpenAI Gym compatible environments for training an agent on various binary
    coin flip gambles based on
    https://aip.scitation.org/doi/pdf/10.1063/1.4940236 and
    https://www.nature.com/articles/s41567-019-0732-0.pdf.
"""

import sys

sys.path.append("./")

from typing import List, Tuple

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

NDArrayFloat = npt.NDArray[np.float_]

from tests.input_tests import coin_flip_env_tests, multi_env_tests
from tools.env_resources import multi_dones

# fmt: off

MAX_VALUE = 1e18            # maximium potfolio value for normalisation
INITIAL_PRICE = 1e3         # initial price of all assets
INITIAL_VALUE = 1e4         # initial portfolio value
MIN_VALUE_RATIO = 1e-2      # minimum portfolio value ratio (psi) relative to MAX_VALUE
MAX_VALUE_RATIO = 1         # maximum possible value relative to MAX_VALUE

MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-3           # minimum step reward (epsilon_2)
MIN_RETURN = -0.9           # minimum step return (epsilon_3)
MAX_RETURN = 1e10           # maximum step return
MIN_WEIGHT = 1e-5           # minimum all asset weights (epsilon_4)

# hyperparameters for the coin flip gamble for investors A-C
UP_PROB = 1 / 2             # probability of up move
UP_R = 0.5                  # upside return (>0)
DOWN_R = -0.4               # downside return (0<)

# fmt: on

# conduct tests
DISCRETE_RETURNS = True
multi_env_tests(
    "Coin Flip",
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

coin_flip_env_tests(UP_PROB, UP_R, DOWN_R)

# maximum (absolute) leverage per assset (eta)
if np.abs(UP_R) < np.abs(DOWN_R):
    LEV_FACTOR = 1 / np.abs(DOWN_R)
else:
    LEV_FACTOR = 1 / np.abs(UP_R)

# environment constraints
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
LOW_RETURN = min(DOWN_R, MIN_RETURN)
MIN_OBS_SPACE = min(
    LOW_RETURN / MAX_VALUE,
    MIN_VALUE / INITIAL_VALUE - 1,
    MIN_REWARD / MAX_VALUE,
    MIN_VALUE,
)
MAX_OBS_SPACE = max(UP_R, np.inf, 1)


class Coin_InvA(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverage at each time step
    for the coin flip gamble.

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
        super(Coin_InvA, self).__init__()

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

        # sample binary return
        r = np.random.choice(
            [UP_R, DOWN_R], p=[UP_PROB, 1 - UP_PROB], size=self.n_gambles
        )

        # one-step portfolio return
        step_return = np.clip(np.sum(lev * r), MIN_RETURN, MAX_RETURN)

        # determine change in values
        self.wealth = initial_wealth * (1 + step_return)

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, MIN_VALUE, MAX_VALUE)
        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

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


class Coin_InvB(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss]
    at each time step for the coin flip gamble.

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
        super(Coin_InvB, self).__init__()

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
        stop_loss = np.abs(action[0])
        lev = action[1:] * LEV_FACTOR

        # sample binary return
        r = np.random.choice(
            [UP_R, DOWN_R], p=[UP_PROB, 1 - UP_PROB], size=self.n_gambles
        )

        # one-step portfolio return
        step_return = np.clip(np.sum(lev * r), MIN_RETURN, MAX_RETURN)

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
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

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


class Coin_InvC(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss,
    retention ratio] at each time step for the coin flip gamble.

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
        super(Coin_InvC, self).__init__()

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

        # sample binary return
        r = np.random.choice(
            [UP_R, DOWN_R], p=[UP_PROB, 1 - UP_PROB], size=self.n_gambles
        )

        # one-step portfolio return
        step_return = np.clip(np.sum(lev * r), MIN_RETURN, MAX_RETURN)

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
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = r
        self.next_state /= MAX_VALUE

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
