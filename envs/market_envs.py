"""
title:                  market_envs.py
python version:         3.10
gym version:            0.24

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <raja (_] grewal1 [at} pm {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal

Description:
    OpenAI Gym compatible environments for training an agent on
    simulated real market environments for all three investor categories.
"""

import sys

sys.path.append("./")

from typing import List, Tuple

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

NDArrayFloat = npt.NDArray[np.float_]

from tests.input_tests import market_env_tests
from tools.env_resources import market_dones

# fmt: off

MAX_VALUE = 1e34            # maximium potfolio value for normalisation
INITIAL_VALUE = 1e4         # initial portfolio value
MIN_VALUE_RATIO = 1e-2      # minimum portfolio value ratio (psi) relative to MAX_VALUE
MAX_VALUE_RATIO = 1         # maximum possible value relative to MAX_VALUE

MAX_ABS_ACTION = 0.99       # maximum normalised (absolute) action value (epsilon_1)
MIN_REWARD = 1e-3           # minimum step reward (epsilon_2)
MIN_RETURN = -0.9           # minimum step return (epsilon_3)
MAX_RETURN = 1e10           # maximum step return
MIN_WEIGHT = 1e-5           # minimum all asset weights (epsilon_4)

LEV_FACTOR = 3              # maximum (absolute) leverage per assset (eta)

# fmt: on

# conduct tests
market_env_tests(
    MAX_VALUE,
    INITIAL_VALUE,
    MIN_VALUE_RATIO,
    MAX_VALUE_RATIO,
    MAX_ABS_ACTION,
    MIN_REWARD,
    MIN_RETURN,
    MAX_RETURN,
    MIN_WEIGHT,
    LEV_FACTOR,
)

# environment constraints
MIN_VALUE = max(MIN_VALUE_RATIO * INITIAL_VALUE, 1)
LOW_RETURN = min(-np.inf, MIN_RETURN)
MIN_OBS_SPACE = min(
    LOW_RETURN, MIN_VALUE / INITIAL_VALUE - 1, MIN_REWARD / MAX_VALUE, MIN_VALUE
)
MAX_OBS_SPACE = max(np.inf, 1)


class Market_InvA_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time
    step for a simulated real market using the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvA_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((3 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        lev = action * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
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
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[4:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_assets), dtype=np.float64)
        state /= MAX_VALUE

        return state


class Market_InvB_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at
    each time step for a simulated real market using the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvB_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages, stop-loss]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(1 + n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        lev = action[1:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
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
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[5:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_assets), dtype=np.float64)
        state /= MAX_VALUE

        return state


class Market_InvC_D1(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss,
    retention ratio] at each time step for a simulated real market using
    the MDP assumption.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed (unused)
        """
        super(Market_InvC_D1, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length

        if n_assets == 1:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_OBS_SPACE,
            high=MAX_OBS_SPACE,
            shape=(4 + n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages, stop-loss, retention ratio]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(2 + n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        retention = (action[1] + MAX_ABS_ACTION) / 2
        lev = action[2:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        r = next_state / assets - 1
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
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[6:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = np.zeros((self.n_assets), dtype=np.float64)
        state /= MAX_VALUE

        return state


class Market_InvA_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal leverages at each time
    step for a simulated real market using a non-MDP assumption incorporating
    multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed
        """
        super(Market_InvA_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((3 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + self.obs_days * n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_VALUE_RATIO,
            high=MAX_OBS_SPACE,
            shape=(4 + obs_days * n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        lev = action * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        history_returns = next_state / assets - 1
        r = history_returns[0 : self.n_assets]
        step_return = np.clip(np.sum(lev * r), MIN_RETURN, MAX_RETURN)

        # determine change in values
        self.wealth = initial_wealth * (1 + step_return)

        # calculate the step reward as 1 + time-average growth rate
        self.wealth = np.clip(self.wealth, MIN_VALUE, MAX_VALUE)

        growth = self.wealth / INITIAL_VALUE

        reward = np.exp(np.log(growth) / self.time)

        # obtain next state
        self.next_state[0:4] = [self.wealth, step_return, growth, reward]
        self.next_state[4:] = history_returns
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[4:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = self.assets
        state /= MAX_VALUE

        return state


class Market_InvB_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss] at
    each time step for a simulated real market using a non-MDP assumption
    incorporating multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed
        """
        super(Market_InvB_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((4 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + self.obs_days * n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_VALUE_RATIO,
            high=MAX_OBS_SPACE,
            shape=(4 + obs_days * n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages, stop-loss]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(1 + n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        lev = action[1:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        history_returns = next_state / assets - 1
        r = history_returns[0 : self.n_assets]
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
        self.next_state[4:] = history_returns
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[5:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = self.assets
        state /= MAX_VALUE

        return state


class Market_InvC_Dx(gym.Env):
    """
    OpenAI gym environment for determining the optimal [leverage, stop-loss,
    retention ratio] at each time step for a simulated real market using a
    non-MDP assumption incorporating multiple past states.

    Methods:
        seed(seed):
            Manually set seed for reproducibility.

        step(actions):
            Obtain next environment state from taking a given set of actions.

        reset():
            Reset enivronment to inital default state.
    """

    def __init__(self, n_assets: int, time_length: int, obs_days: int) -> None:
        """
        Initialise class varaibles by creating state-action space and reward range.

        Parameters:
            n_assets: number of assets
            time_length: maximum training time before termination
            obs_days: number of previous sequential days observed
        """
        super(Market_InvC_Dx, self).__init__()

        self.n_assets = n_assets
        self.time_length = time_length - obs_days + 1
        self.obs_days = obs_days

        if n_assets == 1:
            self.risk = np.empty((5 + n_assets), dtype=np.float64)
        else:
            self.risk = np.empty((6 + n_assets), dtype=np.float64)

        self.next_state = np.empty((4 + self.obs_days * n_assets), dtype=np.float64)

        self.reward_range = (MIN_REWARD, np.inf)

        # state space: [cumulative reward, asset prices, performance measures]
        self.observation_space = spaces.Box(
            low=MIN_VALUE_RATIO,
            high=MAX_OBS_SPACE,
            shape=(4 + obs_days * n_assets,),
            dtype=np.float64,
        )

        # action space: [leverages, stop-loss, retention ratio]
        self.action_space = spaces.Box(
            low=-MAX_ABS_ACTION,
            high=MAX_ABS_ACTION,
            shape=(2 + n_assets,),
            dtype=np.float64,
        )

        self.reset(assets=None)

    def step(
        self, action: NDArrayFloat, next_assets: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float, List[bool], NDArrayFloat]:
        """
        Take action to arrive at next state and calculate reward.

        Parameters:
            action: array of actions to be taken determined by actor network
            next_assets: next sequential state from history

        Returns:
            next_state: state arrived at from taking action
            reward: portfolio geometric mean + 1
            done: Boolean flags for episode termination and whether genuine
            risk: collection of data retrieved from step
        """
        initial_wealth = self.wealth
        assets = self.assets

        # obtain actions from neural network
        stop_loss = (action[0] + MAX_ABS_ACTION) / 2
        retention = (action[1] + MAX_ABS_ACTION) / 2
        lev = action[2:] * LEV_FACTOR

        # receive next set of prices
        next_state = next_assets

        # one-step portfolio return
        history_returns = next_state / assets - 1
        r = history_returns[0 : self.n_assets]
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
        self.next_state[4:] = history_returns
        self.next_state /= MAX_VALUE

        # episode termination criteria
        done = market_dones(
            self.time,
            self.time_length,
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

        if self.n_assets > 1:
            self.risk[6:] = lev

        self.time += 1

        return self.next_state, reward, done, self.risk

    def reset(self, assets: NDArrayFloat) -> NDArrayFloat:
        """
        Reset the environment for a new agent episode.

        Parameters:
            assets: initial asset prices

        Returns:
            state: default initial state
        """
        self.time = 1
        self.wealth = INITIAL_VALUE
        self.assets = assets

        state = np.empty((4 + self.obs_days * self.n_assets), dtype=np.float64)
        state[0:4] = [self.wealth, 0, 1, 1]
        state[4:] = self.assets
        state /= MAX_VALUE

        return state
