"""
title:                  replay_torch.py
python version:         3.10
torch version:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://github.com/rajabinks

Description:
    Responsible for creating experience replay buffer with uniform step sampling
    with GPU acceleration using PyTorch.
"""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch as T

NDArrayFloat = npt.NDArray[np.float_]


class ReplayBufferTorch:
    """
    Experience replay buffer with uniform sampling based on
    https://link.springer.com/content/pdf/10.1023%2FA%3A1022628806385.pdf.

    Methods:
        _episode_history(step, idx, done):
            Store complete history of each training episode.

        store_exp(state, action, reward, next_state, done):
            Stores current step into experience replay buffer.

        _construct_history(step, epis_history):
            Generate reward and state-action pair history for a sampled step.

        _episode_rewards_states_actions(batch):
            Collect reward and state-action pair histories for the batch.

        _multi_step_rewards_states_actions(reward_history, state_history, action_history, multi_length):
            Generate multi-step rewards and initial state-action pair for a sampled step.

        _multi_step_batch(step_rewards, step_states, step_actions):
            Collect multi-step rewards and initial state-action pairs for the batch.

        sample_exp():
            Uniformly sample mini-batch from experience replay buffer.
    """

    def __init__(self, inputs: dict) -> None:
        """
        Initialise class varaibles by creating empty numpy buffer arrays.

        Paramters:
            inputs: dictionary containing all execution details
        """
        self.device = T.device(inputs["gpu"] if T.cuda.is_available() else "cpu")

        self.input_dims = sum(inputs["input_dims"])
        self.num_actions = int(inputs["num_actions"])
        self.batch_size = int(inputs["mini_batch_size"])
        self.gamma = inputs["discount"]
        self.multi_steps = T.tensor(int(inputs["multi_steps"]), device=self.device)

        if inputs["r_abs_zero"] == None:
            self.r_abs_zero = -np.inf
        else:
            self.r_abs_zero = inputs["r_abs_zero"]

        self.dyna = str(inputs["dynamics"])

        if int(inputs["buffer"]) <= int(inputs["n_cumsteps"]):
            self.mem_size = int(inputs["buffer"])
        else:
            self.mem_size = int(inputs["n_cumsteps"])

        self.mem_idx = 0

        self.state_memory = T.empty(
            (self.mem_size, self.input_dims), dtype=T.float, device=self.device
        )
        self.action_memory = T.empty(
            (self.mem_size, self.num_actions), dtype=T.float, device=self.device
        )
        self.reward_memory = T.empty(
            (self.mem_size,), dtype=T.float, device=self.device
        )
        self.next_state_memory = T.empty(
            (self.mem_size, self.input_dims), dtype=T.float, device=self.device
        )
        self.terminal_memory = T.empty(
            (self.mem_size,), dtype=T.bool, device=self.device
        )
        self.eff_length = T.ones((1,), dtype=T.int, device=self.device)

        # multi-step return features
        self.epis_idx = [float("nan")]
        self.epis_reward_memory = []
        self.epis_state_memory = []
        self.epis_action_memory = []

        self.multi_rewards = T.empty((self.batch_size,), device=self.device)
        self.multi_states = T.empty(
            (self.batch_size, self.input_dims), device=self.device
        )
        self.multi_actions = T.empty(
            (self.batch_size, self.num_actions), device=self.device
        )

    def _episode_history(self, idx: int, done: bool) -> None:
        """
        Aggerate and store the history of each training episode for multi-step learning
        including the rewards, states, and actions.

        Parameters:
            idx: training step index
            done: whether episode terminal
        """
        self.epis_idx[-1] = idx

        # aggregate the reward, state, and action histories of the currently trained episode
        try:
            current_start = self.epis_idx[-2] + 1
            current_reward_memory = self.reward_memory[current_start : idx + 1]
            current_state_memory = self.next_state_memory[current_start : idx + 1]
            current_action_memory = self.action_memory[current_start : idx + 1]
        except:
            try:
                # used for the start of a new training episode (excluding the first)
                current_reward_memory = self.reward_memory[0 : idx + 1]
                current_state_memory = self.next_state_memory[0 : idx + 1]
                current_action_memory = self.action_memory[0 : idx + 1]
            except:
                # used for the very first training step of the first episode
                current_reward_memory = self.reward_memory[idx]
                current_state_memory = self.next_state_memory[idx]
                current_action_memory = self.action_memory[idx]

        #  log history for the very first training episode
        if T.all(self.terminal_memory != True) and done is not True:
            self.epis_idx = [idx + 1]
            self.epis_reward_memory = [current_reward_memory]
            self.epis_state_memory = [current_state_memory]
            self.epis_action_memory = [current_action_memory]

        # reset history once the very first training episode has concluded
        if (self.terminal_memory == True).sum() == 1 and done is True:
            self.epis_idx = [idx]
            self.epis_reward_memory = []
            self.epis_state_memory = []
            self.epis_action_memory = []

        # log the aggregated history upon termination of all subsequent training episode
        if done is True:
            self.epis_idx.append(idx + 1)
            self.epis_reward_memory.append(current_reward_memory)
            self.epis_state_memory.append(current_state_memory)
            self.epis_action_memory.append(current_action_memory)

    def store_exp(
        self,
        state: NDArrayFloat,
        action: NDArrayFloat,
        reward: float,
        next_state: NDArrayFloat,
        done: bool,
    ) -> None:
        """
        Store a transistion to the buffer containing a total up to a maximum size and
        log history of rewards, states, and actions for each episode.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state
            reward: reward from arriving at current environment state
            next_state: next or new environment state
            done: flag if new state is terminal
        """
        idx = self.mem_idx % self.mem_size

        self.state_memory[idx] = T.tensor(state, dtype=T.float)
        self.action_memory[idx] = T.tensor(action, dtype=T.float)
        self.reward_memory[idx] = T.tensor(max(reward, self.r_abs_zero), dtype=T.float)
        self.next_state_memory[idx] = T.tensor(next_state, dtype=T.float)
        self.terminal_memory[idx] = T.tensor(done, dtype=T.bool)

        if self.multi_steps > 1:
            self._episode_history(idx, done)

        self.mem_idx += 1

    def _construct_history(
        self, step: int, epis_history: T.FloatTensor
    ) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor]:
        """
        Given a single mini-batch sample (or step), obtain the history of rewards, and
        state-action pairs.

        Parameters:
            step: index or step of sample step
            epis_history: index list containing final steps of all training epsiodes

        Returns:
            rewards: sample reward history
            states: sample state history
            actions: sample action history
        """
        # find which episode the sampled step (experience) is located
        if step > epis_history[0]:
            sample_idx = T.where(step - epis_history > 0)[0]
            sample_idx = T.max(sample_idx + 1)
            n_rewards = step - epis_history[sample_idx - 1]
        else:
            sample_idx, n_rewards = 0, step

        # generate history of episode up till the sample step
        try:
            rewards = self.epis_reward_memory[sample_idx][0 : n_rewards + 1]
            states = self.epis_state_memory[sample_idx][0 : n_rewards + 1]
            actions = self.epis_action_memory[sample_idx][0 : n_rewards + 1]
        except:
            try:
                # used for the first training episode
                rewards = self.epis_reward_memory[0][0 : n_rewards + 1]
                states = self.epis_state_memory[0][0 : n_rewards + 1]
                actions = self.epis_action_memory[0][0 : n_rewards + 1]
            except:
                try:
                    # used for the very first training step of the first training episode
                    rewards = self.epis_reward_memory[0][0]
                    states = self.epis_state_memory[0][0]
                    actions = self.epis_action_memory[0][0]
                except:
                    rewards, states, actions = (
                        [T.zeros((1,), device=self.device)],
                        [T.zeros((self.input_dims,), device=self.device)],
                        [T.zeros((self.num_actions,), device=self.device)],
                    )

        return rewards, states, actions

    def _episode_rewards_states_actions(
        self, batch: List[int]
    ) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor]:
        """
        Collect respective histories for each sample in the mini-batch.

        Parameters:
            batch: list of mini-batch steps representing samples

        Returns:
            sample_epis_rewards: mini-batch reward history
            sample_epis_states: mini-batch state history
            sample_epis_actions: mini-batch actions history
        """
        epis_history = T.tensor(self.epis_idx, device=self.device)
        batch_histories = [
            self._construct_history(step, epis_history) for step in batch
        ]

        sample_epis_rewards = [x[0] for x in batch_histories]
        sample_epis_states = [x[1] for x in batch_histories]
        sample_epis_actions = [x[2] for x in batch_histories]

        return sample_epis_rewards, sample_epis_states, sample_epis_actions

    def _multi_step_rewards_states_actions(
        self,
        reward_history: T.FloatTensor,
        state_history: T.FloatTensor,
        action_history: T.FloatTensor,
        multi_length: int,
    ) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor]:
        """
        For a single mini-batch sample, generate multi-step rewards and identify
        initial state-action pair.

        Parameters:
            reward_history: entire reward history of sample
            state_history: entire state history of sample
            action_history: entire action history of sample
            multi_length: minimum of length of episode or multi-steps

        Returns:
            multi_reward: discounted sum of multi-step rewards
            initial_state: array of initial state before bootstrapping
            initial_action: array of initial action before bootstrapping
        """
        idx = int(multi_length)

        # the sampled step is treated as the (n-1)th step
        discounted_rewards = T.tensor(
            [self.gamma**t * reward_history[-idx + t] for t in range(idx - 1)],
            device=self.device,
        )

        if self.dyna == "A":
            multi_reward = T.sum(discounted_rewards)
        else:
            multi_reward = T.prod(discounted_rewards)

        initial_state = state_history[-idx]
        initial_action = action_history[-idx]

        return multi_reward, initial_state, initial_action

    def _multi_step_batch(
        self,
        step_rewards: List[T.FloatTensor],
        step_states: List[T.FloatTensor],
        step_actions: List[T.FloatTensor],
    ) -> Tuple[T.FloatTensor, T.FloatTensor, T.FloatTensor, T.FloatTensor]:
        """
        Collect respective multi-step returns and initial state-action pairs for each
        sample in the mini-batch.

        Parameters:
            step_rewards: complete reward history of entire mini-batch
            step_states: complete state history of entire mini-batch
            step_actions: complete action history of entire mini-batch

        Returns:
            batch_multi_reward: discounted sum of multi-step rewards
            batch_initial_state: array of initial state before bootstrapping
            batch_initial_action: array of initial action before bootstrapping
            batch_eff_length: effective n-step bootstrapping length
        """
        # effective length taken to be the minimum of either history length or multi-steps
        try:
            batch_eff_length = T.tensor(
                [x.shape[0] for x in step_rewards], device=self.device
            )
        except:
            batch_eff_length = T.tensor(
                [len(x) for x in step_rewards], device=self.device
            )

        batch_eff_length = T.minimum(batch_eff_length, self.multi_steps)

        batch_multi = [
            self._multi_step_rewards_states_actions(
                step_rewards[x], step_states[x], step_actions[x], batch_eff_length[x]
            )
            for x in range(self.batch_size)
        ]

        batch_multi_rewards = [batch_multi[x][0] for x in range(self.batch_size)]
        batch_states = [batch_multi[x][1] for x in range(self.batch_size)]
        batch_actions = [batch_multi[x][2] for x in range(self.batch_size)]

        return batch_multi_rewards, batch_states, batch_actions, batch_eff_length

    def sample_exp(
        self,
    ) -> Tuple[
        T.FloatTensor,
        T.FloatTensor,
        T.FloatTensor,
        T.FloatTensor,
        T.BoolTensor,
        T.FloatTensor,
    ]:
        """
        Uniformly sample a batch from replay buffer for agent learning.

        Returns:
            states: batch of environment states
            actions: batch of continuous actions taken to arrive at states
            rewards: batch of (discounted multi-step)  rewards from current states
            next_states: batch of next environment states
            dones (bool): batch of done flags
            batch_eff_length: batch of effective multi-step episode lengths
        """
        # pool batch from either partial or fully populated buffer
        max_mem = min(self.mem_idx, self.mem_size)
        batch = T.randperm(max_mem, device=self.device)[: self.batch_size]

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        eff_length = self.eff_length[0]

        if self.multi_steps > 1:
            (
                step_rewards,
                step_states,
                step_actions,
            ) = self._episode_rewards_states_actions(batch)
            rewards, states, actions, eff_length = self._multi_step_batch(
                step_rewards, step_states, step_actions
            )

            for step in range(self.batch_size):
                self.multi_rewards[step] = rewards[step]
                self.multi_states[step, :] = states[step]
                self.multi_actions[step, :] = actions[step]
            rewards, states, actions = (
                self.multi_rewards,
                self.multi_states,
                self.multi_actions,
            )

        return states, actions, rewards, next_states, dones, eff_length
