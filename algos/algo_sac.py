"""
title:                  algo_sac.py
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
    Responsible for executing the Soft Actor-Critic (SAC) algorithm.
"""

import sys

sys.path.append("./")

import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch as T

NDArrayFloat = npt.NDArray[np.float_]

import tests.test_live_learning as test_live_learning
import tools.critic_loss as critic
import tools.utils as utils
from algos.networks_sac import ActorNetwork, CriticNetwork
from tools.replay import ReplayBuffer
from tools.replay_torch import ReplayBufferTorch


class Agent_sac:
    """
    Causal entropy arguments based on https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf.
    SAC agent algorithm based on https://arxiv.org/pdf/1812.05905.pdf.
    Agent learning using PyTorch neural networks described in
    https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf.

    Methods:
        store_transistion(state, action, reward, next_state, done):
            Stores current step details into replay buffer.

        select_next_action(state):
            Select next agent action based on provided single state using given
            stochastic policy distribution.

        eval_next_action(state):
            Select next agent action for evaluation based on provided state using
            given mean of stochastic policy distribution.

        _mini_batch(batch_size):
            Randomly collects mini-batch from replay buffer and sends to GPU.

        _multi_step_target(batch_rewards, batch_next_states, batch_dones, batch_epis_rewards):
            Predicits next actions from next states in mini-batch, forward
            propagates these state-action pairs, and construct bootstrapped
            target soft Q-values.

        learn():
            Perform agent learning by sequentially minimising critic and policy
            loss functions every step. Updates Cauchy scale and CIM kernel size
            parameters.

        _update_critic_parameters(tau):
            Update critics target network parameters with Polyak averaging.

        save_models():
            Save actor-critic network parameters.

        load_models():
            Load actor-critic network parameters.
    """

    def __init__(self, inputs: dict) -> None:
        """
        Initialise actor-critic networks and experience replay buffer.

        Parameters:
            inputs: dictionary containing all execution details
        """
        self.input_dims = sum(inputs["input_dims"])  # input dimensions tuple
        self.num_actions = int(inputs["num_actions"])
        self.max_action = float(inputs["max_action"])
        self.min_action = float(inputs["min_action"])

        self.lr_kappa = inputs["sac_temp_learn_rate"]
        self.actor_update_interval = int(inputs["sac_actor_step_update"])
        self.temp_update_interval = int(inputs["sac_temp_step_update"])
        self.target_critic_update = int(inputs["sac_target_critic_update"])
        self.tau = inputs["sac_target_update_rate"]
        self.reward_scale = inputs["reward_scale"]
        self.stoch = str(inputs["s_dist"])

        self.optimise_count = int(inputs["batch_size"][inputs["algo"]])
        self.actor_percentile = inputs["actor_percentile"] / 100
        self.batch_size = inputs["mini_batch_size"]

        self.buffer_torch = inputs["buffer_gpu"]
        if self.buffer_torch == False:
            self.memory = ReplayBuffer(inputs)
        else:
            self.memory = ReplayBufferTorch(inputs)

        self.gamma = inputs["discount"]
        self.multi_steps = int(inputs["multi_steps"])
        self.cauchy_scale_1 = inputs["cauchy_scale"]
        self.cauchy_scale_2 = inputs["cauchy_scale"]

        self.loss_type = str(inputs["loss_fn"])

        self.time_step = 0
        self.learn_step_cntr = 0

        # directory to save network checkpoints
        match inputs["dynamics"]:
            case "A":
                dyna = "additive/"
            case "M":
                dyna = "multiplicative/"
            case "MKT":
                dyna = "market/"
            case "GUD":
                dyna = "guidance/"

        if inputs["test_agent"]:
            dyna = "test_" + dyna

        dir = "./results/" + dyna + "models/" + str(inputs["env_id"])

        if not os.path.exists(dir):
            os.makedirs(dir)

        # generic directory and name of PyTorch model
        model_name = utils.save_directory(inputs, results=False)

        self.actor = ActorNetwork(inputs, model_name, target=0)
        self.target_actor = ActorNetwork(inputs, model_name, target=1)
        self.critic_1 = CriticNetwork(inputs, model_name, critic=1, target=0)
        self.target_critic_1 = CriticNetwork(inputs, model_name, critic=1, target=1)
        self.critic_2 = CriticNetwork(inputs, model_name, critic=2, target=0)
        self.target_critic_2 = CriticNetwork(inputs, model_name, critic=2, target=1)

        self.critic_mean = str(inputs["critic_mean_type"])

        # initialisation for tail exponent estimation
        self.log_noise = T.tensor(
            float(inputs["log_noise"]), device=self.critic_1.device
        )
        self.zipf_x = (T.ones((self.optimise_count,)) + self.optimise_count).view(-1)
        for x in range(self.optimise_count):
            self.zipf_x[x] = self.zipf_x[x] / (x + 1)
        self.zipf_x = T.log(self.zipf_x)
        self.zipf_x = (self.zipf_x - T.mean(self.zipf_x)).to(self.critic_1.device)
        self.zipf_x2 = T.sum(self.zipf_x**2).to(self.critic_1.device)

        # learn temperature via convex optimisation (dual gradient descent approximation)
        self.device = T.device(inputs["gpu"] if T.cuda.is_available() else "cpu")
        self.log_alpha = T.tensor(
            float(inputs["initial_logtemp"]), requires_grad=True, device=self.device
        )
        self.temp_optimiser = T.optim.Adam([self.log_alpha], lr=self.lr_kappa)
        self.entropy_target = -int(self.num_actions)  # heuristic assumption

    def store_transistion(
        self,
        state: NDArrayFloat,
        action: NDArrayFloat,
        reward: float,
        next_state: NDArrayFloat,
        done: bool,
    ) -> None:
        """
        Store a transistion to the buffer containing a total up to max_size.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state
            reward: reward from current environment state
            next_state: next environment state
            done: flag if current state is terminal
        """
        self.memory.store_exp(state, action, reward, next_state, done)

    def select_next_action(self, state: T.FloatTensor) -> NDArrayFloat:
        """
        Agent selects next action from stochastic policy.

        Parameters:
            state: current environment state

        Return:
            next_action: action to be taken by agent during training
        """
        # make states a list for stochastic sampling and then select action
        current_state = T.tensor(
            np.array([state]), dtype=T.float, device=self.actor.device
        )

        match self.stoch:
            case "MVN":
                next_action, _ = self.actor.stochastic_mv_gaussian(current_state)
                return next_action.detach().cpu().numpy()[0]

            case "N":
                next_action, _ = self.actor.stochastic_uv_gaussian(current_state)
                return next_action.detach().cpu().numpy()[0]

            case "L":
                next_action, _ = self.actor.stochastic_uv_laplace(current_state)
                return next_action.detach().cpu().numpy()[0]

    def eval_next_action(self, state: T.FloatTensor) -> NDArrayFloat:
        """
        Agent selects next action from determinstic policy for direct agent
        inference/evaluation.

        Parameters:
            state: current environment state

        Return:
            next_action: action to be taken by agent during evaluation
        """
        current_state = T.tensor(
            np.array([state]), dtype=T.float, device=self.actor.device
        )
        next_action = self.actor.deterministic_policy(current_state)

        return next_action.detach().cpu().numpy()[0]

    def _mini_batch(
        self,
    ) -> Tuple[
        T.FloatTensor,
        T.FloatTensor,
        T.FloatTensor,
        T.FloatTensor,
        T.BoolTensor,
        T.IntTensor,
    ]:
        """
        Uniform sampling from replay buffer and send to GPU.

        Returns:
            states: batch of environment states
            actions: batch of continuous actions taken to arrive at states
            rewards: batch of (discounted multi-step) rewards from current states
            next_states: batch of next environment states
            dones (bool): batch of done flags
            eff_length: batch of effective multi-step episode lengths
        """
        if self.memory.mem_idx < self.batch_size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_eff_length,
        ) = self.memory.sample_exp()

        if self.buffer_torch == False:
            batch_states = T.tensor(
                batch_states, dtype=T.float, device=self.critic_1.device
            )
            batch_actions = T.tensor(
                batch_actions, dtype=T.float, device=self.critic_1.device
            )
            batch_rewards = T.tensor(
                batch_rewards, dtype=T.float, device=self.critic_1.device
            )
            batch_next_states = T.tensor(
                batch_next_states, dtype=T.float, device=self.critic_1.device
            )
            batch_dones = T.tensor(
                batch_dones, dtype=T.bool, device=self.critic_1.device
            )
            batch_eff_length = T.tensor(
                batch_eff_length, dtype=T.int, device=self.critic_1.device
            )

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_eff_length,
        )

    def _multi_step_target(
        self,
        batch_rewards: T.FloatTensor,
        batch_next_states: T.FloatTensor,
        batch_dones: T.BoolTensor,
        batch_eff_length: T.IntTensor,
    ) -> T.FloatTensor:
        """
        Multi-step target soft Q-values for mini-batch.

        Parameters:
            batch_rewards: batch of (discounted multi-step) rewards from current states
            batch_next_states: batch of next environment states
            batch_dones: batch of done flags
            batch_eff_length: batch of effective multi-step episode lengths

        Returns:
            batch_target: clipped double multi-step target Q-values (advantage)
            soft_q_target: clipped double multi-step target Q-values
        """
        # ensure memory buffer large enough for mini-batch
        if self.memory.mem_idx <= self.batch_size:
            return np.nan

        # sample next stochastic action policy for target critic network based on mini-batch
        match self.stoch:
            case "MVN":
                (
                    batch_next_stoc_actions,
                    batch_next_logprob_actions,
                ) = self.actor.stochastic_mv_gaussian(batch_next_states)

            case "N":
                (
                    batch_next_stoc_actions,
                    batch_next_logprob_actions,
                ) = self.actor.stochastic_uv_gaussian(batch_next_states)

            case "L":
                (
                    batch_next_stoc_actions,
                    batch_next_logprob_actions,
                ) = self.actor.stochastic_uv_laplace(batch_next_states)

        batch_next_logprob_actions = batch_next_logprob_actions.view(-1)

        # obtain twin next target soft Q-values for mini-batch and check terminal status
        q1_target = self.target_critic_1.forward(
            batch_next_states, batch_next_stoc_actions
        ).view(-1)
        q2_target = self.target_critic_2.forward(
            batch_next_states, batch_next_stoc_actions
        ).view(-1)

        q1_target[batch_dones], q2_target[batch_dones] = 0.0, 0.0

        # clipped double target critic with bootstrapping
        soft_q_target = T.min(q1_target, q2_target)
        soft_q_target = (
            self.reward_scale * batch_rewards
            + self.gamma**batch_eff_length * soft_q_target
        )

        # soft value as an advantage function
        soft_value = soft_q_target - self.log_alpha.exp() * batch_next_logprob_actions
        batch_target = soft_value.view(self.batch_size, 1)

        return batch_target, soft_q_target

    def learn(self) -> Tuple[List[NDArrayFloat], NDArrayFloat, List[NDArrayFloat]]:
        """
        Agent learning via SAC algorithm with multi-step bootstrapping and robust
        critic loss.

        Returns:
            loss: empirical mean / min / max / shadow mean, tail exponents, mean actor loss
            logtemp: log entropy adjustment factor (temperature)
            loss_params: list of Cauchy scale parameters and CIM kernel sizes for twin critics
        """
        # return nothing till batch size less than replay buffer
        if self.memory.mem_idx <= self.batch_size:
            loss = [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
            cpu_logtmep = self.log_alpha.detach().cpu().numpy()
            loss_params = [np.nan, np.nan, np.nan, np.nan]
            return loss, cpu_logtmep, loss_params

        # uniformly sample from replay buffer (off-policy)
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
            batch_eff_length,
        ) = self._mini_batch()

        # estimate target critic value
        batch_target, soft_q = self._multi_step_target(
            batch_rewards, batch_next_states, batch_dones, batch_eff_length
        )

        # obtain twin current state soft Q-values for mini-batch
        q1 = self.critic_1.forward(batch_states, batch_actions).view(-1)
        q2 = self.critic_2.forward(batch_states, batch_actions).view(-1)
        q1, q2 = q1.view(self.batch_size, 1), q2.view(self.batch_size, 1)

        # updates CIM kernel size empirically
        kernel_1 = critic.cim_size(q1, batch_target).cpu().numpy()
        kernel_2 = critic.cim_size(q2, batch_target).cpu().numpy()

        # backpropogation of critic loss
        self.critic_1.optimiser.zero_grad(set_to_none=True)
        self.critic_2.optimiser.zero_grad(set_to_none=True)

        (
            q1_mean,
            q1_min,
            q1_max,
            q1_shadow,
            q1_alpha,
            q2_mean,
            q2_min,
            q2_max,
            q2_shadow,
            q2_alpha,
        ) = critic.loss_function(
            q1,
            self.cauchy_scale_1,
            kernel_1,
            q2,
            self.cauchy_scale_2,
            kernel_2,
            batch_target,
            self.batch_size,
            self.optimise_count,
            self.log_noise,
            self.zipf_x,
            self.zipf_x2,
            self.loss_type,
        )

        # ensure consistent mean selection for learning (placeholder feature)
        if self.critic_mean == "E":
            q1_loss, q2_loss = q1_mean, q2_mean
        else:
            q1_loss, q2_loss = q1_shadow, q2_shadow

        critic_loss = 0.5 * (q1_loss + q2_loss)  # 0.5 multiplier due to SAC convention

        # retain entire graph due to coupling of soft value and policy
        critic_loss.backward(retain_graph=True)

        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        # updates Cauchy scale parameter using the Nagy algorithm
        self.cauchy_scale_1 = (
            critic.nagy_algo(q1, batch_target, self.cauchy_scale_1).cpu().numpy()
        )
        self.cauchy_scale_2 = (
            critic.nagy_algo(q2, batch_target, self.cauchy_scale_2).cpu().numpy()
        )

        self.learn_step_cntr += 1

        test_live_learning.sac_critic_stability(
            self.learn_step_cntr, q1, q2, soft_q, batch_target
        )

        if self.learn_step_cntr % self.target_critic_update == 0:
            self._update_critic_parameters()

        cpu_q1_mean, cpu_q2_mean = (
            q1_mean.detach().cpu().numpy(),
            q2_mean.detach().cpu().numpy(),
        )
        cpu_q1_min, cpu_q2_min = (
            q1_min.detach().cpu().numpy(),
            q2_min.detach().cpu().numpy(),
        )
        cpu_q1_max, cpu_q2_max = (
            q1_max.detach().cpu().numpy(),
            q2_max.detach().cpu().numpy(),
        )
        cpu_q1_alpha, cpu_q2_alpha = (
            q1_alpha.detach().cpu().numpy(),
            q2_alpha.detach().cpu().numpy(),
        )
        cpu_logtmep = self.log_alpha.detach().cpu().numpy()

        loss = [
            cpu_q1_mean,
            cpu_q2_mean,
            cpu_q1_min,
            cpu_q2_min,
            cpu_q1_max,
            cpu_q2_max,
            np.nan,
            np.nan,
            cpu_q1_alpha,
            cpu_q2_alpha,
            np.nan,
        ]
        loss_params = [self.cauchy_scale_1, self.cauchy_scale_2, kernel_1, kernel_2]

        # update actor, temperature, and target critic networks every interval
        if self.learn_step_cntr % self.actor_update_interval != 0:
            # remove unneeded graph components using python garbage collection
            q1_loss, q2_loss, critic_loss = None, None, None
            return loss, cpu_logtmep, loss_params

        # sample current stochastic action policy for critic network based on mini-batch
        match self.stoch:
            case "MVN":
                (
                    batch_stoc_actions,
                    batch_logprob_actions,
                ) = self.actor.stochastic_mv_gaussian(batch_states)

            case "N":
                (
                    batch_stoc_actions,
                    batch_logprob_actions,
                ) = self.actor.stochastic_uv_gaussian(batch_states)

            case "L":
                (
                    batch_stoc_actions,
                    batch_logprob_actions,
                ) = self.actor.stochastic_uv_laplace(batch_states)

        batch_logprob_actions = batch_logprob_actions.view(-1)

        # obtain twin current soft-Q values for mini-batch
        q1 = self.critic_1.forward(batch_states, batch_stoc_actions)
        q2 = self.critic_2.forward(batch_states, batch_stoc_actions)
        soft_q = T.min(q1, q2).view(-1)

        # learn stochastic actor policy by minimising KL divergence (advantage function)
        self.actor.optimiser.zero_grad(set_to_none=True)
        actor_loss = (soft_q - self.log_alpha.exp() * batch_logprob_actions).view(-1)

        # fractional Kelly i.e. emphasis placed on improving worst performers
        if self.actor_percentile != 1:
            actor_loss = actor_loss.sort(descending=True)[0]
            actor_loss = actor_loss[: self.optimise_count]

        actor_loss = -T.mean(actor_loss)
        actor_loss.backward()

        self.actor.optimiser.step()

        cpu_actor_loss = actor_loss.detach().cpu().numpy()
        loss[-1] = cpu_actor_loss

        if self.learn_step_cntr % self.temp_update_interval != 0:
            # remove unneeded graph components using python garbage collection
            q1_loss, q2_loss, critic_loss = None, None, None
            batch_logprob_actions, q1, q2, soft_q, actor_loss = (
                None,
                None,
                None,
                None,
                None,
            )
            return loss, cpu_logtmep, loss_params

        # learn temperature by approximating dual gradient descent
        self.temp_optimiser.zero_grad()
        temp_loss = -(
            self.log_alpha.exp()
            * (batch_logprob_actions.detach() + self.entropy_target)
        )
        temp_loss = T.mean(temp_loss)
        temp_loss.backward()
        self.temp_optimiser.step()

        cpu_logtmep = self.log_alpha.detach().cpu().numpy()

        # remove unneeded graph components using python garbage collection
        q1_loss, q2_loss, critic_loss = None, None, None
        batch_logprob_actions, q1, q2, soft_q, actor_loss = None, None, None, None, None

        return loss, cpu_logtmep, loss_params

    def _update_critic_parameters(self) -> None:
        """
        Update target critic deep network parameters with Polyak averaging
        rate smoothing.
        """
        for param_1, target_param_1, param_2, target_param_2 in zip(
            self.critic_1.parameters(),
            self.target_critic_1.parameters(),
            self.critic_2.parameters(),
            self.target_critic_2.parameters(),
        ):

            target_param_1.data.copy_(
                self.tau * param_1.data + (1 - self.tau) * target_param_1.data
            )

            target_param_2.data.copy_(
                self.tau * param_2.data + (1 - self.tau) * target_param_2.data
            )

    def save_models(self) -> None:
        """
        Saves all 3 networks.
        """
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self) -> None:
        """
        Loads all 3 networks.
        """
        print("Loading network checkpoints")
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
