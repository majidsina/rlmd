"""
title:                  networks_sac.py
python version:         3.10
torch verison:          1.11

code style:             black==22.3
import style:           isort==5.10

copyright (C):          https://www.gnu.org/licenses/agpl-3.0.en.html
author:                 J. S. Grewal (2022)
email:                  <rg (_] public [at} proton {dot) me>
linkedin:               https://www.linkedin.com/in/rajagrewal
website:                https://github.com/rajabinks

Description:
    Responsible for generating actor-critic deep (linear) neural networks for the
    Soft Actor-Critic (SAC) algorithm.
"""

import os
from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.laplace import Laplace
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class ActorNetwork(nn.Module):
    """
    Actor network for single GPU.

    Methods:
        forward(state):
            Forward propogate state to obtain policy distribution moments for
            each action component.

        stochastic_uv_gaussian(state):
            Obtain tanh bounded actions values and log probabilites for a state
            using stochastic Gaussian policy.

        stochastic_uv_laplace(state):
            Obtain tanh bounded actions values and log probabilites for a state
            using stochastic Laplace policy.

        stochasic_mv_gaussian(state):
            Obtain tanh bounded actions values and log probabilites for a state
            using multi-variable spherical Gaussian distribution with each state
            having a unique covariance matrix.

        deterministic_policy(state):
            Obtain tanh bounded deterministic actions for inference.

        save_checkpoint():
            Saves network parameters.

        load_checkpoint():
            Loads network parameters.
    """

    def __init__(self, inputs: dict, model_name: str, target: bool) -> None:
        """
        Initialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs: dictionary containing all execution details
            target: whether constructing target network (1) or not (0)
        """
        super(ActorNetwork, self).__init__()
        self.input_dims = sum(inputs["input_dims"])
        self.num_actions = int(inputs["num_actions"])
        self.max_action = float(inputs["max_action"])

        nn_name = "actor" if target == 0 else "actor_target"

        fc1_dim = int(inputs["sac_layer_1_units"])
        fc2_dim = int(inputs["sac_layer_2_units"])
        lr_alpha = inputs["sac_actor_learn_rate"]
        self.stoch = str(inputs["s_dist"])
        self.log_scale_min = float(inputs["log_scale_min"])
        self.log_scale_max = float(inputs["log_scale_max"])
        self.reparam_noise = inputs["reparam_noise"]

        file = model_name + "_" + nn_name + ".pt"
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features
        self.fc1 = T.jit.script(nn.Linear(self.input_dims, fc1_dim))
        self.fc2 = T.jit.script(nn.Linear(fc1_dim, fc2_dim))
        self.pi = T.jit.script(nn.Linear(fc2_dim, self.num_actions))
        self.log_scale = T.jit.script(nn.Linear(fc2_dim, self.num_actions))

        self.optimiser = optim.Adam(self.parameters(), lr=lr_alpha)
        self.device = T.device(inputs["gpu"] if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state: T.FloatTensor) -> Tuple[T.FloatTensor, T.FloatTensor]:
        """
        Forward propogation of state to obtain fixed Gaussian distribution parameters
        (moments) for each possible action component.

        Clamping of log scales is verfired in
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py,
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py,
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py, and
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        so that divergence is avoided when sampling.

        Parameters:
            state: current environment state

        Returns:
            mu: deterministic loc of action components
            scale: exponentiated log scales of loc means of action components
        """
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)

        mu, log_scale = self.pi(actions), self.log_scale(actions)

        log_scale = T.clamp(log_scale, self.log_scale_min, self.log_scale_max)

        scale = log_scale.exp()

        # important to enhance learning stability with very smooth critic loss functions
        if T.any(T.isnan(mu)) or T.any(T.isnan(scale)):
            mu = T.nan_to_num(mu, nan=0, posinf=0, neginf=0)
            scale = T.nan_to_num(scale, nan=3, posinf=3, neginf=3)

        return mu, scale

    def stochastic_uv_gaussian(
        self, state: T.FloatTensor
    ) -> Tuple[T.FloatTensor, T.FloatTensor]:
        """
        Stochastic action selection sampled from unbounded univarite Gaussian
        distirbution with tanh bounding using a Jacobian transformation and the
        reparameterisation trick from https://arxiv.org/pdf/1312.6114.pdf.

        Addition of constant reparameterisation noise to the logarithm is crucial, as verified in
        https://github.com/haarnoja/sac/blob/master/sac/policies/gaussian_policy.py,
        https://github.com/rail-berkeley/softlearning/blob/master/softlearning/policies/gaussian_policy.py,
        https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/policies/gaussian_mlp_policy.py,
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py, and
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        where orders of magnitude smaller than 1e-6 prevent learning from occuring.

        Parameters:
            state: current environment state or mini-batch

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
            bounded_logprob_action: log probability (log likelihood) of sampled truncated action
        """
        mu, scale = self.forward(state)

        action_distribution = Normal(loc=mu, scale=scale)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = action_distribution.rsample()
        unbounded_logprob_action = action_distribution.log_prob(unbounded_action).sum(
            dim=1, keepdim=True
        )
        bounded_action = T.tanh(unbounded_action) * self.max_action

        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(
            1 - (bounded_action / self.max_action) ** 2 + self.reparam_noise
        ).sum(dim=1, keepdim=True)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action

    def stochastic_uv_laplace(
        self, state: T.FloatTensor
    ) -> Tuple[T.FloatTensor, T.FloatTensor]:
        """
        Stochastic action selection sampled from unbounded univarite Laplace (double
        exponential) distirbution with tanh bounding using a Jacobian transformation
        and the reparameterisation trick from https://arxiv.org/pdf/1312.6114.pdf.

        Addition of constant reparameterisation noise to the logarithm is crucial, as verified in
        https://github.com/haarnoja/sac/blob/master/sac/policies/gaussian_policy.py,
        https://github.com/rail-berkeley/softlearning/blob/master/softlearning/policies/gaussian_policy.py,
        https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/policies/gaussian_mlp_policy.py,
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py, and
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        where orders of magnitude smaller than 1e-6 prevent learning from occuring.

        Parameters:
            state: current environment state or mini-batch

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
            bounded_logprob_action: log probability (log likelihood) of sampled truncated action
        """
        mu, scale = self.forward(state)

        action_distribution = Laplace(loc=mu, scale=scale)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = action_distribution.rsample()
        unbounded_logprob_action = action_distribution.log_prob(unbounded_action).sum(
            dim=1, keepdim=True
        )
        bounded_action = T.tanh(unbounded_action) * self.max_action

        # ensure defined bounded log by adding minute noise
        log_inv_jacobian = T.log(
            1 - (bounded_action / self.max_action) ** 2 + self.reparam_noise
        ).sum(dim=1, keepdim=True)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action

    def stochastic_mv_gaussian(
        self, state: T.FloatTensor
    ) -> Tuple[T.FloatTensor, T.FloatTensor]:
        """
        Stochastic action selection sampled from unbounded spherical Gaussian input
        noise with tanh bounding using Jacobian transformation and the reparameterisation
        trick from https://arxiv.org/pdf/1312.6114.pdf. Allows each mini-batch state
        to have a unique covariance matrix allowing faster learning in terms of
        cumulative steps but longer run time per step.

        Addition of constant reparameterisation noise to the logarithm is crucial, as verified in
        https://github.com/haarnoja/sac/blob/master/sac/policies/gaussian_policy.py,
        https://github.com/rail-berkeley/softlearning/blob/master/softlearning/policies/gaussian_policy.py,
        https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/policies/gaussian_mlp_policy.py,
        https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/blob/master/SAC/networks.py, and
        https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py
        where orders of magnitude smaller than 1e-6 prevent learning from occuring.

        Parameters:
            state: current environment state or mini-batch

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
            bounded_logprob_action: log probability of sampled truncated action
        """
        mu, var = self.forward(state)

        # create diagonal covariance matrices for each sample and perform Cholesky decomposition
        cov_mat = T.diag_embed(var)
        chol_ltm = T.linalg.cholesky(cov_mat, upper=False)

        action_distribution = MultivariateNormal(loc=mu, scale_tril=chol_ltm)

        # reparmeterise trick for random variable sample to be pathwise differentiable
        unbounded_action = action_distribution.rsample()
        unbounded_logprob_action = action_distribution.log_prob(unbounded_action)
        bounded_action = T.tanh(unbounded_action) * self.max_action

        # ensure logarithm is defined in the vicinity of zero by adding minute noise
        log_inv_jacobian = T.log(
            1 - (bounded_action / self.max_action) ** 2 + self.reparam_noise
        ).sum(dim=1)
        bounded_logprob_action = unbounded_logprob_action - log_inv_jacobian

        return bounded_action, bounded_logprob_action

    def deterministic_policy(self, state: T.FloatTensor) -> T.FloatTensor:
        """
        Deterministic action selection for agent evaluation/inference.

        Parameters:
            state: current environment state

        Returns:
            bounded_action: action truncated by tanh and scaled by max action
        """
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)

        mu = self.pi(actions)

        return T.tanh(mu) * self.max_action

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        print("Loading actor checkpoint")
        self.load_state_dict(T.load(self.file_checkpoint))


class CriticNetwork(nn.Module):
    """
    Critic network for single GPU.

    Methods:
        forward(state):
            Forward propogate concatenated state and action to obtain soft Q-values.

        save_checkpoint():
            Saves network parameters.

        load_checkpoint():
            Loads network parameters.
    """

    def __init__(
        self, inputs: dict, model_name: str, critic: int, target: bool
    ) -> None:
        """
        Initialise class varaibles by creating neural network with Adam optimiser.

        Parameters:
            inputs: dictionary containing all execution details
            critic: number assigned to critic
            target: whether constructing target network (1) or not (0)
        """
        super(CriticNetwork, self).__init__()
        self.input_dims = sum(inputs["input_dims"])
        self.num_actions = int(inputs["num_actions"])
        self.max_action = float(inputs["max_action"])

        nn_name = "critic" if target == 0 else "target_critic"
        nn_name += "_" + str(critic)

        fc1_dim = int(inputs["sac_layer_1_units"])
        fc2_dim = int(inputs["sac_layer_2_units"])
        lr_beta = inputs["sac_critic_learn_rate"]

        file = model_name + "_" + nn_name + ".pt"
        self.file_checkpoint = os.path.join(file)

        # network inputs environment state space features and number of actions
        self.fc1 = T.jit.script(nn.Linear(self.input_dims + self.num_actions, fc1_dim))
        self.fc2 = T.jit.script(nn.Linear(fc1_dim, fc2_dim))
        self.q_value = T.jit.script(nn.Linear(fc2_dim, 1))

        self.optimiser = optim.Adam(self.parameters(), lr=lr_beta)
        self.device = T.device(inputs["gpu"] if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state: T.FloatTensor, action: T.FloatTensor) -> T.FloatTensor:
        """
        Forward propogation of state-action pair to obtain soft Q-value.

        Parameters:
            state: current environment state
            action: continuous actions taken to arrive at current state

        Returns:
            soft_Q: estimated soft Q action-value
        """
        Q_action_value = self.fc1(T.cat([state, action], dim=1))
        Q_action_value = F.relu(Q_action_value)
        Q_action_value = self.fc2(Q_action_value)
        Q_action_value = F.relu(Q_action_value)

        return self.q_value(Q_action_value)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_checkpoint)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_checkpoint))
