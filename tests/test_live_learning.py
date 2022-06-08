"""
title:                  test_live_learning.py
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
    Responsible for conducting tests for agent critic network while live learning
    for both algorithms.
"""

from typing import List

import numpy as np
import numpy.typing as npt
import torch as T

NDArrayFloat = npt.NDArray[np.float_]


def sac_critic_stability(
    step: int,
    q1: T.FloatTensor,
    q2: T.FloatTensor,
    q_soft: T.FloatTensor,
    q_target: T.FloatTensor,
) -> None:
    """
    Check whether SAC critic losses contain internal backpropagation errors and print
    mini-batch components.

    Parameters:
        step: learning step
        q1, q2: critic losses 1 and 2
        q_target, q_soft: target critic losses
    """
    combine = T.concat([q1, q2, q_target])

    if T.any(T.isnan(combine) == True):

        q1, q2, q_soft, q_target = (
            q1.view(-1),
            q2.view(-1),
            q_soft.view(-1),
            q_target.view(-1),
        )

        print(
            """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within SAC critic losses.

            Learning Step: {}

            Critic Loss 1:
            {}

            Criict Loss 2:
            {}

            Critic Soft Target Loss:
            {}

            Critic Target Loss:
            {}
            """.format(
                step, q1, q2, q_soft, q_target
            )
        )


def td3_critic_stability(
    step: int, q1: T.FloatTensor, q2: T.FloatTensor, q_target: T.FloatTensor
) -> None:
    """
    Check whether SAC critic losses contain internal backpropagation errors and print
    mini-batch components.

    Parameters:
        step: learning step
        q1, q2: critic losses 1 and 2
        q_target, q_soft: target critic losses
    """
    combine = T.concat([q1, q2, q_target])

    if T.any(T.isnan(combine) == True):

        q1, q2, q_target = q1.view(-1), q2.view(-1), q_target.view(-1)

        print(
            """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within TD3 critic losses.

            Learning Step: {}

            Critic Loss 1:
            {}

            Criict Loss 2:
            {}

            Critic Target Loss:
            {}
            """.format(
                step, q1, q2, q_target
            )
        )


def critic_learning(
    cum_step: int,
    batch_size: int,
    episode: int,
    step: int,
    loss: List[float],
    loss_params: List[float],
    logtemp: NDArrayFloat,
    state: NDArrayFloat,
    action: NDArrayFloat,
    reward: float,
    next_state: NDArrayFloat,
    done: bool,
    learn_done: bool = None,
    risk: NDArrayFloat = None,
) -> None:
    """
    Check whether critic losses contain internal backpropagation errors causing the
    entire script to terminate as learning ceases to occur.

    Parameters:
        cum_step: current amount of cumulative steps
        batch_size: mini-batch size
        episode: current episode number
        step: current step in episode
        loss: loss values of critic 1, critic 2 and actor
        loss_params: values of Cauchy scale parameters and kernel sizes for critics
        logtemp: log entropy adjustment factor (temperature)
        state: initial state
        action: array of actions to be taken determined by actor network
        reward: agent signal to maximise
        next_state: state arrived at from taking action
        done: Boolean flag for episode termination
        learn_done: Boolean flag for whether genuine termination
        risk: collection of additional data retrieved
    """
    if cum_step > batch_size:

        critic = np.array(loss[0:6] + loss[8:10], dtype=np.float32).flatten()

        if np.any(np.isnan(critic) == True):

            print(
                """
            --------------------------------------------------------------------------------------
            Script terminated due to the presence of NaN's within critic losses
            indicating failed agent neural network backpropagation. This issue is
            likely due to several reasons either individual or combined.

            For additive environments, mini-batch losses might be excessively supressed
            by highly smoothing loss functions.

            For multiplicative/market environments it may be due to several reasons
            such as the previous cause, state components diverge due to the possibility
            of unbounded environments, and/or other mysterious events.

            Cumulative Step: {}
            Episode: {}
            Step: {}

            Critic Loss 1:
            Mean: {}
            Min: {}
            Max: {}
            Tail: {}
            Scale: {}
            Kernel: {}

            Critic Loss 2:
            Mean: {}
            Min: {}
            Max: {}
            Tail: {}
            Scale: {}
            Kernel: {}

            Actor Loss Mean:
            {}

            Log Entropy Temperature (only SAC):
            {}

            State:
            {}

            Action:
            {}

            Reward:
            {}

            Next State:
            {}

            Done:
            {}
            """.format(
                    cum_step,
                    episode,
                    step,
                    loss[0],
                    loss[2],
                    loss[4],
                    loss[8],
                    loss_params[0],
                    loss_params[2],
                    loss[1],
                    loss[3],
                    loss[5],
                    loss[9],
                    loss_params[1],
                    loss_params[3],
                    logtemp,
                    loss[-1],
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
            )

            if learn_done != None:
                print(
                    """
                Learn Done:
                {}

                Risk:
                {}
                """.format(
                        learn_done, risk
                    )
                )

            # terminate script as agent learning is compromised
            exit()
