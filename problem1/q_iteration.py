"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize Q-function to zeros (shape: [n_states, n_actions])
        # TODO: Iterate until convergence:
        #       - For each state-action pair:
        #           - Compute updated Q-value using Bellman equation:
        #             Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
        #       - Check convergence: max|Q_new - Q_old| < epsilon
        #       - Update Q-function
        # TODO: Return final Q-values and iteration count

        raise NotImplementedError

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair.

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function

        Returns:
            Updated Q-value for (s,a)
        """
        # TODO: Get transition probabilities P(s'|s,a)
        # TODO: For each possible next state:
        #       - Get reward R(s,a,s')
        #       - Get max Q-value for next state: max_a' Q(s',a')
        #       - Accumulate: prob * [reward + gamma * max_q_next]
        # TODO: Return updated Q-value

        raise NotImplementedError

    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function.

        Args:
            q_values: Optimal Q-function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Select action with maximum Q-value: argmax_a Q(s,a)
        # TODO: Return policy array

        raise NotImplementedError

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function.

        Args:
            q_values: Q-function

        Returns:
            values: State value function V(s) = max_a Q(s,a)
        """
        # TODO: For each state:
        #       - Compute V(s) = max_a Q(s,a)
        # TODO: Return value function

        raise NotImplementedError

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        # TODO: For each state-action pair:
        #       - Compute updated Q-value using Bellman update
        #       - Calculate absolute difference from current Q-value
        # TODO: Return maximum error

        raise NotImplementedError