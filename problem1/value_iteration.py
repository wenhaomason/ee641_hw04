"""
Value Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class ValueIteration:
    """
    Value Iteration solver for gridworld MDP.

    Computes optimal value function V* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Value Iteration solver.

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
        Run value iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Converged value function V(s)
            n_iterations: Number of iterations until convergence
        """
        # TODO: Initialize value function to zeros
        # TODO: Iterate until convergence:
        #       - For each state:
        #           - Compute Q(s,a) for all actions using Bellman backup
        #           - Set V(s) = max_a Q(s,a)
        #       - Check convergence: max|V_new - V_old| < epsilon
        #       - Update value function
        # TODO: Return final values and iteration count

        raise NotImplementedError

    def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.

        Args:
            state: State index
            values: Current value function

        Returns:
            q_values: Array of Q(s,a) for each action
        """
        # TODO: For each action:
        #       - Get transition probabilities P(s'|s,a)
        #       - Compute expected value:
        #           Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
        # TODO: Return Q-values array

        raise NotImplementedError

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.

        Args:
            values: Optimal value function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Compute Q-values for all actions
        #       - Select action with maximum Q-value
        # TODO: Return policy array

        raise NotImplementedError

    def bellman_backup(self, state: int, values: np.ndarray) -> float:
        """
        Perform Bellman backup for a single state.

        Args:
            state: State index
            values: Current value function

        Returns:
            Updated value for state
        """
        # TODO: If terminal state, return 0
        # TODO: Compute Q-values for all actions
        # TODO: Return maximum Q-value

        raise NotImplementedError

    def compute_bellman_error(self, values: np.ndarray) -> float:
        """
        Compute Bellman error for current value function.

        Bellman error = max_s |V(s) - max_a Q(s,a)|

        Args:
            values: Current value function

        Returns:
            Maximum Bellman error across all states
        """
        # TODO: For each state:
        #       - Compute optimal value using Bellman backup
        #       - Calculate absolute difference from current value
        # TODO: Return maximum error

        raise NotImplementedError