"""
Q-Iteration algorithm for solving MDPs.
"""

from typing import Tuple

import numpy as np
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
        q_values = np.zeros((self.n_states, self.n_actions), dtype=float)
        n_iter = 0
        # TODO: Iterate until convergence:
        for it in range(max_iterations):
            n_iter = it + 1
            new_q = np.copy(q_values)
            # For each state-action pair:
            for s in range(self.n_states):
                # If terminal, keep Q(s, a) = 0 for all actions
                if self.env.is_terminal(s):
                    new_q[s, :] = 0.0
                    continue
                for a in range(self.n_actions):
                    # Compute updated Q-value using Bellman equation:
                    new_q[s, a] = self.bellman_update(s, a, q_values)
            delta = float(np.max(np.abs(new_q - q_values)))
            q_values = new_q
            if delta < self.epsilon:
                break
        # TODO: Return final Q-values and iteration count
        return q_values, n_iter

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
        if self.env.is_terminal(state):
            return 0.0
        trans = self.env.get_transition_prob(state, action)
        total = 0.0
        # TODO: For each possible next state:
        for s_next, p in trans.items():
            #       - Get reward R(s,a,s')
            r = self.env.get_reward(state, action, s_next)
            #       - Get max Q-value for next state: max_a' Q(s',a')
            max_q_next = float(np.max(q_values[s_next]))
            #       - Accumulate: prob * [reward + gamma * max_q_next]
            total += p * (r + self.gamma * max_q_next)
        # TODO: Return updated Q-value
        return float(total)

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
        policy = np.argmax(q_values, axis=1).astype(np.int64)
        # TODO: Return policy array
        return policy

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
        values = np.max(q_values, axis=1)
        # TODO: Return value function
        return values

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        # TODO: For each state-action pair:
        errors = []
        for s in range(self.n_states):
            for a in range(self.n_actions):
                #       - Compute updated Q-value using Bellman update
                updated = self.bellman_update(s, a, q_values)
                #       - Calculate absolute difference from current Q-value
                errors.append(abs(q_values[s, a] - updated))
        # TODO: Return maximum error
        return float(np.max(errors)) if errors else 0.0