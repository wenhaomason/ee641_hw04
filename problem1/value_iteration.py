"""
Value Iteration algorithm for solving MDPs.
"""

from typing import Tuple

import numpy as np
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
        values = np.zeros(self.n_states, dtype=float)
        # TODO: Iterate until convergence:
        n_iter = 0
        for it in range(max_iterations):
            n_iter = it + 1
            new_values = np.copy(values)
            #       - For each state:
            for s in range(self.n_states):
                #           - Compute Q(s,a) for all actions using Bellman backup
                if self.env.is_terminal(s):
                    new_values[s] = 0.0
                else:
                    q_vals = self.compute_q_values(s, values)
                    #           - Set V(s) = max_a Q(s,a)
                    new_values[s] = float(np.max(q_vals))
            #       - Check convergence: max|V_new - V_old| < epsilon
            delta = float(np.max(np.abs(new_values - values)))
            values = new_values
            #       - Update value function
            if delta < self.epsilon:
                break
        # TODO: Return final values and iteration count
        return values, n_iter

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
        q_values = np.zeros(self.n_actions, dtype=float)
        for a in range(self.n_actions):
            #       - Get transition probabilities P(s'|s,a)
            trans = self.env.get_transition_prob(state, a)
            q = 0.0
            for s_next, p in trans.items():
                #       - Compute expected value:
                #           Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                r = self.env.get_reward(state, a, s_next)
                q += p * (r + self.gamma * values[s_next])
            q_values[a] = q
        # TODO: Return Q-values array
        return q_values

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.

        Args:
            values: Optimal value function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        policy = np.zeros(self.n_states, dtype=np.int64)
        for s in range(self.n_states):
            #       - Compute Q-values for all actions
            if self.env.is_terminal(s):
                policy[s] = 0
            else:
                q_vals = self.compute_q_values(s, values)
                #       - Select action with maximum Q-value
                policy[s] = int(np.argmax(q_vals))
        # TODO: Return policy array
        return policy

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
        if self.env.is_terminal(state):
            return 0.0
        # TODO: Compute Q-values for all actions
        q_vals = self.compute_q_values(state, values)
        # TODO: Return maximum Q-value
        return float(np.max(q_vals))

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
        errors = []
        for s in range(self.n_states):
            #       - Compute optimal value using Bellman backup
            backed = self.bellman_backup(s, values)
            #       - Calculate absolute difference from current value
            errors.append(abs(values[s] - backed))
        # TODO: Return maximum error
        return float(np.max(errors)) if errors else 0.0