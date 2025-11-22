"""
Stochastic gridworld environment for reinforcement learning.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (2, 2), (1, 3)
    - Penalties: (3, 1), (0, 3)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50

        # Define special cells
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

        # Rewards
        self.goal_reward = 10.0
        self.penalty_reward = -5.0
        self.step_cost = -0.1

        # Transition probabilities
        self.prob_intended = 0.8
        self.prob_drift = 0.1

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        # TODO: Initialize agent position to start_pos
        self.position = tuple(self.start_pos)
        # TODO: Reset step counter
        self.step_count = 0
        # TODO: Set done flag to False
        self.done = False
        # TODO: Return state index (use _pos_to_state)
        return self._pos_to_state(self.position)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        # TODO: Check if episode already done
        if getattr(self, "done", False):
            state = self._pos_to_state(self.position)
            return state, 0.0, True, {}
        # TODO: Get next position based on stochastic transitions
        possible = self._get_next_positions(self.position, action)
        next_positions, probs = zip(*possible)
        idx = np.random.choice(len(next_positions), p=np.array(probs))
        next_pos = next_positions[idx]
        # TODO: Calculate reward (use _calculate_reward helper)
        reward = self._calculate_reward(next_pos)
        # TODO: Update position and step count
        self.position = next_pos
        self.step_count = getattr(self, "step_count", 0) + 1
        # TODO: Check termination conditions
        self.done = (self.position == self.goal_pos) or (self.step_count >= self.max_steps)
        # TODO: Return (next_state, reward, done, info)
        return self._pos_to_state(self.position), float(reward), bool(self.done), {}

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a).

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        # TODO: Convert state to position
        pos = self._state_to_pos(state)
        # TODO: For given action, compute all possible next positions
        #       considering stochastic transitions
        transitions = self._get_next_positions(pos, action)
        # TODO: Handle boundary and obstacle collisions
        # (Handled inside _get_next_positions by staying in place if invalid)
        # TODO: Return probability distribution over next states
        prob_dict: Dict[int, float] = {}
        for npos, p in transitions:
            ns = self._pos_to_state(npos)
            prob_dict[ns] = prob_dict.get(ns, 0.0) + p
        return prob_dict

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition.

        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # TODO: Convert next_state to position
        next_pos = self._state_to_pos(next_state)
        # TODO: Check if goal reached (+10)
        # TODO: Check if penalty cell (-5)
        # TODO: Otherwise return step cost (-0.1)
        return float(self._calculate_reward(next_pos))

    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        # TODO: Convert state to position
        pos = self._state_to_pos(state)
        # TODO: Return True if position equals goal_pos
        return pos == self.goal_pos

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        # TODO: Convert 2D position to 1D state index
        # State = row * grid_size + col
        row, col = pos
        return row * self.grid_size + col

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        # TODO: Convert 1D state index to 2D position
        # row = state // grid_size
        # col = state % grid_size
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        r, c = pos
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return False
        # TODO: Check if position is not an obstacle
        return pos not in self.obstacles

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        # TODO: Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
        deltas = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
        }
        # TODO: Get intended direction and perpendicular directions
        intended = action
        drift_left = (action - 1) % 4
        drift_right = (action + 1) % 4
        candidates = [
            (intended, self.prob_intended),
            (drift_left, self.prob_drift),
            (drift_right, self.prob_drift),
        ]
        # TODO: For each possible outcome (intended, drift left, drift right):
        #       - Calculate next position
        #       - If invalid, stay in current position
        #       - Add (position, probability) to list
        results: Dict[Tuple[int, int], float] = {}
        for a, p in candidates:
            dr, dc = deltas[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)
            if not self._is_valid_pos(npos):
                npos = pos
            results[npos] = results.get(npos, 0.0) + p
        # TODO: Merge probabilities for same positions
        merged = [(k, v) for k, v in results.items()]
        return merged

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position.

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        # TODO: Check if position is goal (+10)
        if pos == self.goal_pos:
            return float(self.goal_reward)
        # TODO: Check if position is penalty (-5)
        if pos in self.penalties:
            return float(self.penalty_reward)
        # TODO: Otherwise return step cost (-0.1)
        return float(self.step_cost)

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment.

        Args:
            value_function: Optional value function to display
        """
        # TODO: Create visual representation of grid
        grid = np.zeros((self.grid_size, self.grid_size), dtype=float)
        # TODO: Mark current position, goal, obstacles, penalties
        for r, c in self.obstacles:
            grid[r, c] = -1.0
        for r, c in self.penalties:
            grid[r, c] = -5.0
        gr, gc = self.goal_pos
        grid[gr, gc] = 10.0
        pr, pc = getattr(self, "position", self.start_pos)
        # Represent agent by 1.0 if not overlapping with special cells
        if (pr, pc) not in self.obstacles and (pr, pc) not in self.penalties and (pr, pc) != self.goal_pos:
            grid[pr, pc] = 1.0
        # TODO: If value_function provided, show as heatmap
        # For interface purposes, we will just print a simple textual grid.
        # Users should prefer visualize.py for detailed plots.
        print("GridWorld State:")
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == (pr, pc):
                    row.append("A")
                elif (r, c) == self.goal_pos:
                    row.append("G")
                elif (r, c) in self.obstacles:
                    row.append("X")
                elif (r, c) in self.penalties:
                    row.append("P")
                else:
                    row.append(".")
            print(" ".join(row))