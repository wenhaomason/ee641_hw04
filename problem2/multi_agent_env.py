from typing import List, Optional, Tuple

import numpy as np


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        rows, cols = self.grid_size
        self.grid = np.zeros((rows, cols), dtype=np.int32)

        # Place obstacles while avoiding corners to keep spawn points free
        max_obstacles = 6
        num_obstacles = np.random.randint(1, max_obstacles + 1)
        corners = {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}
        available_positions = [(r, c) for r in range(rows) for c in range(cols)
                               if (r, c) not in corners]
        np.random.shuffle(available_positions)

        for pos in available_positions[:num_obstacles]:
            self.grid[pos] = 1

        # Place single target on a free cell
        free_cells = [(r, c) for r in range(rows) for c in range(cols)
                      if self.grid[r, c] == 0]
        target_idx = np.random.randint(len(free_cells))
        self.target_pos = free_cells[target_idx]
        self.grid[self.target_pos] = 2

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.

        Returns:
            obs_A: Observation for Agent A (11-dimensional vector)
            obs_B: Observation for Agent B (11-dimensional vector)

        Observation format:
        - Elements 0-8: Flattened 3x3 grid patch (row-major order)
        - Element 9: Communication signal from other agent
        - Element 10: Normalized L2 distance between agents
        """
        self._initialize_grid()
        self.step_count = 0

        free_cells = self._find_free_cells()
        if len(free_cells) < 2:
            raise RuntimeError("Not enough free cells to place both agents.")

        chosen_indices = np.random.choice(len(free_cells), size=2, replace=False)
        self.agent_positions[0] = free_cells[int(chosen_indices[0])]
        self.agent_positions[1] = free_cells[int(chosen_indices[1])]

        self.comm_signals = [0.0, 0.0]

        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        return obs_A.astype(np.float32), obs_B.astype(np.float32)

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.

        Args:
            action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
            action_B: Agent B's movement action
            comm_A: Communication signal from Agent A to B
            comm_B: Communication signal from Agent B to A

        Returns:
            observations: Tuple of (obs_A, obs_B), each 11-dimensional
            reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
            done: True if both agents at target or max steps reached
        """
        new_pos_A = self._apply_action(self.agent_positions[0], action_A)
        new_pos_B = self._apply_action(self.agent_positions[1], action_B)
        self.agent_positions = [new_pos_A, new_pos_B]

        # Store communication for next observation (bounded)
        self.comm_signals[0] = float(np.clip(comm_A, 0.0, 1.0))
        self.comm_signals[1] = float(np.clip(comm_B, 0.0, 1.0))

        on_target_A = self.grid[new_pos_A] == 2
        on_target_B = self.grid[new_pos_B] == 2

        reward = -0.1
        if on_target_A and on_target_B:
            reward = 10.0
        elif on_target_A or on_target_B:
            reward = 2.0

        self.step_count += 1
        done = (on_target_A and on_target_B) or (self.step_count >= self.max_steps)

        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        return (obs_A.astype(np.float32), obs_B.astype(np.float32)), float(reward), done

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent.

        Args:
            agent_idx: Agent index (0 for A, 1 for B)

        Returns:
            observation: 10-dimensional vector
        """
        position = self.agent_positions[agent_idx]
        if position is None:
            return np.zeros(11, dtype=np.float32)

        half_window = self.obs_window // 2
        patch = np.full((self.obs_window, self.obs_window), -1.0, dtype=np.float32)

        for dr in range(-half_window, half_window + 1):
            for dc in range(-half_window, half_window + 1):
                row = position[0] + dr
                col = position[1] + dc
                target_row = dr + half_window
                target_col = dc + half_window

                if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]:
                    patch[target_row, target_col] = float(self.grid[row, col])

        patch_flat = patch.flatten()
        comm_value = np.float32(self.comm_signals[1 - agent_idx])
        distance = np.float32(self._compute_normalized_distance())
        observation = np.concatenate((patch_flat, [comm_value, distance])).astype(np.float32)
        return observation

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        row, col = pos
        if not (0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]):
            return False
        return self.grid[row, col] != 1

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.

        Args:
            pos: Current position (row, col)
            action: Movement action (0-4)

        Returns:
            new_pos: Updated position (stays same if invalid)
        """
        deltas = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0)
        }
        delta = deltas.get(action, (0, 0))
        new_pos = (pos[0] + delta[0], pos[1] + delta[1])
        if self._is_valid_position(new_pos):
            return new_pos
        return pos

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid.

        Returns:
            List of (row, col) positions that are free
        """
        rows, cols = self.grid.shape
        free_positions = []
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 0:
                    free_positions.append((r, c))
        return free_positions

    def _compute_normalized_distance(self) -> float:
        """Compute normalized distance between agents."""
        pos_A, pos_B = self.agent_positions
        if pos_A is None or pos_B is None:
            return 0.0

        diff = np.array(pos_A, dtype=np.float32) - np.array(pos_B, dtype=np.float32)
        distance = float(np.linalg.norm(diff))
        norm_factor = float(np.sqrt(self.grid_size[0] ** 2 + self.grid_size[1] ** 2))
        if norm_factor == 0:
            return 0.0
        return float(np.clip(distance / norm_factor, 0.0, 1.0))

    def render(self) -> None:
        """
        Render current environment state.
        """
        rows, cols = self.grid_size
        pos_A = self.agent_positions[0]
        pos_B = self.agent_positions[1]

        def cell_symbol(r: int, c: int) -> str:
            symbol = '.'
            if self.grid[r, c] == 1:
                symbol = 'X'
            elif self.grid[r, c] == 2:
                symbol = 'T'

            if pos_A is not None and (r, c) == pos_A:
                symbol = 'A'
            if pos_B is not None and (r, c) == pos_B:
                symbol = 'B' if symbol != 'A' else 'C'
            return symbol

        print("Environment State (step {}):".format(self.step_count))
        for r in range(rows):
            row_symbols = [cell_symbol(r, c) for c in range(cols)]
            print(" ".join(row_symbols))
        print(f"Comm A->B: {self.comm_signals[0]:.2f} | Comm B->A: {self.comm_signals[1]:.2f}")