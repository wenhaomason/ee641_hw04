import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state_A: Agent A's observation
            state_B: Agent B's observation
            action_A: Agent A's action
            action_B: Agent B's action
            comm_A: Communication from A to B
            comm_B: Communication from B to A
            reward: Shared reward
            next_state_A: Agent A's next observation
            next_state_B: Agent B's next observation
            done: Whether episode terminated
        """
        transition = (state_A, state_B, action_A, action_B,
                      comm_A, comm_B, reward,
                      next_state_A, next_state_B, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate arrays for each component
        """
        if batch_size > len(self.buffer):
            raise ValueError("Not enough samples in replay buffer.")

        batch = random.sample(self.buffer, batch_size)
        state_A, state_B, action_A, action_B, comm_A, comm_B, reward, \
            next_state_A, next_state_B, done = zip(*batch)

        state_A = np.stack(state_A).astype(np.float32)
        state_B = np.stack(state_B).astype(np.float32)
        action_A = np.array(action_A, dtype=np.int64)
        action_B = np.array(action_B, dtype=np.int64)
        comm_A = np.array(comm_A, dtype=np.float32)
        comm_B = np.array(comm_B, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state_A = np.stack(next_state_A).astype(np.float32)
        next_state_B = np.stack(next_state_B).astype(np.float32)
        done = np.array(done, dtype=np.float32)

        return (state_A, state_B, action_A, action_B,
                comm_A, comm_B, reward, next_state_A, next_state_B, done)

    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.

    Samples transitions based on TD-error magnitude.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1

        self.buffer = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.next_idx = 0
        self.size = 0
        self.max_priority = 1.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, *args, **kwargs) -> None:
        """
        Store transition with maximum priority.

        New transitions get maximum priority to ensure they're sampled at least once.
        """
        transition = args
        self.buffer[self.next_idx] = transition
        self.priorities[self.next_idx] = self.max_priority
        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritization.

        Returns:
            transitions: Batch of transitions
            weights: Importance sampling weights
            indices: Indices for updating priorities
        """
        if self.size == 0:
            raise ValueError("Buffer is empty.")

        self.beta = min(1.0, self.beta_start + (self.frame / max(1, self.beta_steps)) * (1.0 - self.beta_start))
        self.frame += 1

        current_priorities = self.priorities[:self.size]
        if np.all(current_priorities == 0):
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float32)
        else:
            scaled = np.power(current_priorities, self.alpha)
            probabilities = scaled / scaled.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        batch = [self.buffer[idx] for idx in indices]

        sampling_probs = probabilities[indices]
        weights = np.power(self.size * sampling_probs, -self.beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        formatted_batch = self._format_batch(batch)
        return formatted_batch, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        for idx, priority in zip(indices, priorities):
            if idx is None:
                continue
            clipped_priority = float(max(priority, 1e-5))
            buffer_idx = int(idx) % self.capacity
            self.priorities[buffer_idx] = clipped_priority
            self.max_priority = max(self.max_priority, clipped_priority)

    def _format_batch(self, batch: List[Tuple]) -> Tuple:
        """Convert list of transitions into batched numpy arrays."""
        state_A, state_B, action_A, action_B, comm_A, comm_B, reward, \
            next_state_A, next_state_B, done = zip(*batch)

        state_A = np.stack(state_A).astype(np.float32)
        state_B = np.stack(state_B).astype(np.float32)
        action_A = np.array(action_A, dtype=np.int64)
        action_B = np.array(action_B, dtype=np.int64)
        comm_A = np.array(comm_A, dtype=np.float32)
        comm_B = np.array(comm_B, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next_state_A = np.stack(next_state_A).astype(np.float32)
        next_state_B = np.stack(next_state_B).astype(np.float32)
        done = np.array(done, dtype=np.float32)

        return (state_A, state_B, action_A, action_B,
                comm_A, comm_B, reward, next_state_A, next_state_B, done)

    def __len__(self) -> int:
        return self.size