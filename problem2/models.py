from typing import Tuple

import torch
import torch.nn as nn


class AgentDQN(nn.Module):
    """
    Deep Q-Network for agent with communication capability.

    Network processes observations and outputs both Q-values and communication signal.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize DQN with dual outputs.

        Args:
            input_dim: Dimension of input observation (default 10)
            hidden_dim: Number of hidden units
            num_actions: Number of discrete actions (default 5)
        """
        super(AgentDQN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.comm_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            action_values: Q-values for each action [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        features = self.feature_extractor(x)
        action_values = self.action_head(features)
        comm_signal = torch.sigmoid(self.comm_head(features))
        return action_values, comm_signal


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for improved value estimation.

    Separates value and advantage streams for better learning.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize Dueling DQN.

        Args:
            input_dim: Dimension of input observation
            hidden_dim: Number of hidden units
            num_actions: Number of discrete actions
        """
        super(DuelingDQN, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        self.comm_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dueling architecture.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            q_values: Combined Q-values [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        features = self.features(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        q_values = value + (advantages - advantages_mean)
        comm_signal = torch.sigmoid(self.comm_head(features))
        return q_values, comm_signal