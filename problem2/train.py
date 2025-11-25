import argparse
import json
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import AgentDQN
from multi_agent_env import MultiAgentEnv
from replay_buffer import ReplayBuffer


def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation
    """
    masked = np.array(obs, dtype=np.float32, copy=True)
    if masked.shape[0] < 11:
        return masked

    if mode == 'independent':
        masked[9] = 0.0
        masked[10] = 0.0
    elif mode == 'comm':
        masked[10] = 0.0

    return masked


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.

    Handles training loop, exploration, and network updates.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer.

        Args:
            env: Multi-agent environment
            args: Training arguments
        """
        self.env = env
        self.args = args

        # Use CPU for small networks
        self.device = torch.device("cpu")

        input_dim = 11
        num_actions = 5
        self.network_A = AgentDQN(input_dim=input_dim,
                                  hidden_dim=self.args.hidden_dim,
                                  num_actions=num_actions).to(self.device)
        self.network_B = AgentDQN(input_dim=input_dim,
                                  hidden_dim=self.args.hidden_dim,
                                  num_actions=num_actions).to(self.device)

        self.target_A = AgentDQN(input_dim=input_dim,
                                 hidden_dim=self.args.hidden_dim,
                                 num_actions=num_actions).to(self.device)
        self.target_B = AgentDQN(input_dim=input_dim,
                                 hidden_dim=self.args.hidden_dim,
                                 num_actions=num_actions).to(self.device)
        self.target_A.load_state_dict(self.network_A.state_dict())
        self.target_B.load_state_dict(self.network_B.state_dict())

        self.optimizer_A = optim.Adam(self.network_A.parameters(), lr=self.args.lr)
        self.optimizer_B = optim.Adam(self.network_B.parameters(), lr=self.args.lr)

        self.replay_buffer = ReplayBuffer(capacity=50000, seed=self.args.seed)
        self.epsilon = self.args.epsilon_start
        self.num_actions = 5
        self.comm_loss_weight = getattr(self.args, 'comm_loss_weight', 0.1)
        self.comm_target_scale = getattr(self.args, 'comm_target_scale', 10.0)

    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Agent observation (11-dimensional, may need masking)
            network: Agent's DQN
            epsilon: Exploration probability

        Returns:
            action: Selected action
            comm_signal: Communication signal
        """
        masked_state = apply_observation_mask(state, self.args.mode)
        state_tensor = torch.from_numpy(masked_state).float().unsqueeze(0).to(self.device)

        was_training = network.training
        network.eval()
        with torch.no_grad():
            q_values, comm_signal = network(state_tensor)
            comm_value = float(comm_signal.squeeze().cpu().item())
        if was_training:
            network.train()

        if np.random.random() < epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            action = int(torch.argmax(q_values, dim=1).cpu().item())

        return action, np.clip(comm_value, 0.0, 1.0)

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)
        state_A, state_B, action_A, action_B, _, _, reward, \
            next_state_A, next_state_B, done = batch

        def mask_batch(states: np.ndarray) -> np.ndarray:
            return np.stack([apply_observation_mask(s, self.args.mode) for s in states]).astype(np.float32)

        state_A = torch.from_numpy(mask_batch(state_A)).to(self.device)
        state_B = torch.from_numpy(mask_batch(state_B)).to(self.device)
        next_state_A = torch.from_numpy(mask_batch(next_state_A)).to(self.device)
        next_state_B = torch.from_numpy(mask_batch(next_state_B)).to(self.device)

        action_A = torch.from_numpy(action_A).long().unsqueeze(1).to(self.device)
        action_B = torch.from_numpy(action_B).long().unsqueeze(1).to(self.device)
        reward = torch.from_numpy(reward).float().unsqueeze(1).to(self.device)
        done = torch.from_numpy(done).float().unsqueeze(1).to(self.device)

        current_q_A, comm_pred_A = self.network_A(state_A)
        current_q_B, comm_pred_B = self.network_B(state_B)

        q_A = current_q_A.gather(1, action_A)
        q_B = current_q_B.gather(1, action_B)

        with torch.no_grad():
            target_q_A, _ = self.target_A(next_state_A)
            target_q_B, _ = self.target_B(next_state_B)
            max_next_A = target_q_A.max(dim=1, keepdim=True)[0]
            max_next_B = target_q_B.max(dim=1, keepdim=True)[0]

            target_A = reward + self.args.gamma * max_next_A * (1 - done)
            target_B = reward + self.args.gamma * max_next_B * (1 - done)

        loss_A = F.mse_loss(q_A, target_A)
        loss_B = F.mse_loss(q_B, target_B)
        comm_target_A = torch.sigmoid(target_A / self.comm_target_scale)
        comm_target_B = torch.sigmoid(target_B / self.comm_target_scale)
        comm_loss_A = F.mse_loss(comm_pred_A, comm_target_A)
        comm_loss_B = F.mse_loss(comm_pred_B, comm_target_B)
        loss = loss_A + loss_B + self.comm_loss_weight * (comm_loss_A + comm_loss_B)

        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
        loss.backward()
        self.optimizer_A.step()
        self.optimizer_B.step()

        return float(loss.item())

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        state_A, state_B = self.env.reset()
        episode_reward = 0.0
        success = False
        done = False

        while not done:
            action_A, comm_A = self.select_action(state_A, self.network_A, self.epsilon)
            action_B, comm_B = self.select_action(state_B, self.network_B, self.epsilon)

            (next_state_A, next_state_B), reward, done = self.env.step(
                action_A, action_B, comm_A, comm_B)

            self.replay_buffer.push(state_A, state_B,
                                    action_A, action_B,
                                    comm_A, comm_B,
                                    reward,
                                    next_state_A, next_state_B,
                                    done)

            state_A, state_B = next_state_A, next_state_B
            episode_reward += reward

            if len(self.replay_buffer) >= self.args.batch_size:
                self.update_networks(self.args.batch_size)

            if done and reward >= 10.0:
                success = True

        return episode_reward, success

    def train(self) -> None:
        """
        Main training loop.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, 'results')
        models_dir = os.path.join(results_dir, 'agent_models')
        logs_dir = os.path.join(results_dir, 'training_logs')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        reward_history = []
        success_history = []
        best_success = 0.0

        for episode in range(1, self.args.num_episodes + 1):
            reward, success = self.train_episode()
            reward_history.append(float(reward))
            success_history.append(1 if success else 0)

            self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)

            if episode % self.args.target_update == 0:
                self.target_A.load_state_dict(self.network_A.state_dict())
                self.target_B.load_state_dict(self.network_B.state_dict())

            if episode % 50 == 0:
                avg_reward = float(np.mean(reward_history[-50:]))
                success_rate = float(np.mean(success_history[-50:]))
                print(f"Episode {episode}/{self.args.num_episodes} | "
                      f"Avg Reward (50 ep): {avg_reward:.2f} | "
                      f"Success Rate: {success_rate:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

            if episode % self.args.save_freq == 0:
                checkpoint = {
                    'agent_A': self.network_A.state_dict(),
                    'agent_B': self.network_B.state_dict(),
                    'episode': episode,
                    'reward_history': reward_history,
                    'success_history': success_history
                }
                torch.save(checkpoint, os.path.join(models_dir, f'checkpoint_{episode}.pth'))

                window = success_history[-self.args.save_freq:] or success_history
                window_success = float(np.mean(window))
                if window_success >= best_success:
                    best_success = window_success
                    torch.save(checkpoint, os.path.join(models_dir, 'best_checkpoint.pth'))

        history = {
            'episode_rewards': reward_history,
            'success_flags': success_history
        }
        with open(os.path.join(logs_dir, f'history_{self.args.mode}.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        scripted_A = torch.jit.script(self.network_A.cpu())
        scripted_B = torch.jit.script(self.network_B.cpu())
        scripted_A.save(os.path.join(models_dir, 'agent_A_scripted.pt'))
        scripted_B.save(os.path.join(models_dir, 'agent_B_scripted.pt'))
        self.network_A.to(self.device)
        self.network_B.to(self.device)

    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            mean_reward: Average reward
            success_rate: Fraction of successful episodes
        """
        prev_mode_A = self.network_A.training
        prev_mode_B = self.network_B.training
        self.network_A.eval()
        self.network_B.eval()

        rewards = []
        successes = 0

        for _ in range(num_episodes):
            state_A, state_B = self.env.reset()
            done = False
            episode_reward = 0.0
            reward = 0.0

            while not done:
                action_A, comm_A = self.select_action(state_A, self.network_A, epsilon=0.0)
                action_B, comm_B = self.select_action(state_B, self.network_B, epsilon=0.0)
                (state_A, state_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
                episode_reward += reward

            rewards.append(episode_reward)
            if reward >= 10.0:
                successes += 1

        if prev_mode_A:
            self.network_A.train()
        if prev_mode_B:
            self.network_B.train()

        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        success_rate = successes / max(1, num_episodes)
        return mean_reward, success_rate


def main():
    """
    Parse arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.9995,
                       help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100,
                       help='Target network update frequency')
    parser.add_argument('--comm-loss-weight', type=float, default=0.1,
                       help='Weight applied to auxiliary communication loss')
    parser.add_argument('--comm-target-scale', type=float, default=10.0,
                       help='Scale factor when mapping TD targets into [0,1] for comm supervision')

    # Ablation study mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    env = MultiAgentEnv(grid_size=tuple(args.grid_size),
                        obs_window=3,
                        max_steps=args.max_steps,
                        seed=args.seed)

    trainer = MultiAgentTrainer(env, args)
    trainer.train()
    mean_reward, success_rate = trainer.evaluate(num_episodes=20)
    print(f"Final Evaluation -> Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2f}")


if __name__ == '__main__':
    main()