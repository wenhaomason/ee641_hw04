import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from models import AgentDQN
from multi_agent_env import MultiAgentEnv


class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        # Use CPU for small networks
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()

    def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
        """
        Run single evaluation episode.

        Args:
            render: Whether to render environment

        Returns:
            reward: Episode reward
            success: Whether target was reached
            info: Episode statistics
        """
        state_A, state_B = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        comm_history = []
        positions_A: List[Tuple[int, int]] = []
        positions_B: List[Tuple[int, int]] = []

        while not done:
            state_tensor_A = torch.from_numpy(state_A).float().unsqueeze(0).to(self.device)
            state_tensor_B = torch.from_numpy(state_B).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_A, comm_A = self.model_A(state_tensor_A)
                q_B, comm_B = self.model_B(state_tensor_B)
                action_A = int(torch.argmax(q_A, dim=1).cpu().item())
                action_B = int(torch.argmax(q_B, dim=1).cpu().item())
                comm_val_A = float(comm_A.squeeze().cpu().item())
                comm_val_B = float(comm_B.squeeze().cpu().item())

            (state_A, state_B), reward, done = self.env.step(
                action_A, action_B, comm_val_A, comm_val_B)

            positions_A.append(self.env.agent_positions[0])
            positions_B.append(self.env.agent_positions[1])
            comm_history.append((comm_val_A, comm_val_B))
            total_reward += reward
            steps += 1

            if render:
                self.env.render()

        if reward >= 10.0:
            success = True

        info = {
            'steps': steps,
            'comm_history': comm_history,
            'positions_A': positions_A,
            'positions_B': positions_B
        }
        return total_reward, success, info

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Statistics dictionary
        """
        rewards = []
        steps = []
        successes = 0

        for _ in range(num_episodes):
            reward, success, info = self.run_episode(render=False)
            rewards.append(reward)
            steps.append(info.get('steps', 0))
            successes += int(success)

        stats = {
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
            'success_rate': successes / max(1, num_episodes),
            'avg_steps': float(np.mean(steps)) if steps else 0.0,
            'rewards': rewards,
            'steps': steps
        }
        return stats

    def analyze_communication(self, num_episodes: int = 20) -> Dict:
        """
        Analyze emergent communication protocols.

        Returns:
            Communication analysis results
        """
        comm_A_values = []
        comm_B_values = []

        for _ in range(num_episodes):
            _, _, info = self.run_episode(render=False)
            for comm_A, comm_B in info.get('comm_history', []):
                comm_A_values.append(comm_A)
                comm_B_values.append(comm_B)

        def summarize(values: List[float]) -> Dict:
            if not values:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            arr = np.array(values, dtype=np.float32)
            return {
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'min': float(arr.min()),
                'max': float(arr.max())
            }

        analysis = {
            'agent_A': summarize(comm_A_values),
            'agent_B': summarize(comm_B_values),
            'sample_count': len(comm_A_values)
        }
        if comm_A_values and comm_B_values:
            analysis['correlation'] = float(np.corrcoef(comm_A_values, comm_B_values)[0, 1])
        else:
            analysis['correlation'] = 0.0
        return analysis

    def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
        """
        Visualize agent trajectories in an episode.

        Args:
            save_path: Path to save visualization
        """
        reward, success, info = self.run_episode(render=False)
        positions_A = info.get('positions_A', [])
        positions_B = info.get('positions_B', [])

        rows, cols = self.env.grid_size
        plt.figure(figsize=(6, 6))
        plt.title(f"Trajectories (reward={reward:.1f}, success={success})")
        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Display obstacles (1) and target (2) underneath trajectories
        grid = np.array(self.env.grid, dtype=np.int32)
        cmap = ListedColormap(['#FFFFFF', '#6C757D', '#E0C200'])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = BoundaryNorm(bounds, cmap.N)
        plt.imshow(grid, cmap=cmap, norm=norm, alpha=0.5)

        legend_handles = [
            Patch(facecolor='#6C757D', edgecolor='k', label='Obstacle'),
            Patch(facecolor='#E0C200', edgecolor='k', label='Target')
        ]

        if positions_A:
            path_A = np.array(positions_A)
            handle_A, = plt.plot(path_A[:, 1], path_A[:, 0], '-o', label='Agent A', markersize=3)
            legend_handles.append(handle_A)
        if positions_B:
            path_B = np.array(positions_B)
            handle_B, = plt.plot(path_B[:, 1], path_B[:, 0], '-s', label='Agent B', markersize=3)
            legend_handles.append(handle_B)

        plt.legend(handles=legend_handles, loc='upper right')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
        """
        Create heatmap of communication signals across grid positions.

        Args:
            save_path: Path to save figure
        """
        rows, cols = self.env.grid_size
        heat_A = np.zeros((rows, cols), dtype=np.float32)
        heat_B = np.zeros((rows, cols), dtype=np.float32)
        counts_A = np.zeros((rows, cols), dtype=np.float32)
        counts_B = np.zeros((rows, cols), dtype=np.float32)

        samples = max(20, rows * cols // 2)
        for _ in range(samples):
            _, _, info = self.run_episode(render=False)
            for (comm_A, comm_B), pos_A, pos_B in zip(info.get('comm_history', []),
                                                     info.get('positions_A', []),
                                                     info.get('positions_B', [])):
                rA, cA = pos_A
                rB, cB = pos_B
                heat_A[rA, cA] += comm_A
                counts_A[rA, cA] += 1
                heat_B[rB, cB] += comm_B
                counts_B[rB, cB] += 1

        avg_A = np.divide(heat_A, counts_A, out=np.zeros_like(heat_A), where=counts_A > 0)
        avg_B = np.divide(heat_B, counts_B, out=np.zeros_like(heat_B), where=counts_B > 0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im_a = axes[0].imshow(avg_A, cmap='viridis')
        axes[0].set_title('Agent A Communication')
        fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

        im_b = axes[1].imshow(avg_B, cmap='viridis')
        axes[1].set_title('Agent B Communication')
        fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def test_generalization(self, num_configs: int = 10) -> Dict:
        """
        Test generalization to new environment configurations.

        Args:
            num_configs: Number of test configurations

        Returns:
            Generalization performance statistics
        """
        original_env = self.env
        stats = []

        for _ in range(num_configs):
            test_env = MultiAgentEnv(seed=np.random.randint(0, 10000))
            self.env = test_env
            perf = self.evaluate_performance(num_episodes=10)
            stats.append(perf)

        self.env = original_env

        if not stats:
            return {'mean_reward': 0.0, 'success_rate': 0.0}

        mean_rewards = [s['mean_reward'] for s in stats]
        success_rates = [s['success_rate'] for s in stats]
        return {
            'mean_reward': float(np.mean(mean_rewards)),
            'success_rate': float(np.mean(success_rates)),
            'per_config': stats
        }


def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved models

    Returns:
        model_A: Agent A's trained model
        model_B: Agent B's trained model
    """
    if os.path.isdir(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    else:
        checkpoint_path = checkpoint_dir

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_A = AgentDQN(input_dim=11)
    model_B = AgentDQN(input_dim=11)

    if isinstance(checkpoint, dict) and 'agent_A' in checkpoint:
        model_A.load_state_dict(checkpoint['agent_A'])
        model_B.load_state_dict(checkpoint['agent_B'])
    else:
        model_A.load_state_dict(checkpoint)
        model_B.load_state_dict(checkpoint)

    return model_A, model_B


def _save_json(payload: Dict, save_path: str) -> None:
    """Persist a dictionary to disk as JSON."""
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report.

    Args:
        results: Evaluation results
        save_path: Path to save report
    """
    _save_json(results, save_path)


def main():
    """
    Run full evaluation suite on trained models.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained multi-agent models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file or directory')
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render an evaluation episode')
    args = parser.parse_args()

    model_A, model_B = load_trained_models(args.checkpoint)
    env = MultiAgentEnv()
    evaluator = MultiAgentEvaluator(env, model_A, model_B)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(base_dir, 'results')
    results_dir = os.path.join(base_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    performance = evaluator.evaluate_performance(num_episodes=args.num_episodes)
    comm_analysis = evaluator.analyze_communication()
    generalization = evaluator.test_generalization()

    trajectory_path = os.path.join(results_dir, 'trajectories.png')
    heatmap_path = os.path.join(results_dir, 'communication_heatmap.png')
    evaluator.visualize_trajectory(trajectory_path)
    evaluator.plot_communication_heatmap(heatmap_path)

    performance_path = os.path.join(results_dir, 'performance.json')
    communication_path = os.path.join(results_dir, 'communication_analysis.json')
    generalization_path = os.path.join(results_dir, 'generalization.json')
    _save_json(performance, performance_path)
    _save_json(comm_analysis, communication_path)
    _save_json(generalization, generalization_path)

    if args.render:
        evaluator.run_episode(render=True)

    report = {
        'performance': performance,
        'communication': comm_analysis,
        'generalization': generalization,
        'trajectory_path': trajectory_path,
        'heatmap_path': heatmap_path
    }
    report_path = os.path.join(results_dir, 'evaluation_report.json')
    create_evaluation_report(report, report_path)
    print(f"Evaluation report saved to {report_path}")


if __name__ == '__main__':
    main()