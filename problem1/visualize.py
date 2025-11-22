"""
Visualization utilities for gridworld and policies.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
        """
        # TODO: Reshape values to 2D grid
        grid = values.reshape(self.grid_size, self.grid_size)
        # TODO: Create heatmap with appropriate colormap
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap='viridis', origin='upper')
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        ax.scatter(self.start_pos[1], self.start_pos[0], marker='s', color='white', label='Start')
        ax.scatter(self.goal_pos[1], self.goal_pos[0], marker='*', color='yellow', label='Goal')
        if self.obstacles:
            ys, xs = zip(*self.obstacles)
            ax.scatter(xs, ys, marker='X', color='red', label='Obstacle')
        if self.penalties:
            ys, xs = zip(*self.penalties)
            ax.scatter(xs, ys, marker='o', color='orange', label='Penalty')
        # TODO: Add colorbar and labels
        ax.set_title(title)
        ax.set_xlabel('Col')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
        # TODO: Save figure to results/visualizations/
        os.makedirs('results/visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/visualizations/value_heatmap.png', dpi=150)
        plt.close(fig)

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        # TODO: Create grid plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        # TODO: For each state:
        for s in range(self.grid_size * self.grid_size):
            r, c = divmod(s, self.grid_size)
            a = policy[s]
            #       - Draw arrow indicating action direction
            dr, dc = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}.get(int(a), (0, 0))
            if (r, c) in self.obstacles:
                ax.text(c, r, 'X', ha='center', va='center', color='red')
            elif (r, c) in self.penalties:
                ax.text(c, r, 'P', ha='center', va='center', color='orange')
            elif (r, c) == self.goal_pos:
                ax.text(c, r, 'G', ha='center', va='center', color='yellow')
            elif (r, c) == self.start_pos:
                ax.text(c, r, 'S', ha='center', va='center', color='white')
                ax.arrow(c, r, dc, dr, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            else:
                ax.arrow(c, r, dc, dr, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        # TODO: Mark start, goal, obstacles, penalties
        ax.set_title(title)
        # TODO: Save figure to results/visualizations/
        os.makedirs('results/visualizations', exist_ok=True)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('results/visualizations/policy_arrows.png', dpi=150)
        plt.close(fig)

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        n_actions = q_values.shape[1]
        fig, axs = plt.subplots(1, n_actions, figsize=(4 * n_actions, 4))
        if n_actions == 1:
            axs = [axs]
        # TODO: For each action:
        for a in range(n_actions):
            grid = q_values[:, a].reshape(self.grid_size, self.grid_size)
            im = axs[a].imshow(grid, cmap='coolwarm', origin='upper')
            axs[a].set_title(f"Action {a}")
            axs[a].set_xlabel('Col')
            axs[a].set_ylabel('Row')
            for r, c in self.obstacles:
                axs[a].text(c, r, 'X', ha='center', va='center', color='black')
            for r, c in self.penalties:
                axs[a].text(c, r, 'P', ha='center', va='center', color='black')
            gr, gc = self.goal_pos
            axs[a].text(gc, gr, 'G', ha='center', va='center', color='black')
            plt.colorbar(im, ax=axs[a], fraction=0.046, pad=0.04)
        # TODO: Add overall title and save
        fig.suptitle(title)
        os.makedirs('results/visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/visualizations/q_functions.png', dpi=150)
        plt.close(fig)

    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(vi_history, label='Value Iteration')
        ax.plot(qi_history, label='Q-Iteration')
        # TODO: Use log scale for y-axis
        ax.set_yscale('log')
        # TODO: Add legend and labels
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Bellman Error')
        ax.legend()
        # TODO: Save figure
        os.makedirs('results/visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/visualizations/convergence.png', dpi=150)
        plt.close(fig)

    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
        """
        # TODO: Create 2x2 subplot
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        #       - Top left: VI value function
        im0 = axs[0, 0].imshow(vi_values.reshape(self.grid_size, self.grid_size),
                               cmap='viridis', origin='upper')
        axs[0, 0].set_title('VI Values')
        plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
        #       - Top right: QI value function
        im1 = axs[0, 1].imshow(qi_values.reshape(self.grid_size, self.grid_size),
                               cmap='viridis', origin='upper')
        axs[0, 1].set_title('QI Values')
        plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
        #       - Bottom left: VI policy
        axs[1, 0].set_xlim(-0.5, self.grid_size - 0.5)
        axs[1, 0].set_ylim(-0.5, self.grid_size - 0.5)
        axs[1, 0].grid(True, linestyle='--', linewidth=0.5)
        for s in range(self.grid_size * self.grid_size):
            r, c = divmod(s, self.grid_size)
            a = vi_policy[s]
            dr, dc = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}.get(int(a), (0, 0))
            axs[1, 0].arrow(c, r, dc, dr, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        axs[1, 0].invert_yaxis()
        axs[1, 0].set_title('VI Policy')
        #       - Bottom right: QI policy
        axs[1, 1].set_xlim(-0.5, self.grid_size - 0.5)
        axs[1, 1].set_ylim(-0.5, self.grid_size - 0.5)
        axs[1, 1].grid(True, linestyle='--', linewidth=0.5)
        for s in range(self.grid_size * self.grid_size):
            r, c = divmod(s, self.grid_size)
            a = qi_policy[s]
            dr, dc = {0: (-0.3, 0), 1: (0, 0.3), 2: (0.3, 0), 3: (0, -0.3)}.get(int(a), (0, 0))
            axs[1, 1].arrow(c, r, dc, dr, head_width=0.2, head_length=0.2, fc='green', ec='green')
        axs[1, 1].invert_yaxis()
        axs[1, 1].set_title('QI Policy')
        # TODO: Highlight any differences
        # Basic difference heatmap (absolute difference)
        diff = np.abs(vi_values.reshape(self.grid_size, self.grid_size) -
                      qi_values.reshape(self.grid_size, self.grid_size))
        max_diff = float(np.max(diff))
        if max_diff > 1e-6:
            fig.suptitle(f'Comparison (max value diff: {max_diff:.4f})')
        # TODO: Save comprehensive comparison figure
        os.makedirs('results/visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig('results/visualizations/comparison.png', dpi=150)
        plt.close(fig)


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    # TODO: Load saved value functions and policies
    vi_values = np.load('results/value_function.npz')['values']
    qi_q = np.load('results/q_function.npz')['q_values']
    vi_policy = np.load('results/vi_policy.npz')['policy']
    qi_policy = np.load('results/qi_policy.npz')['policy']
    qi_values = np.max(qi_q, axis=1)
    # TODO: Create visualizer instance
    viz = GridWorldVisualizer(grid_size=5)
    # TODO: Generate all visualization plots
    viz.plot_value_function(vi_values, title="Value Function (VI)")
    viz.plot_policy(vi_policy, title="Optimal Policy (VI)")
    viz.plot_q_function(qi_q, title="Q-Function (QI)")
    viz.create_comparison_figure(vi_values, qi_values, vi_policy, qi_policy)
    # TODO: Print summary statistics
    print(f"VI values: min={vi_values.min():.3f}, max={vi_values.max():.3f}")
    print(f"QI values: min={qi_values.min():.3f}, max={qi_values.max():.3f}")


if __name__ == '__main__':
    visualize_results()