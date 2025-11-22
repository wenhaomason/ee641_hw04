"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os


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
        # TODO: Create heatmap with appropriate colormap
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        # TODO: Add colorbar and labels
        # TODO: Save figure to results/visualizations/

        raise NotImplementedError

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        # TODO: Create grid plot
        # TODO: For each state:
        #       - Draw arrow indicating action direction
        #       - Handle special cells appropriately
        # TODO: Mark start, goal, obstacles, penalties
        # TODO: Save figure to results/visualizations/

        raise NotImplementedError

    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        # TODO: For each action:
        #       - Show Q-values as heatmap
        #       - Mark special cells
        # TODO: Add overall title and save

        raise NotImplementedError

    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        # TODO: Use log scale for y-axis
        # TODO: Add legend and labels
        # TODO: Save figure

        raise NotImplementedError

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
        #       - Top left: VI value function
        #       - Top right: QI value function
        #       - Bottom left: VI policy
        #       - Bottom right: QI policy
        # TODO: Highlight any differences
        # TODO: Save comprehensive comparison figure

        raise NotImplementedError


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    # TODO: Load saved value functions and policies
    # TODO: Create visualizer instance
    # TODO: Generate all visualization plots
    # TODO: Print summary statistics

    raise NotImplementedError


if __name__ == '__main__':
    visualize_results()