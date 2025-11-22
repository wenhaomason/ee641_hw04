"""
Training script for Value Iteration and Q-Iteration.
"""

import numpy as np
import argparse
import json
import os
from environment import GridWorldEnv
from value_iteration import ValueIteration
from q_iteration import QIteration


def main():
    """
    Run both algorithms and save results.
    """
    parser = argparse.ArgumentParser(description='Train RL algorithms on GridWorld')
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations')
    args = parser.parse_args()

    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/visualizations', exist_ok=True)

    # TODO: Initialize environment with seed
    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    #       - Solve for optimal values
    #       - Extract policy
    #       - Save results
    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    #       - Solve for optimal Q-values
    #       - Extract policy and values
    #       - Save results
    # TODO: Compare algorithms
    #       - Print convergence statistics
    #       - Check if policies match
    #       - Save comparison results

    raise NotImplementedError


if __name__ == '__main__':
    main()