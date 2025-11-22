"""
Training script for Value Iteration and Q-Iteration.
"""

import argparse
import json
import os

import numpy as np
from environment import GridWorldEnv
from q_iteration import QIteration
from value_iteration import ValueIteration


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
    env = GridWorldEnv(seed=args.seed)
    # TODO: Run Value Iteration
    #       - Create ValueIteration solver
    vi_solver = ValueIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    #       - Solve for optimal values
    vi_values, vi_iters = vi_solver.solve(max_iterations=args.max_iter)
    #       - Extract policy
    vi_policy = vi_solver.extract_policy(vi_values)
    #       - Save results
    np.savez('results/value_function.npz', values=vi_values)
    np.savez('results/vi_policy.npz', policy=vi_policy)
    # TODO: Run Q-Iteration
    #       - Create QIteration solver
    qi_solver = QIteration(env, gamma=args.gamma, epsilon=args.epsilon)
    #       - Solve for optimal Q-values
    qi_qvalues, qi_iters = qi_solver.solve(max_iterations=args.max_iter)
    #       - Extract policy and values
    qi_policy = qi_solver.extract_policy(qi_qvalues)
    qi_values = qi_solver.extract_values(qi_qvalues)
    #       - Save results
    np.savez('results/q_function.npz', q_values=qi_qvalues)
    np.savez('results/qi_policy.npz', policy=qi_policy)
    # TODO: Compare algorithms
    #       - Print convergence statistics
    print(json.dumps({
        "value_iteration_iterations": int(vi_iters),
        "q_iteration_iterations": int(qi_iters)
    }, indent=2))
    #       - Check if policies match
    policies_match = bool(np.all(vi_policy == qi_policy))
    #       - Save comparison results
    with open('results/comparison.json', 'w') as f:
        json.dump({
            "vi_iterations": int(vi_iters),
            "qi_iterations": int(qi_iters),
            "policies_match": policies_match
        }, f, indent=2)
    print(f"Policies match: {policies_match}")


if __name__ == '__main__':
    main()