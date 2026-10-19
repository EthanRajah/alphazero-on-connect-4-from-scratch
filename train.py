"""
AlphaZero on Connect-4 from Scratch

Training script for long-running self-play training and hyperparameter tuning.

Run this with: python train.py
Uses functions defined in model.py.
"""

import argparse
import os

from model import *
import numpy as np
import torch

import matplotlib.pyplot as plt

def train_connect4(hidden_channels, learning_rate, batch_size, num_simulations, c_puct, temperature, num_games, num_iterations, num_epochs):
    """Train a policy-value network on Connect-4 using self play and MCTS"""
    # Run training loop
    net = build_policy_value_net(in_channels=2, hidden_channels=hidden_channels, num_columns=7)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    training_history = train_loop(net, optimizer, num_iterations, num_games, num_simulations, c_puct, batch_size, num_epochs=num_epochs, temperature=temperature)
    # Save the trained network and plot the training history
    train_dir = "training/trained_connect4_net_{}h_{}b_{}lr_{}s_{}c_{}t_{}g_{}i_{}e".format(hidden_channels, batch_size, learning_rate, num_simulations, c_puct, temperature, num_games, num_iterations, num_epochs)
    save_network(net, filename="trained_connect4_net.pth", save_dir=train_dir)
    plot_training_history(training_history, save_dir=train_dir)
    # Run evaluations with greedy play against random
    net.eval()
    baseline_results = evaluate_connect4(net, num_matches=5, seed=0, mode="baseline")
    return training_history, baseline_results

def plot_training_history(training_history, save_dir=None):
    """Plot the total loss over the training iterations."""
    total_losses = [np.mean([epoch_loss["total"] for epoch_loss in iteration["losses"]]) for iteration in training_history]
    policy_losses = [np.mean([epoch_loss["policy"] for epoch_loss in iteration["losses"]]) for iteration in training_history]
    value_losses = [np.mean([epoch_loss["value"] for epoch_loss in iteration["losses"]]) for iteration in training_history]
    # Create three different plots for total loss, policy loss, and value loss
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label="Total Loss", color="blue")
    plt.plot(policy_losses, label="Policy Loss", color="orange")
    plt.plot(value_losses, label="Value Loss", color="green")
    plt.title("Training Loss Over Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_loss.png"))

def evaluate_connect4(net, alt_net=None, num_matches=1, seed=0, mode="baseline"):
    """Evaluate the trained policy-value network against a random player. Baseline will use uniform random moves, while network evaluation will use another trained network."""
    if mode == "baseline":
        print("Starting baseline evaluation against random player...")
        return evaluate_against_random(net, num_matches, seed)
    elif mode == "network" and alt_net is not None:
        print("Starting evaluation against another trained network...")
        return evaluate_net_against_net(net, alt_net, num_matches)

def save_network(net, filename="trained_connect4_net.pth", save_dir="."):
    """Save the trained policy-value network to a file in cwd."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(save_dir, filename))
    print(f"Saved trained network to {os.path.join(save_dir, filename)}")

def load_network(filename="trained_connect4_net.pth", load_dir=".", hidden_channels=16):
    """Load a trained policy-value network from a file in cwd."""
    net = build_policy_value_net(in_channels=2, hidden_channels=hidden_channels, num_columns=7)
    net.load_state_dict(torch.load(os.path.join(load_dir, filename), map_location="cpu", weights_only=True))
    net.eval()
    print(f"Loaded trained network from {os.path.join(load_dir, filename)}")
    return net

def hyperparameter_tuning():
    """Run a hyperparameter tuning loop to find the best combination of parameters for training."""
    # Network parameters
    # hidden_channels_list = [8, 16, 24]
    # batch_size_list = [4, 8, 16]
    # learning_rate_list = [5e-3, 1e-4, 5e-4]
    # MCTS parameters
    # num_simulations_list = [25] # MCTS strength
    # num_games_list = [20] # Data volume per iteration
    # num_iterations_list = [5]
    # num_epochs_list = [5]

    # Network parameters - tuned first indepdently of MCTS parameters
    hidden_channels_list = [16]
    batch_size_list = [8]
    learning_rate_list = [5e-4]
    # MCTS parameters
    num_simulations_list = [25, 50, 100, 250, 500] # MCTS strength
    num_games_list = [5, 10, 20, 50] # Data volume per iteration
    num_iterations_list = [5, 10, 15, 20]
    num_epochs_list = [5, 10, 15]

    best_loss = float("inf")
    best_params = None

    hyperparam_log = open("hyperparameter_tuning_log.txt", "w", buffering=1)

    for hidden_channels in hidden_channels_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                for num_simulations in num_simulations_list:
                    for num_games in num_games_list:
                        for num_iterations in num_iterations_list:
                            for num_epochs in num_epochs_list:
                                config_text = f"hidden_channels={hidden_channels}, batch_size={batch_size}, learning_rate={learning_rate}, num_simulations={num_simulations}, c_puct={1.5}, temperature={1.0}, num_games={num_games}, num_iterations={num_iterations}, num_epochs={num_epochs}"
                                print(f"Training with params: {config_text}")
                                hyperparam_log.write("\n" + "=" * 80 + "\n")
                                hyperparam_log.write("CONFIGURATION\n")
                                hyperparam_log.write(f"{config_text}\n")
                                hyperparam_log.write("-" * 80 + "\n")
                                hyperparam_log.write("TRAINING IN PROGRESS\n")
                                hyperparam_log.flush()
                                training_history, evaluate_results = train_connect4(
                                    hidden_channels=hidden_channels,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    num_simulations=num_simulations,
                                    c_puct=1.5,
                                    temperature=1.0,
                                    num_games=num_games,
                                    num_iterations=num_iterations,
                                    num_epochs=num_epochs
                                )
                                total_losses = [np.mean([epoch_loss["total"] for epoch_loss in iteration["losses"]]) for iteration in training_history]
                                avg_loss = np.mean(total_losses)
                                print(f"Average loss for this configuration: {avg_loss}")
                                prev_net = None
                                if best_params is not None:
                                    # Load the previous best network for head-to-head evaluation.
                                    previous_dir = "training/trained_connect4_net_{}h_{}b_{}lr_{}s_{}c_{}t_{}g_{}i_{}e".format(*best_params)
                                    previous_filename = "trained_connect4_net.pth"
                                    prev_net = load_network(filename=previous_filename, load_dir=previous_dir, hidden_channels=best_params[0])
                                # Update best loss and parameters if the current configuration is better
                                eval_results = None
                                if avg_loss < best_loss:
                                    print(f"New best params found: {best_params} with loss {best_loss}")
                                    # Run evaluation only when a previous best exists.
                                    if prev_net is not None:
                                        current_dir = "training/trained_connect4_net_{}h_{}b_{}lr_{}s_{}c_{}t_{}g_{}i_{}e".format(*best_params)
                                        current_filename = "trained_connect4_net.pth"
                                        net = load_network(filename=current_filename, load_dir=current_dir, hidden_channels=best_params[0])
                                        eval_results = evaluate_net_against_net(net, prev_net, num_matches=25)
                                        # Only update the current network as being the best if it is better than or equal to the previous best in evaluation
                                        if eval_results["wins"] >= eval_results["losses"]:
                                            best_loss = avg_loss
                                            best_params = (hidden_channels, batch_size, learning_rate, num_simulations, 1.5, 1.0, num_games, num_iterations, num_epochs)
                                    else:
                                        best_loss = avg_loss
                                        best_params = (hidden_channels, batch_size, learning_rate, num_simulations, 1.5, 1.0, num_games, num_iterations, num_epochs)
                                # Update log file with the average loss and best params so far, as well as network evaluation results if available
                                hyperparam_log.write("TRAINING RESULTS\n")
                                hyperparam_log.write(f"Average loss: {avg_loss}\n")
                                hyperparam_log.write(f"Baseline evaluation: {evaluate_results}\n")
                                hyperparam_log.write(f"Best loss so far: {best_loss}\n")
                                hyperparam_log.write(f"Best parameters so far: {best_params}\n")
                                hyperparam_log.write("\nEVALUATION RESULTS\n")
                                if eval_results is not None:
                                    hyperparam_log.write(f"Against previous best network: {eval_results}\n")
                                else:
                                    hyperparam_log.write("Against previous best network: not run\n")
                                hyperparam_log.write("=" * 80 + "\n")
                                hyperparam_log.flush()
    print(f"Best hyperparameters found: {best_params} with loss {best_loss}")
    hyperparam_log.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=8, help="Number of channels in the hidden layers of the policy-value network.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of steps from the rollout to use per gradient descent step.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_simulations", type=int, default=8, help="Number of MCTS simulations per move.")
    parser.add_argument("--c_puct", type=float, default=1.5, help="Exploration constant for MCTS.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter for action selection.")
    parser.add_argument("--num_games", type=int, default=2, help="Number of self-play games to generate.")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of training iterations as a whole.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of iterations of the collected rollout to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    print("==== AlphaZero on Connect-4 Training ===")
    print("Training Connect-4 with the following parameters:")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of simulations: {args.num_simulations}")
    print(f"c_puct: {args.c_puct}")
    print(f"Temperature: {args.temperature}")
    print(f"Number of self-play games: {args.num_games}")
    print(f"Number of training iterations: {args.num_iterations}")
    print(f"Number of epochs per iteration: {args.num_epochs}")
    print("==== Starting Training ===")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    training_history, evaluate_results = train_connect4(
        hidden_channels=args.hidden_channels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        num_games=args.num_games,
        num_iterations=args.num_iterations,
        num_epochs=args.num_epochs
    )
    # hyperparameter_tuning()

if __name__ == "__main__":
    main()