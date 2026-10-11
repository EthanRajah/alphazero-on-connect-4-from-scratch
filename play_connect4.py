"""Launch the graphical Connect-4 evaluator with a saved network."""

import argparse

import torch

from connect4_gui import play_against_network
from model import build_policy_value_net


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Play Connect-4 against a saved AlphaZero network.")
    parser.add_argument(
        "--checkpoint",
        default="trained_connect4_net_16h_20s_1.5c_1.0t.pth",
        help="Path to a saved policy-value network state dict.",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=16,
        help="Hidden-channel count used when the network was trained.",
    )
    parser.add_argument("--matches", type=int, default=1, help="Number of games to play.")
    parser.add_argument(
        "--num_simulations",
        type=positive_int,
        default=8,
        help="Number of MCTS simulations per network move.",
    )
    parser.add_argument(
        "--c_puct",
        type=positive_float,
        default=1.5,
        help="MCTS exploration constant.",
    )
    parser.add_argument(
        "--temperature",
        type=positive_float,
        default=1.0,
        help="Temperature used to convert MCTS visit counts into move probabilities.",
    )
    args = parser.parse_args()

    net = build_policy_value_net(
        in_channels=2, hidden_channels=args.hidden_channels, num_columns=7
    )
    try:
        state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()
    play_against_network(
        net,
        num_matches=args.matches,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
