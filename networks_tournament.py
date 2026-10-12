"""Re-evaluate saved Connect-4 networks against one another, tournament-style."""

import argparse
import re
from pathlib import Path

from model import build_policy_value_net, evaluate_net_against_net
import torch

NETWORK_DIRECTORY_PATTERN = re.compile(r"trained_connect4_net_(\d+)h_")

def load_saved_network(directory):
    """Load the network checkpoint stored in a training directory."""
    match = NETWORK_DIRECTORY_PATTERN.match(directory.name)
    if match is None:
        raise ValueError(f"Could not determine hidden channels from {directory.name}")
    hidden_channels = int(match.group(1))
    net = build_policy_value_net(
        in_channels=2,
        hidden_channels=hidden_channels,
        num_columns=7,
    )
    checkpoint = directory / "trained_connect4_net.pth"
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    net.eval()
    return net


def find_network_directories(root):
    """Find saved training directories containing a network checkpoint."""
    directories = []
    for directory in root.glob("trained_connect4_net_*"):
        if directory.is_dir() and (directory / "trained_connect4_net.pth").is_file():
            directories.append(directory)
    return sorted(directories)


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved Connect-4 networks with MCTS."
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=10,
        help="Number of matches for each tournament round.",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=250,
        help="MCTS simulations per move during evaluation.",
    )
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Directory containing trained network directories.",
    )
    args = parser.parse_args()

    directories = find_network_directories(args.root)
    if len(directories) < 2:
        raise SystemExit("Need at least two saved network directories to compare.")

    champion_directory = directories[0]
    champion = load_saved_network(champion_directory)
    champion_record = {"wins": 0, "losses": 0, "draws": 0}

    for challenger_directory in directories[1:]:
        challenger = load_saved_network(challenger_directory)
        print(f"\nEvaluating champion against challenger:")
        print(f"  Champion:   {champion_directory.name}")
        print(f"  Challenger: {challenger_directory.name}")
        result = evaluate_net_against_net(
            champion,
            challenger,
            num_matches=args.matches,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
        )
        champion_score = result["wins"] + 0.5 * result["draws"]
        challenger_score = result["losses"] + 0.5 * result["draws"]
        if challenger_score > champion_score:
            print("Challenger wins the round and becomes the new champion.")
            champion_directory = challenger_directory
            champion = challenger
            champion_record = {
                "wins": result["losses"],
                "losses": result["wins"],
                "draws": result["draws"],
            }
        else:
            print("Champion retains the title.")
            champion_record = result

    total = sum(champion_record.values())
    score = (
        (champion_record["wins"] + 0.5 * champion_record["draws"]) / total
        if total
        else 0.0
    )
    print("\nTOURNAMENT WINNER")
    print("=================")
    print(champion_directory.name)
    print(
        f"Final round: wins={champion_record['wins']} "
        f"losses={champion_record['losses']} "
        f"draws={champion_record['draws']} score={score:.1%}"
    )

if __name__ == "__main__":
    main()
