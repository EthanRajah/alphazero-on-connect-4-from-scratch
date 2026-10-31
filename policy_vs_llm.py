"""Assess the performance of a policy network against a Monte Carlo Tree Search (MCTS) agent in Connect 4 against an LLM agent hosted on Ollama"""

import argparse
import logging
import os

from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
from train import load_network
from model import *

prompt_logger = logging.getLogger("policy_vs_llm.prompts")
game_logger = logging.getLogger("policy_vs_llm.game")

def setup_logging(log_dir="."):
    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter("%(message)s")

    for log, filename in (
        (prompt_logger, "prompts.log"),
        (game_logger, "game.log"),
    ):
        log.setLevel(logging.INFO)
        log.propagate = False
        handler = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
        handler.setFormatter(formatter)
        log.addHandler(handler)

def game_print(*args, **kwargs):
    message = kwargs.get("sep", " ").join(str(arg) for arg in args)
    print(*args, **kwargs)
    game_logger.info(message.rstrip("\n"))

def test_call():
    response: ChatResponse = chat(
        model='qwen3.5:9b',
        messages=[
            {
            'role': 'user',
            'content': 'Why is the sky blue?',
            },
        ],
    )
    print(response.message.content)
    return response.message.content
class Connect4OutputModel(BaseModel):
    action: int
    reasoning: str

def ollama_call(prompt):
    response = chat(
        model='qwen3.5:9b',
        messages=[{'role': 'user', 'content': prompt}],
        format=Connect4OutputModel.model_json_schema(),
    )
    return Connect4OutputModel.model_validate_json(response.message.content)

def visualize_board(board):
    # Visualize the Connect 4 board in a human-readable format
    game_print("\nCurrent Board:")
    for row in board:
        game_print(" | ".join(str(cell) for cell in row))
    game_print("-" * 29)
    game_print(" 0   1   2   3   4   5   6\n")

def play_connect4_with_llm(policy, num_simulations=50, num_matches=1, c_puct=1.5, temperature=1.0):
    # Play Connect 4 against an LLM agent using the provided policy network and MCTS parameters
    for match in range(num_matches):
        board = make_empty_board()
        to_play = 1
        policy_wins = 0
        llm_wins = 0
        draws = 0
        prior_reasoning = ""
        prior_action = None
        system_prompt = "You are an expert Connect 4 player. You will be given the current board state and your turn to play. Please choose a column (0-6) to drop your piece and provide reasoning for your choice. Your future moves will be influenced by your previous reasoning and actions. Output your response in JSON format with 'action' and 'reasoning' fields."
        game_print(f"Starting match {match + 1}")
        while True:
            if to_play == 1:
                # Policy network's turn
                col, pi = mcts_choose_action(board, to_play, policy, num_simulations, c_puct, temperature)
                board = drop_piece(board, col, to_play)
                game_print(f"Policy network plays column {col}")
                visualize_board(board)
            else:
                # LLM agent's turn
                prompt = f"{system_prompt}\nCurrent board state:\n{board}\nYour previous reasoning: {prior_reasoning}\nYour previous action: {prior_action}\nIt's your turn to play. Please provide your move and reasoning."
                prompt_logger.info("Prompt sent to LLM:\n%s", prompt)
                response = ollama_call(prompt)
                prompt_logger.info(
                    "Response received from LLM:\n%s",
                    response.model_dump_json(),
                )
                action = response.action
                reasoning = response.reasoning
                if action not in valid_moves(board):
                    game_print(f"LLM agent attempted an invalid move: {action}. Please choose a valid column (0-6).")
                    continue
                board = drop_piece(board, action, to_play)
                game_print(f"LLM agent plays column {action} with reasoning: {reasoning}")
                visualize_board(board)
                prior_reasoning = reasoning
                prior_action = action
            to_play = other_player(to_play)
            # Check winning conditions and display info
            done, winner = is_terminal(board)
            if done:
                if winner == 0:
                    game_print("The game is a draw.")
                    draws += 1
                elif winner == 1:
                    game_print("Policy network wins!")
                    policy_wins += 1
                else:
                    game_print("LLM agent wins!")
                    llm_wins += 1
                break
    game_print("Final Results: LLM agent wins:", llm_wins, "Policy network wins:", policy_wins, "Draws:", draws)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy network against an LLM agent in Connect 4.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved policy-value network state dict.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Hidden-channel count used when the network was trained.")
    parser.add_argument("--matches", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--num_simulations", type=int, default=8, help="Number of MCTS simulations per network move.")
    parser.add_argument("--c_puct", type=float, default=1.5, help="MCTS exploration constant.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature used to convert MCTS visit counts into move probabilities.")
    parser.add_argument("--log_dir", default="logs", help="Directory for prompts.log and game.log.")
    args = parser.parse_args()

    setup_logging(args.log_dir)
    checkpoint_dir, checkpoint_filename = os.path.split(args.checkpoint)
    net = load_network(filename=checkpoint_filename, load_dir=checkpoint_dir or ".", hidden_channels=args.hidden_channels)
    # test_call()
    play_connect4_with_llm(net, num_simulations=args.num_simulations, num_matches=args.matches, c_puct=args.c_puct, temperature=args.temperature)
