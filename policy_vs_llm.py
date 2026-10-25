"""Assess the performance of a policy network against a Monte Carlo Tree Search (MCTS) agent in Connect 4 against an LLM agent hosted on Ollama"""

import argparse
from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
from train import load_network
from model import *

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

def Connect4OutputModel(BaseModel):
    action: int
    reasoning: str

def ollama_call(prompt):
    response = chat(model='qwen3.5:9b',messages=[{'role': 'user','content': prompt}], format=Connect4OutputModel.model_json_schema())
    return response

def visualize_board(board):
    # Visualize the Connect 4 board in a human-readable format
    print("\nCurrent Board:")
    for row in board:
        print(" | ".join(str(cell) for cell in row))
    print("-" * 29)
    print(" 0   1   2   3   4   5   6\n")

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
        print(f"Starting match {match + 1}")
        while True:
            if to_play == 1:
                # Policy network's turn
                col, pi = mcts_choose_action(board, to_play, policy, num_simulations, c_puct, temperature)
                board = drop_piece(board, col, to_play)
                print(f"Policy network plays column {col}")
                visualize_board(board)
            else:
                # LLM agent's turn
                prompt = f"{system_prompt}\nCurrent board state:\n{board}\nYour previous reasoning: {prior_reasoning}\nYour previous action: {prior_action}\nIt's your turn to play. Please provide your move and reasoning."
                response = ollama_call(prompt)
                action = response.action
                reasoning = response.reasoning
                if action not in valid_moves(board):
                    print(f"LLM agent attempted an invalid move: {action}. Please choose a valid column (0-6).")
                    continue
                board = drop_piece(board, action, to_play)
                print(f"LLM agent plays column {action} with reasoning: {reasoning}")
                visualize_board(board)
                prior_reasoning = reasoning
                prior_action = action
            winner = check_winner(board)
            if winner is not None:
                if winner == 0:
                    print("The game is a draw.")
                    draws += 1
                elif winner == 1:
                    print("Policy network wins!")
                    policy_wins += 1
                else:
                    print("LLM agent wins!")
                    llm_wins += 1
                break
    print("Final Results: LLM agent wins:", llm_wins, "Policy network wins:", policy_wins, "Draws:", draws)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy network against an LLM agent in Connect 4.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved policy-value network state dict.")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Hidden-channel count used when the network was trained.")
    parser.add_argument("--matches", type=int, default=1, help="Number of games to play.")
    parser.add_argument("--num_simulations", type=int, default=8, help="Number of MCTS simulations per network move.")
    parser.add_argument("--c_puct", type=float, default=1.5, help="MCTS exploration constant.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature used to convert MCTS visit counts into move probabilities.")
    args = parser.parse_args()

    net = load_network(args.checkpoint, args.hidden_channels)
    # test_call()
    play_connect4_with_llm(net, num_simulations=args.num_simulations, num_matches=args.matches, c_puct=args.c_puct, temperature=args.temperature)
