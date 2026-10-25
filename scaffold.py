"""
AlphaZero on Connect-4 from Scratch

Scaffold for testing and debugging.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in solution functions)

"""AlphaZero on Connect-4: end-to-end demo of self-play training and evaluation."""

import numpy as np
import torch

def _debug_root_search(state, to_play, net, num_simulations, c_puct, temperature=1.0):
    """Run MCTS and print root child statistics from the root player's view."""
    root = run_mcts(state, to_play, net, num_simulations, c_puct)
    policy = visit_count_policy(root, temperature)
    action = int(np.argmax(policy))

    print("Root search statistics:")
    print("column | prior | visits | child Q | root Q | PUCT")
    for column, child in root["children"].items():
        child_q = node_q_value(child)
        root_q = -child_q
        puct = ucb_score(root, child, c_puct)
        print(
            f"{column:>6} | {child['prior']:.6f} | {child['visit_count']:>6} "
            f"| {child_q:>7.4f} | {root_q:>6.4f} | {puct:>7.4f}"
        )
    print(f"Root visits: {root['visit_count']}")
    return action, policy

def _summarize_loss(v):
    """Convert a loss value (scalar or list of scalars) to a single rounded float."""
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return float("nan")
        return round(float(np.mean([float(x) for x in v])), 4)
    return round(float(v), 4)

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Sanity-check the board engine ---
    board = make_empty_board()
    print("Empty board shape:", board.shape, "dtype:", board.dtype)
    board = drop_piece(board, 3, 1)
    board = drop_piece(board, 3, 2)
    print("Legal moves after two drops:", valid_moves(board))
    print("Terminal?", is_terminal(board))

    # --- Build a tiny policy-value network ---
    net = build_policy_value_net(in_channels=2, hidden_channels=8, num_columns=7)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Policy-value net params: {n_params}")

    # --- One forward pass on a fresh board ---
    enc = board_to_torch_tensor(make_empty_board(), current_player=1)
    with torch.no_grad():
        logits, value = policy_value_forward(net, enc)
    print("Policy logits shape:", tuple(logits.shape), "value shape:", tuple(value.shape))

    # --- One MCTS rollout from the empty board ---
    action, pi = mcts_choose_action(
        make_empty_board(), to_play=1, net=net,
        num_simulations=8, c_puct=1.5, temperature=1.0,
    )
    print(f"MCTS picked column {action}; pi = {np.round(pi, 3).tolist()}")

    # --- Generate a small self-play buffer ---
    buffer = generate_self_play_batch(
        net, num_games=2, num_simulations=6, c_puct=1.5, temperature=1.0,
    )
    print(f"Self-play buffer size: {len(buffer)} steps")

    # --- A couple of training steps on that buffer ---
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses_before = training_epoch(
        net, optimizer, buffer, batch_size=8,
        policy_weight=1.0, value_weight=1.0, l2_weight=1e-4, seed=0,
    )
    print("Epoch 1 losses:", {k: _summarize_loss(v) for k, v in losses_before.items()})

    # --- One AlphaZero outer iteration (self-play + training) ---
    iter_result = self_play_iteration(
        net, optimizer,
        num_games=2, num_simulations=6, c_puct=1.5,
        batch_size=8, num_epochs=1, temperature=1.0,
    )
    # self_play_iteration returns {'buffer_size': int, 'losses': [epoch_dict, ...]}
    if isinstance(iter_result, dict) and 'losses' in iter_result and isinstance(iter_result['losses'], list):
        iter_losses = iter_result['losses'][-1] if iter_result['losses'] else {}
    else:
        iter_losses = iter_result
    print("Outer-iter losses:", {k: _summarize_loss(v) for k, v in iter_losses.items()})

    # --- Evaluate the greedy net agent against a random baseline ---
    results = evaluate_against_random(net, num_matches=6, seed=0)
    print("Greedy-net vs random:", results)

    # --- Terminal board check to ensure policy is also blocking winning moves ---
    loaded_net = build_policy_value_net(in_channels=2, hidden_channels=16, num_columns=7)
    loaded_net.load_state_dict(torch.load("training/trained_connect4_net_16h_8b_0.0001lr_200s_1.5c_1.0t_25g_50i_10e/trained_connect4_net.pth", map_location="cpu", weights_only=True))
    loaded_net.eval()
    # Create a board where player 1 is about to win and check if the policy blocks
    board = make_empty_board()
    board = drop_piece(board, 0, 1)
    board = drop_piece(board, 1, 2)
    board = drop_piece(board, 0, 1)
    board = drop_piece(board, 2, 2)
    board = drop_piece(board, 0, 1)
    action, pi = _debug_root_search(
        board,
        to_play=2,
        net=loaded_net,
        num_simulations=50,
        c_puct=1.5,
        temperature=1.0,
    )
    print(f"Policy should block column 0; MCTS picked column {action}; pi = {np.round(pi, 3).tolist()}")
    raw_action = np.argmax(evaluate_with_network(loaded_net, board, to_play=2)[0])
    print(f"Direct network evaluation picked column {raw_action}")