"""
AlphaZero on Connect-4 from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_empty_board
import numpy as np

def make_empty_board():
    """Return a 6x7 integer numpy array of zeros representing an empty Connect-4 board."""
    # TODO: create a 6x7 integer array of zeros and return it
    return np.zeros([6,7], dtype=int)

# Step 2 - column_top_row
def column_top_row(board, column):
    """Return the lowest empty row in `column`, or -1 if the column is full."""
    # TODO: scan the column from the bottom up and return the first empty row index
    empty_idxs = np.where(board[:,column] == 0)[0]
    if len(empty_idxs) == 0:
        return -1
    else:
        return empty_idxs[-1]

# Step 3 - drop_piece
def drop_piece(board, column, player):
    # TODO: place `player` in the lowest empty row of `column` and return the new board
    free_row = column_top_row(board, column)
    if free_row != -1:
        # Empty row in that column available
        new_board = board.copy()
        new_board[free_row][column] = player
        return new_board
    else:
        raise ValueError

# Step 4 - column_full
import numpy as np

def column_full(board, column):
    """Return True if `column` has no empty rows left."""
    # TODO: check whether the column can still accept a piece
    if column_top_row(board, column) == -1:
        return True
    else:
        return False

# Step 5 - valid_moves
def valid_moves(board):
    # TODO: return a list of column indices that still have at least one empty row
    valid_arr = []
    for col in range(len(board[0])):
        # Loop through the columns of the board
        if not column_full(board, col):
            valid_arr.append(col)
    return valid_arr

# Step 6 - four_in_a_row_horizontal
def four_in_a_row_horizontal(board):
    # TODO: scan every row for four consecutive matching non-zero pieces horizontally
    for row in board:
        # Iterate through the row looking for repetitions of 4 1s or 2s
        for cell_idx in range(len(row)):
            if ((cell_idx + 3) < len(row)):
                if len(set(row[cell_idx:cell_idx+4])) == 1:
                    if row[cell_idx] == 1:
                        return 1
                    elif row[cell_idx] == 2:
                        return 2
    return 0

# Step 7 - four_in_a_row_vertical
def four_in_a_row_vertical(board):
    # TODO: scan every column for four consecutive matching non-zero pieces vertically
    for col_idx in range (len(board[0])):
        col = board[:,col_idx]
        for cell_idx in range(len(col)):
            if ((cell_idx + 3) < len(col)):
                if len(set(col[cell_idx:cell_idx+4])) == 1:
                    if col[cell_idx] == 1:
                        return 1
                    elif col[cell_idx] == 2:
                        return 2
    return 0

# Step 8 - four_in_a_row_diagonal_down_right
def four_in_a_row_diagonal_down_right(board):
    # TODO: scan every down-right diagonal of the 6x7 board for four matching non-zero pieces
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # Valid diagonal range offset from main top left to bottom right is [-2, 3], where 0 is main
    for i in range(-2, 4):
        diag = board.diagonal(i)
        for j in range(len(diag)-3):
            if np.all(diag[j:j+4] == diag[j]) and diag[j] != 0:
                if diag[j] == 1:
                    return 1
                else:
                    return 2
    return 0

# Step 9 - four_in_a_row_diagonal_up_right
def four_in_a_row_diagonal_up_right(board):
    # TODO: scan every up-right diagonal for four consecutive matching non-zero pieces
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0
    # Valid diagonal range offset from main top left to bottom right is [-2, 3], where 0 is main
    flipped_board = np.flipud(board)
    for i in range(-2, 4):
        # flipud does a vertical flip
        diag = flipped_board.diagonal(i)
        for j in range(len(diag)-3):
            if np.all(diag[j:j+4] == diag[j]) and diag[j] != 0:
                if diag[j] == 1:
                    return 1
                else:
                    return 2
    return 0

# Step 10 - check_winner (not yet solved)
# TODO: implement

# Step 11 - board_is_full (not yet solved)
# TODO: implement

# Step 12 - is_terminal (not yet solved)
# TODO: implement

# Step 13 - other_player (not yet solved)
# TODO: implement

# Step 14 - step_env (not yet solved)
# TODO: implement

# Step 15 - encode_board (not yet solved)
# TODO: implement

# Step 16 - board_to_torch_tensor (not yet solved)
# TODO: implement

# Step 17 - init_conv_backbone (not yet solved)
# TODO: implement

# Step 18 - init_policy_head (not yet solved)
# TODO: implement

# Step 19 - init_value_head (not yet solved)
# TODO: implement

# Step 20 - build_policy_value_net (not yet solved)
# TODO: implement

# Step 21 - policy_value_forward (not yet solved)
# TODO: implement

# Step 22 - action_mask (not yet solved)
# TODO: implement

# Step 23 - masked_policy_logits (not yet solved)
# TODO: implement

# Step 24 - masked_log_softmax (not yet solved)
# TODO: implement

# Step 25 - sample_action_from_policy (not yet solved)
# TODO: implement

# Step 26 - greedy_action_from_policy (not yet solved)
# TODO: implement

# Step 27 - make_mcts_node (not yet solved)
# TODO: implement

# Step 28 - node_q_value (not yet solved)
# TODO: implement

# Step 29 - ucb_score (not yet solved)
# TODO: implement

# Step 30 - select_best_child (not yet solved)
# TODO: implement

# Step 31 - select_leaf (not yet solved)
# TODO: implement

# Step 32 - evaluate_with_network (not yet solved)
# TODO: implement

# Step 33 - expand_node (not yet solved)
# TODO: implement

# Step 34 - backup_value (not yet solved)
# TODO: implement

# Step 35 - run_one_simulation (not yet solved)
# TODO: implement

# Step 36 - run_mcts (not yet solved)
# TODO: implement

# Step 37 - visit_count_policy (not yet solved)
# TODO: implement

# Step 38 - mcts_choose_action (not yet solved)
# TODO: implement

# Step 39 - record_self_play_step (not yet solved)
# TODO: implement

# Step 40 - play_self_play_game (not yet solved)
# TODO: implement

# Step 41 - assign_value_targets (not yet solved)
# TODO: implement

# Step 42 - generate_self_play_batch (not yet solved)
# TODO: implement

# Step 43 - value_loss_mse (not yet solved)
# TODO: implement

# Step 44 - policy_loss_cross_entropy (not yet solved)
# TODO: implement

# Step 45 - l2_regularization_loss (not yet solved)
# TODO: implement

# Step 46 - combined_loss (not yet solved)
# TODO: implement

# Step 47 - encode_batch_states (not yet solved)
# TODO: implement

# Step 48 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 49 - training_step (not yet solved)
# TODO: implement

# Step 50 - training_epoch (not yet solved)
# TODO: implement

# Step 51 - self_play_iteration (not yet solved)
# TODO: implement

# Step 52 - train_loop (not yet solved)
# TODO: implement

# Step 53 - random_policy_action (not yet solved)
# TODO: implement

# Step 54 - greedy_agent_action (not yet solved)
# TODO: implement

# Step 55 - play_one_match (not yet solved)
# TODO: implement

# Step 56 - match_win_rate (not yet solved)
# TODO: implement

# Step 57 - evaluate_against_random (not yet solved)
# TODO: implement

