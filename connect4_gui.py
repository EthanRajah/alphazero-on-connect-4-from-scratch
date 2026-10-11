"""Tkinter interface for playing Connect-4 against the trained policy network."""

import tkinter as tk
from tkinter import messagebox

import numpy as np
import torch

from model import (
    action_mask,
    encode_board,
    greedy_agent_action,
    is_terminal,
    make_empty_board,
    masked_policy_logits,
    step_env,
    valid_moves,
)


class Connect4GUI:
    """Interactive, animated human-vs-network Connect-4 game."""

    ROWS = 6
    COLUMNS = 7
    CELL_SIZE = 84
    BOARD_PADDING = 24
    PIECE_RADIUS = 30
    COLORS = {0: "#f5f7fb", 1: "#ff5c5c", 2: "#ffd166"}

    def __init__(self, net, num_matches=1):
        self.net = net
        self.num_matches = max(1, int(num_matches))
        self.match_number = 0
        self.results = {"wins": 0, "losses": 0, "draws": 0}
        self.board = make_empty_board()
        self.to_play = 1
        self.game_over = False
        self.animating = False
        self.ai_column = None
        self._animation_job = None

        self.root = tk.Tk()
        self.root.title("Connect 4 - AlphaZero")
        self.root.resizable(False, False)
        self.root.configure(bg="#101827")

        width = self.COLUMNS * self.CELL_SIZE + 2 * self.BOARD_PADDING
        height = self.ROWS * self.CELL_SIZE + 2 * self.BOARD_PADDING
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            bg="#1769aa",
            highlightthickness=0,
        )
        self.canvas.pack(padx=18, pady=(18, 8))
        self.canvas.bind("<Button-1>", self._on_click)
        self.root.bind("<Key>", self._on_key)

        controls = tk.Frame(self.root, bg="#101827")
        controls.pack(fill="x", padx=18, pady=(0, 8))
        self.status = tk.Label(
            controls,
            text="",
            fg="#f8fafc",
            bg="#101827",
            font=("Helvetica", 14, "bold"),
            anchor="w",
        )
        self.status.pack(fill="x")
        self.policy_label = tk.Label(
            controls,
            text="",
            fg="#b8c5d6",
            bg="#101827",
            font=("Helvetica", 10),
            anchor="w",
        )
        self.policy_label.pack(fill="x", pady=(3, 0))

        button_row = tk.Frame(self.root, bg="#101827")
        button_row.pack(fill="x", padx=18, pady=(0, 18))
        tk.Button(
            button_row,
            text="New game",
            command=self.start_game,
            bg="#2d8cff",
            fg="white",
            activebackground="#57a3ff",
            relief="flat",
            padx=12,
        ).pack(side="left")
        tk.Label(
            button_row,
            text="You: red   |   Network: yellow   |   Click a column or press 1-7",
            fg="#b8c5d6",
            bg="#101827",
            font=("Helvetica", 10),
        ).pack(side="left", padx=12)

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.start_game()

    def run(self):
        """Start the Tk event loop and return completed match results."""
        self.root.mainloop()
        return self.results

    def _column_x(self, column):
        return self.BOARD_PADDING + column * self.CELL_SIZE + self.CELL_SIZE / 2

    def _row_y(self, row):
        return self.BOARD_PADDING + row * self.CELL_SIZE + self.CELL_SIZE / 2

    def _draw_board(self):
        self.canvas.delete("all")
        for row in range(self.ROWS):
            for column in range(self.COLUMNS):
                self.canvas.create_oval(
                    self._column_x(column) - self.PIECE_RADIUS,
                    self._row_y(row) - self.PIECE_RADIUS,
                    self._column_x(column) + self.PIECE_RADIUS,
                    self._row_y(row) + self.PIECE_RADIUS,
                    fill=self.COLORS[int(self.board[row, column])],
                    outline="#0d4f82",
                    width=2,
                )
        if self.ai_column is not None and not self.game_over:
            x = self._column_x(self.ai_column)
            self.canvas.create_rectangle(
                x - self.CELL_SIZE / 2 + 5,
                self.BOARD_PADDING - 8,
                x + self.CELL_SIZE / 2 - 5,
                self.BOARD_PADDING + 2,
                fill="#ffd166",
                outline="",
                tags="ai-choice",
            )

    def _set_status(self, text):
        self.status.config(text=text)

    def start_game(self):
        if self._animation_job is not None:
            self.root.after_cancel(self._animation_job)
            self._animation_job = None
        self.match_number += 1
        self.board = make_empty_board()
        self.to_play = 1
        self.game_over = False
        self.animating = False
        self.ai_column = None
        self.policy_label.config(text="")
        self._draw_board()
        self._set_status(f"Game {self.match_number}/{self.num_matches}: your turn")

    def _on_click(self, event):
        if self.to_play != 1 or self.animating or self.game_over:
            return
        column = int((event.x - self.BOARD_PADDING) // self.CELL_SIZE)
        self._play_human_column(column)

    def _on_key(self, event):
        if event.keysym.lower() == "r":
            self.start_game()
            return
        if event.char in "1234567" and self.to_play == 1:
            self._play_human_column(int(event.char) - 1)

    def _play_human_column(self, column):
        if not 0 <= column < self.COLUMNS or column not in valid_moves(self.board):
            self._set_status("That column is full. Choose another column.")
            return
        self._animate_move(column, 1, self._after_human_move)

    def _after_human_move(self):
        done, winner = is_terminal(self.board)
        if self._finish_if_terminal(done, winner):
            return
        self.to_play = 2
        self._set_status("Network is thinking...")
        self._animation_job = self.root.after(300, self._play_network_move)

    def _play_network_move(self):
        self._animation_job = None
        if self.game_over:
            return
        encoded = torch.from_numpy(encode_board(self.board, 2)).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(encoded)
        probabilities = torch.softmax(
            masked_policy_logits(logits, action_mask(self.board)), dim=-1
        )[0].numpy()
        column = greedy_agent_action(self.net, self.board, 2)
        self.ai_column = column
        policy_text = "  ".join(
            f"{index + 1}: {probability:.0%}" for index, probability in enumerate(probabilities)
        )
        self.policy_label.config(text=f"Network policy: {policy_text}")
        self._draw_board()
        self._set_status(f"Network chooses column {column + 1}")
        self._animation_job = self.root.after(
            550, lambda: self._animate_move(column, 2, self._after_network_move)
        )

    def _after_network_move(self):
        self.ai_column = None
        done, winner = is_terminal(self.board)
        if self._finish_if_terminal(done, winner):
            return
        self.to_play = 1
        self._set_status("Your turn")

    def _animate_move(self, column, player, callback):
        row = int(np.where(self.board[:, column] == 0)[0][-1])
        target_y = self._row_y(row)
        x = self._column_x(column)
        current_y = self.BOARD_PADDING - self.PIECE_RADIUS
        self.animating = True
        self._draw_board()
        piece = self.canvas.create_oval(
            x - self.PIECE_RADIUS,
            current_y - self.PIECE_RADIUS,
            x + self.PIECE_RADIUS,
            current_y + self.PIECE_RADIUS,
            fill=self.COLORS[player],
            outline="#0d4f82",
            width=2,
        )

        def advance():
            nonlocal current_y
            current_y = min(target_y, current_y + 22)
            self.canvas.coords(
                piece,
                x - self.PIECE_RADIUS,
                current_y - self.PIECE_RADIUS,
                x + self.PIECE_RADIUS,
                current_y + self.PIECE_RADIUS,
            )
            if current_y < target_y:
                self._animation_job = self.root.after(16, advance)
                return
            self._animation_job = None
            self.canvas.delete(piece)
            self.board, _, _, _ = step_env(self.board, column, player)
            self.animating = False
            self._draw_board()
            callback()

        advance()

    def _finish_if_terminal(self, done, winner):
        if not done:
            return False
        self.game_over = True
        if winner == 1:
            self.results["wins"] += 1
            message = "You win!"
        elif winner == 2:
            self.results["losses"] += 1
            message = "The network wins."
        else:
            self.results["draws"] += 1
            message = "Draw game."
        self._set_status(
            f"{message}  Series: {self.results['wins']}W / "
            f"{self.results['losses']}L / {self.results['draws']}D"
        )
        if self.match_number < self.num_matches:
            self._animation_job = self.root.after(900, self.start_game)
        else:
            self.root.after(
                250,
                lambda: messagebox.showinfo(
                    "Match complete",
                    f"{message}\n\nSeries result: {self.results['wins']}W / "
                    f"{self.results['losses']}L / {self.results['draws']}D",
                ),
            )
        return True


def play_against_network(net, num_matches=1):
    """Open the graphical human-vs-network evaluator."""
    return Connect4GUI(net, num_matches=num_matches).run()
