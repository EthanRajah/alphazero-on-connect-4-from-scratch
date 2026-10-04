"""
AlphaZero on Connect-4 from Scratch

Training script for long-running self-play training and hyperparameter tuning.

Run this with: python train.py
Uses functions defined in model.py.
"""

import argparse

from model import *
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=8, help="Number of channels in the hidden layers of the policy-value network.")
    parser.add_argument("--num_simulations", type=int, default=8, help="Number of MCTS simulations per move.")