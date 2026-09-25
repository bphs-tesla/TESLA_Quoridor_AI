import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(
    '../TESLA_Quoridor_AI/build/Debug'
)

sys.path.append(
    '../TESLA_Quoridor_AI'
)

sys.path.append(os.path.join(os.path.dirname(__file__), 'build-linux'))

import Quoridor

from AlphaZero_MCTS import (
    NeuralNetwork,
    MCTS_Search
)


# ============================================================
# Board test
# ============================================================

board = Quoridor.Quoridor()

Quoridor.PlaceWallH(
    board,
    [4, 4]
)

Quoridor.PlaceWallH(
    board,
    [1, 0]
)

Quoridor.PlaceWallH(
    board,
    [0, 1]
)

Quoridor.PlaceWallV(
    board,
    [1, 1]
)

Quoridor.PlaceWallH(
    board,
    [1, 2]
)

Quoridor.PlaceWallH(
    board,
    [0, 4]
)

Quoridor.PlaceWallV(
    board,
    [0, 5]
)

Quoridor.PlaceWallH(
    board,
    [0, 6]
)

Quoridor.PlaceWallV(
    board,
    [1, 7]
)

Quoridor.PlaceWallH(
    board,
    [2, 7]
)

Quoridor.PlaceWallV(
    board,
    [1, 4]
)

Quoridor.PlaceWallV(
    board,
    [2, 3]
)

Quoridor.PlaceWallV(
    board,
    [3, 4]
)

board.P1 = [4, 4]
board.P2 = [2, 4]

board.PrintBoard()


# ============================================================
# GetStateBatch test
# ============================================================

boards = [
    Quoridor.Quoridor(),
    Quoridor.Quoridor(),
    Quoridor.Quoridor(),
    Quoridor.Quoridor()
]

players = [
    1,
    2,
    1,
    2
]

states = Quoridor.GetStateBatch(
    boards,
    players
)

print(
    "states.shape =",
    states.shape
)

print(
    "states.dtype =",
    states.dtype
)


# ============================================================
# State test
# ============================================================

states = np.asarray(
    states
)

print(
    "P1 state own position:",
    np.argwhere(
        states[0, :, :, 0] == 1
    )
)

print(
    "P2 state own position:",
    np.argwhere(
        states[1, :, :, 0] == 1
    )
)


# ============================================================
# Model
# ============================================================

model = tf.keras.models.load_model(
    "your_model_path"
)

network = NeuralNetwork(
    model
)


# ============================================================
# MCTS
# ============================================================

policy, root = MCTS_Search(
    board,
    network,
    simulations=800,
    batch_size=32
)


# ============================================================
# Result
# ============================================================

print(
    "Policy shape:",
    policy.shape
)

print(
    "Policy sum:",
    policy.sum()
)

print(
    "Root visit count:",
    root.visit_count
)

print(
    "Selected action:",
    np.argmax(policy)
)

print(
    "Selected probability:",
    np.max(policy)
)


# ============================================================
# Visit distribution
# ============================================================

for action in range(132):

    if policy[action] > 0:

        print(
            f"Action {action:3d}: "
            f"{policy[action]:.6f}"
        )