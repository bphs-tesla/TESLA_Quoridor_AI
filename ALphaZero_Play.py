import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import mixed_precision

# ============================================================
# 설정
# ============================================================

BOARD_SIZE = 9
ACTION_SIZE = 132

NUM_RES_BLOCKS = 6
NUM_FILTERS = 64

MCTS_SIMULATIONS = 800
MCTS_BATCH_SIZE = 32
MCTS_C_PUCT = 1.5

CHECKPOINT_PATH = "AlphaZero_weights/alphazero.weights.h5"

mixed_precision.set_global_policy("float32")

# ============================================================
# Quoridor
# ============================================================

sys.path.insert(0, "./build-linux")
sys.path.append("../TESLA_Quoridor_AI/build/Debug")
sys.path.append("../TESLA_Quoridor_AI")

import Quoridor

# ============================================================
# GPU
# ============================================================

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"GPU detected: {len(gpus)}")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )
        except RuntimeError as e:
            print(e)

    GPU_DEVICE = "/GPU:0"

else:
    print("WARNING: GPU를 찾지 못했습니다.")
    GPU_DEVICE = "/CPU:0"


# ============================================================
# State Encoding
#
# 학습할 때 사용한 get_state()와 반드시 동일해야 함
# ============================================================

def get_state(board, player):

    state = np.zeros(
        (BOARD_SIZE, BOARD_SIZE, 7),
        dtype=np.float32
    )

    if player == 1:

        my_pos = board.P1
        opp_pos = board.P2

        my_walls = int(board.WallCount[0])
        opp_walls = int(board.WallCount[1])

        my_turn = bool(board.isP1Turn)

    else:

        my_pos = board.P2
        opp_pos = board.P1

        my_walls = int(board.WallCount[1])
        opp_walls = int(board.WallCount[0])

        my_turn = not bool(board.isP1Turn)

    # --------------------------------------------------------
    # Channel 0 : 내 위치
    # --------------------------------------------------------

    state[
        int(my_pos[0]),
        int(my_pos[1]),
        0
    ] = 1.0

    # --------------------------------------------------------
    # Channel 1 : 상대 위치
    # --------------------------------------------------------

    state[
        int(opp_pos[0]),
        int(opp_pos[1]),
        1
    ] = 1.0

    # --------------------------------------------------------
    # Channel 2 : Horizontal Wall
    # --------------------------------------------------------

    for x in range(8):

        value = int(board.WallH[x])

        for y in range(8):

            if value & (1 << y):

                state[
                    x,
                    y,
                    2
                ] = 1.0

    # --------------------------------------------------------
    # Channel 3 : Vertical Wall
    # --------------------------------------------------------

    for y in range(8):

        value = int(board.WallVt[y])

        for x in range(8):

            if value & (1 << x):

                state[
                    y,
                    x,
                    3
                ] = 1.0

    # --------------------------------------------------------
    # Channel 4 : Turn
    # --------------------------------------------------------

    state[:, :, 4] = float(my_turn)

    # --------------------------------------------------------
    # Channel 5 : My Wall Count
    # --------------------------------------------------------

    state[:, :, 5] = my_walls / 10.0

    # --------------------------------------------------------
    # Channel 6 : Opponent Wall Count
    # --------------------------------------------------------

    state[:, :, 6] = opp_walls / 10.0

    return state


# ============================================================
# Action
# ============================================================

def execute_action(board, action):

    action = int(action)

    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(
            f"Invalid action: {action}"
        )

    # --------------------------------------------------------
    # 0 ~ 3 : 이동
    # --------------------------------------------------------

    if action < 4:

        return int(
            Quoridor.MovePlayer(
                board,
                action
            )
        )

    # --------------------------------------------------------
    # 4 ~ 67 : Horizontal Wall
    # --------------------------------------------------------

    elif action < 68:

        wall_index = action - 4

        x = wall_index // 8
        y = wall_index % 8

        return int(
            Quoridor.PlaceWallH(
                board,
                [x, y]
            )
        )

    # --------------------------------------------------------
    # 68 ~ 131 : Vertical Wall
    # --------------------------------------------------------

    else:

        wall_index = action - 68

        x = wall_index // 8
        y = wall_index % 8

        return int(
            Quoridor.PlaceWallV(
                board,
                [x, y]
            )
        )


# ============================================================
# Terminal
# ============================================================

def get_terminal_value(board, player):

    winner = int(
        Quoridor.isGameOver(board)
    )

    if winner == 0:
        return None

    if winner == player:
        return 1.0

    return -1.0


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(keras.layers.Layer):

    def __init__(self, filters):

        super().__init__()

        self.conv1 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False
        )

        self.bn1 = keras.layers.BatchNormalization()

        self.relu1 = keras.layers.LeakyReLU(
            negative_slope=0.1
        )

        self.conv2 = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False
        )

        self.bn2 = keras.layers.BatchNormalization()

        self.relu2 = keras.layers.LeakyReLU(
            negative_slope=0.1
        )

    def call(self, x, training=False):

        shortcut = x

        y = self.conv1(x)

        y = self.bn1(
            y,
            training=training
        )

        y = self.relu1(y)

        y = self.conv2(y)

        y = self.bn2(
            y,
            training=training
        )

        y = y + shortcut

        y = self.relu2(y)

        return y


# ============================================================
# AlphaZero Network
# ============================================================

class AlphaZeroNetwork(keras.Model):

    def __init__(
        self,
        num_blocks=NUM_RES_BLOCKS,
        filters=NUM_FILTERS
    ):

        super().__init__()

        self.input_conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False
        )

        self.input_bn = keras.layers.BatchNormalization()

        self.input_relu = keras.layers.LeakyReLU(
            negative_slope=0.1
        )

        self.blocks = [
            ResidualBlock(filters)
            for _ in range(num_blocks)
        ]

        # Policy

        self.policy_pool = (
            keras.layers.GlobalAveragePooling2D()
        )

        self.policy_dense = keras.layers.Dense(
            256
        )

        self.policy_relu = keras.layers.LeakyReLU(
            negative_slope=0.1
        )

        self.policy_head = keras.layers.Dense(
            ACTION_SIZE,
            dtype="float32"
        )

        # Value

        self.value_pool = (
            keras.layers.GlobalAveragePooling2D()
        )

        self.value_dense = keras.layers.Dense(
            256
        )

        self.value_relu = keras.layers.LeakyReLU(
            negative_slope=0.1
        )

        self.value_head = keras.layers.Dense(
            1,
            activation="tanh",
            dtype="float32"
        )

    def call(
        self,
        inputs,
        training=False
    ):

        x = self.input_conv(inputs)

        x = self.input_bn(
            x,
            training=training
        )

        x = self.input_relu(x)

        for block in self.blocks:

            x = block(
                x,
                training=training
            )

        # Policy

        p = self.policy_pool(x)
        p = self.policy_dense(p)
        p = self.policy_relu(p)

        policy_logits = self.policy_head(p)

        # Value

        v = self.value_pool(x)
        v = self.value_dense(v)
        v = self.value_relu(v)

        value = self.value_head(v)

        return policy_logits, value


# ============================================================
# Model Load
# ============================================================

with tf.device(GPU_DEVICE):

    model = AlphaZeroNetwork(
        num_blocks=NUM_RES_BLOCKS,
        filters=NUM_FILTERS
    )

    dummy_input = tf.zeros(
        (1, 9, 9, 7),
        dtype=tf.float32
    )

    model(
        dummy_input,
        training=False
    )

    print()
    print("Loading AlphaZero weights...")
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    model.load_weights(
        CHECKPOINT_PATH
    )

    print("Weights loaded successfully.")


# ============================================================
# Network Evaluation
# ============================================================

def evaluate(model, board, player):

    state = get_state(
        board,
        player
    )

    state = np.expand_dims(
        state,
        axis=0
    )

    with tf.device(GPU_DEVICE):

        policy_logits, value = model(
            state,
            training=False
        )

    policy_logits = (
        policy_logits
        .numpy()[0]
    )

    value = float(
        value.numpy()[0, 0]
    )

    return policy_logits, value


# ============================================================
# MCTS
# ============================================================

class MCTSNode:

    def __init__(
        self,
        board,
        player,
        parent=None,
        action=None,
        prior=0.0
    ):

        self.board = board
        self.player = player

        self.parent = parent
        self.action = action

        self.prior = float(prior)

        self.visit_count = 0
        self.value_sum = 0.0

        self.children = {}

        self.expanded = False

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum /
            self.visit_count
        )


class MCTS:

    def __init__(
        self,
        model,
        simulations=MCTS_SIMULATIONS,
        c_puct=MCTS_C_PUCT
    ):

        self.model = model

        self.simulations = int(
            simulations
        )

        self.c_puct = float(
            c_puct
        )

    # --------------------------------------------------------
    # Valid actions
    # --------------------------------------------------------

    def get_valid_actions(self, node):

        return [
            int(a)
            for a in Quoridor.GetValidActions(
                node.board
            )
        ]

    # --------------------------------------------------------
    # Expand
    # --------------------------------------------------------

    def expand(
        self,
        node,
        policy
    ):

        if node.expanded:
            return

        valid_actions = (
            self.get_valid_actions(node)
        )

        next_player = (
            2
            if node.player == 1
            else 1
        )

        # Mask + softmax

        valid_logits = policy[
            valid_actions
        ]

        max_logit = np.max(
            valid_logits
        )

        exp_logits = np.exp(
            valid_logits - max_logit
        )

        probs = (
            exp_logits /
            np.sum(exp_logits)
        )

        for action, prior in zip(
            valid_actions,
            probs
        ):

            child_board = (
                Quoridor.CloneBoard(
                    node.board
                )
            )

            result = execute_action(
                child_board,
                action
            )

            if result != 0:
                continue

            child = MCTSNode(
                board=child_board,
                player=next_player,
                parent=node,
                action=action,
                prior=float(prior)
            )

            node.children[
                action
            ] = child

        node.expanded = True

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    def select_child(self, node):

        best_child = None
        best_score = -np.inf

        sqrt_visits = np.sqrt(
            node.visit_count + 1
        )

        for child in node.children.values():

            q = -child.value

            u = (
                self.c_puct
                * child.prior
                * sqrt_visits
                /
                (1 + child.visit_count)
            )

            score = q + u

            if score > best_score:

                best_score = score
                best_child = child

        return best_child

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    def simulate(self, root):

        node = root

        path = [node]

        # Selection

        while node.expanded:

            terminal = get_terminal_value(
                node.board,
                node.player
            )

            if terminal is not None:

                value = terminal

                break

            if not node.children:

                value = 0.0

                break

            node = self.select_child(
                node
            )

            path.append(node)

        else:

            terminal = get_terminal_value(
                node.board,
                node.player
            )

            if terminal is not None:

                value = terminal

            else:

                policy, value = evaluate(
                    self.model,
                    node.board,
                    node.player
                )

                self.expand(
                    node,
                    policy
                )

        # Backup

        for current in reversed(path):

            current.visit_count += 1

            current.value_sum += value

            value = -value

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        board,
        player
    ):

        root = MCTSNode(
            board=Quoridor.CloneBoard(
                board
            ),
            player=player
        )

        policy, value = evaluate(
            self.model,
            root.board,
            root.player
        )

        self.expand(
            root,
            policy
        )

        for _ in range(
            self.simulations
        ):

            self.simulate(
                root
            )

        # 가장 많이 방문한 수

        if not root.children:
            raise RuntimeError(
                "MCTS found no valid actions."
            )

        best_child = max(
            root.children.values(),
            key=lambda child:
                child.visit_count
        )

        return (
            best_child.action,
            root
        )


# ============================================================
# Human Input
# ============================================================

def print_actions():

    print()
    print("입력 방법")
    print("----------------------------")
    print("w              : 위")
    print("s              : 아래")
    print("a              : 왼쪽")
    print("d              : 오른쪽")
    print("h x y          : 가로 벽")
    print("v x y          : 세로 벽")
    print("q              : 종료")
    print("----------------------------")
    print()


def human_action(board):

    while True:

        print_actions()

        command = input(
            "당신의 수 > "
        ).strip().lower()

        if command == "q":
            raise KeyboardInterrupt

        if command == "w":
            action = 0

        elif command == "s":
            action = 1

        elif command == "a":
            action = 2

        elif command == "d":
            action = 3

        else:

            parts = command.split()
            if (
                len(parts) == 3
                and parts[0] in ("h", "v")
            ):

                try:

                    x = int(parts[1])
                    y = int(parts[2])

                except ValueError:

                    print(
                        "좌표는 숫자로 입력하세요."
                    )

                    continue

                if not (
                    0 <= x < 8
                    and 0 <= y < 8
                ):

                    print(
                        "벽 좌표는 0~7입니다."
                    )

                    continue

                if parts[0] == "h":

                    action = (
                        4
                        + x * 8
                        + y
                    )

                else:

                    action = (
                        68
                        + x * 8
                        + y
                    )

            else:

                print(
                    "잘못된 입력입니다."
                )

                continue

        valid_actions = [
            int(a)
            for a in Quoridor.GetValidActions(
                board
            )
        ]

        if action not in valid_actions:

            print()
            print(
                "❌ 현재 상태에서는 "
                "그 수를 둘 수 없습니다."
            )

            continue

        return action


# ============================================================
# AI Move
# ============================================================

def ai_action(
    board,
    player,
    mcts
):

    start = time.time()

    action, root = mcts.search(
        board,
        player
    )

    elapsed = time.time() - start

    child = root.children[
        action
    ]

    print()
    print("================================")
    print("AI")
    print("================================")

    print(
        f"Action       : {action}"
    )

    print(
        f"Visit count  : "
        f"{child.visit_count}"
    )

    print(
        f"Value        : "
        f"{child.value:.4f}"
    )

    print(
        f"MCTS time    : "
        f"{elapsed:.3f}s"
    )

    print(
        "================================"
    )

    return action


# ============================================================
# Main
# ============================================================

def main():

    board = Quoridor.Quoridor()

    # --------------------------------------------------------
    # 사람이 P1인지 P2인지
    # --------------------------------------------------------

    while True:

        choice = input(
            "선택하세요 (1 = P1 / 2 = P2): "
        ).strip()

        if choice in ("1", "2"):
            human_player = int(choice)
            break

        print(
            "1 또는 2를 입력하세요."
        )

    ai_player = (
        2
        if human_player == 1
        else 1
    )

    mcts = MCTS(
        model,
        simulations=MCTS_SIMULATIONS,
        c_puct=MCTS_C_PUCT
    )

    print()
    print("========================================")
    print("       AlphaZero Quoridor")
    print("========================================")

    print(
        f"Human : P{human_player}"
    )

    print(
        f"AI    : P{ai_player}"
    )

    print(
        f"MCTS simulations : "
        f"{MCTS_SIMULATIONS}"
    )

    print(
        "========================================"
    )

    board.PrintBoard()

    move_count = 0

    while True:

        winner = int(
            Quoridor.isGameOver(
                board
            )
        )

        if winner != 0:

            print()
            print(
                "================================"
            )
            print(
                f"P{winner} 승리!"
            )
            print(
                "================================"
            )

            board.PrintBoard()

            break

        current_player = (
            1
            if bool(board.isP1Turn)
            else 2
        )

        print()
        print(
            f"현재 차례: P{current_player}"
        )

        # ----------------------------------------------------
        # Human
        # ----------------------------------------------------

        if current_player == human_player:

            action = human_action(
                board
            )

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        else:

            action = ai_action(
                board,
                ai_player,
                mcts
            )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        result = execute_action(
            board,
            action
        )

        if result != 0:

            print(
                f"ERROR: action {action} "
                f"failed ({result})"
            )

            continue

        move_count += 1

        print()
        print(
            f"Move {move_count}"
        )

        board.PrintBoard()


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "게임을 종료합니다."
        )