import os

# ============================================================
# TensorFlow 설정
# ============================================================

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
import time
from datetime import datetime, timedelta

import sys
import random
import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow import keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam


# ============================================================
# 기본 설정
# ============================================================

BOARD_SIZE = 9
ACTION_SIZE = 132

NUM_RES_BLOCKS = 6
NUM_FILTERS = 64

LEARNING_RATE = 1e-4
L2_WEIGHT = 1e-5
GRAD_CLIP_NORM = 5.0

MCTS_SIMULATIONS = 32
MCTS_BATCH_SIZE = 32
MCTS_C_PUCT = 1.5

REPLAY_BUFFER_SIZE = 200000
BATCH_SIZE = 128

# 처음에는 반드시 작게 테스트
NUM_GAMES = 10

# Root Dirichlet noise
DIRICHLET_EPSILON = 0.25
DIRICHLET_ALPHA = 0.30

# float32부터 안정성 확인
mixed_precision.set_global_policy("float32")

# ============================================================
# AlphaZero Checkpoint
# ============================================================

CHECKPOINT_PATH = "AlphaZero_weights/alphazero.weights.h5"

# 기존 가중치가 있으면 불러오기
LOAD_CHECKPOINT = True

NUM_GAMES = 10000

# ============================================================
# Training Time Estimation
# ============================================================

training_start_time = time.time()

def format_time(seconds):
    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"
    elif minutes > 0:
        return f"{minutes}분 {seconds}초"
    else:
        return f"{seconds}초"

# ============================================================
# Training Time Estimation
# ============================================================

training_start_time = time.time()


def format_time(seconds):

    seconds = max(0, int(seconds))

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}시간 {minutes}분 {seconds}초"

    elif minutes > 0:
        return f"{minutes}분 {seconds}초"

    else:
        return f"{seconds}초"


def print_progress(
    current_game,
    total_games,
    start_time
):

    elapsed = time.time() - start_time

    if current_game <= 0:
        return

    # 평균 게임 시간
    avg_time = elapsed / current_game

    # 남은 게임
    remaining_games = total_games - current_game

    # 예상 남은 시간
    estimated_remaining = (
        avg_time * remaining_games
    )

    # 진행률
    progress = (
        current_game /
        total_games *
        100.0
    )

    # 예상 총 시간
    estimated_total = (
        avg_time *
        total_games
    )

    # 예상 완료 시각
    finish_time = (
        datetime.now()
        +
        timedelta(
            seconds=estimated_remaining
        )
    )

    print()
    print("=" * 60)
    print("Training Progress")
    print("=" * 60)

    print(
        f"진행        : "
        f"{current_game} / {total_games} "
        f"({progress:.1f}%)"
    )

    print(
        f"경과 시간    : "
        f"{format_time(elapsed)}"
    )

    print(
        f"평균/게임    : "
        f"{format_time(avg_time)}"
    )

    print(
        f"예상 남은 시간: "
        f"{format_time(estimated_remaining)}"
    )

    print(
        f"예상 총 시간  : "
        f"{format_time(estimated_total)}"
    )

    print(
        f"예상 완료 시각: "
        f"{finish_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    print("=" * 60)

# ============================================================
# Quoridor module
# ============================================================

sys.path.insert(0, "./build-linux")
sys.path.append("../TESLA_Quoridor_AI/build/Debug")
sys.path.append("../TESLA_Quoridor_AI")

import Quoridor


# ============================================================
# GPU 설정
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


print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("GPU:", gpus)
print("Build info:", tf.sysconfig.get_build_info())
print("Using device:", GPU_DEVICE)


# ============================================================
# State Encoding
# ============================================================

def get_state(board, player):

    state = np.zeros(
        (BOARD_SIZE, BOARD_SIZE, 7),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Player-relative representation
    # --------------------------------------------------------

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
    # Player positions
    # --------------------------------------------------------

    state[
        int(my_pos[0]),
        int(my_pos[1]),
        0
    ] = 1.0

    state[
        int(opp_pos[0]),
        int(opp_pos[1]),
        1
    ] = 1.0


    # --------------------------------------------------------
    # Horizontal walls
    #
    # C++:
    # WallH[x] & (1 << y)
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
    # Vertical walls
    #
    # C++:
    # WallVt[y] & (1 << x)
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
    # Turn
    # --------------------------------------------------------

    state[:, :, 4] = float(my_turn)


    # --------------------------------------------------------
    # Wall count
    # --------------------------------------------------------

    state[:, :, 5] = my_walls / 10.0
    state[:, :, 6] = opp_walls / 10.0


    return state


# ============================================================
# Action execution
# ============================================================

def execute_action(board, action):

    action = int(action)

    if action < 0 or action >= ACTION_SIZE:

        raise ValueError(
            f"Invalid action: {action}"
        )


    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    if action < 4:

        result = Quoridor.MovePlayer(
            board,
            action
        )


    # --------------------------------------------------------
    # Horizontal wall
    # --------------------------------------------------------

    elif action < 68:

        wall_index = action - 4

        x = wall_index // 8
        y = wall_index % 8

        result = Quoridor.PlaceWallH(
            board,
            [x, y]
        )


    # --------------------------------------------------------
    # Vertical wall
    # --------------------------------------------------------

    else:

        wall_index = action - 68

        x = wall_index // 8
        y = wall_index % 8

        result = Quoridor.PlaceWallV(
            board,
            [x, y]
        )


    return int(result)


# ============================================================
# Terminal value
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
# AlphaZero Residual Network
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


        # ====================================================
        # Policy head
        # ====================================================

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


        # ====================================================
        # Value head
        # ====================================================

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


        # ====================================================
        # Policy
        # ====================================================

        p = self.policy_pool(x)

        p = self.policy_dense(p)

        p = self.policy_relu(p)

        policy_logits = self.policy_head(p)


        # ====================================================
        # Value
        # ====================================================

        v = self.value_pool(x)

        v = self.value_dense(v)

        v = self.value_relu(v)

        value = self.value_head(v)


        return policy_logits, value


# ============================================================
# MCTS Node
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

        # 이 node에서 행동할 player
        self.player = player

        self.parent = parent

        self.action = action

        self.prior = float(prior)

        self.visit_count = 0

        self.value_sum = 0.0

        self.children = {}

        self.expanded = False

        # GetValidActions() 결과 캐시
        self.valid_actions = None

        # Batched MCTS virtual loss
        self.virtual_loss = 0


    @property
    def value(self):

        if self.visit_count == 0:

            return 0.0

        return (
            self.value_sum /
            self.visit_count
        )

# ============================================================
# Batched Neural Network Evaluation
# ============================================================
# ============================================================
# Batched Neural Network Evaluation
# ============================================================

def evaluate_batch(model, boards, players):
    """
    여러 MCTS leaf를 한 번에 Neural Network로 평가한다.

    boards:
        list[Quoridor.Quoridor]

    players:
        list[int]
        각 board에서 평가할 player
        1 = P1
        2 = P2

    return:
        policy_logits:
            shape = (N, 132)

        values:
            shape = (N,)
    """

    if len(boards) != len(players):
        raise ValueError(
            f"boards/players size mismatch: "
            f"{len(boards)} != {len(players)}"
        )

    if len(boards) == 0:
        return (
            np.empty(
                (0, ACTION_SIZE),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # State batch
    # --------------------------------------------------------

    states = np.stack(
        [
            get_state(
                board,
                player
            )
            for board, player
            in zip(boards, players)
        ],
        axis=0
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Neural Network
    # --------------------------------------------------------

    with tf.device(GPU_DEVICE):

        policy_logits, values = model(
            states,
            training=False
        )

    policy_logits = (
        policy_logits
        .numpy()
        .astype(np.float32)
    )

    values = (
        values
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )

    return (
        policy_logits,
        values
    )

# ============================================================
# MCTS
# ============================================================
# ============================================================
# Batched MCTS
# ============================================================

class MCTS:

    def __init__(
        self,
        model,
        simulations=MCTS_SIMULATIONS,
        batch_size=MCTS_BATCH_SIZE,
        c_puct=MCTS_C_PUCT
    ):

        self.model = model

        self.simulations = int(
            simulations
        )

        self.batch_size = int(
            batch_size
        )

        self.c_puct = float(
            c_puct
        )

    # ========================================================
    # Valid Actions
    # ========================================================

    def get_valid_actions(
        self,
        node
    ):

        if node.valid_actions is None:

            node.valid_actions = [
                int(a)
                for a in Quoridor.GetValidActions(
                    node.board
                )
            ]

        return node.valid_actions

    # ========================================================
    # Batched Neural Network Evaluation
    # ========================================================

    def evaluate_batch(
        self,
        nodes
    ):

        if not nodes:

            return (
                np.empty(
                    (0, ACTION_SIZE),
                    dtype=np.float32
                ),
                np.empty(
                    (0,),
                    dtype=np.float32
                )
            )

        boards = [
            node.board
            for node in nodes
        ]

        players = [
            node.player
            for node in nodes
        ]

        policy_logits, values = evaluate_batch(
            self.model,
            boards,
            players
        )

        # ----------------------------------------------------
        # Mask invalid actions
        # ----------------------------------------------------

        policies = []

        for i, node in enumerate(nodes):

            valid_actions = (
                self.get_valid_actions(node)
            )

            policy_logits_i = (
                policy_logits[i]
            )

            policy = np.zeros(
                ACTION_SIZE,
                dtype=np.float32
            )

            if not valid_actions:

                policies.append(
                    policy
                )

                continue

            valid_logits = (
                policy_logits_i[
                    valid_actions
                ]
            )

            max_logit = np.max(
                valid_logits
            )

            exp_logits = np.exp(
                valid_logits - max_logit
            )

            denominator = np.sum(
                exp_logits
            )

            if (
                not np.isfinite(
                    denominator
                )
                or
                denominator <= 0.0
            ):

                probs = np.ones(
                    len(valid_actions),
                    dtype=np.float32
                )

                probs /= len(
                    valid_actions
                )

            else:

                probs = (
                    exp_logits /
                    denominator
                )

            policy[
                valid_actions
            ] = probs

            policies.append(
                policy
            )

        return (
            np.stack(
                policies,
                axis=0
            ),
            values
        )

    # ========================================================
    # Expand
    # ========================================================

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

        node.children.clear()

        next_player = (
            2
            if node.player == 1
            else 1
        )

        for action in valid_actions:

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

                raise RuntimeError(
                    f"MCTS action failed: "
                    f"{action}, "
                    f"result={result}"
                )

            child = MCTSNode(
                board=child_board,
                player=next_player,
                parent=node,
                action=action,
                prior=float(
                    policy[action]
                )
            )

            node.children[
                action
            ] = child

        node.expanded = True

    # ========================================================
    # PUCT Selection
    # ========================================================

    def select_child(
        self,
        node
    ):

        best_action = None

        best_child = None

        best_score = -np.inf

        sqrt_parent_visits = np.sqrt(
            node.visit_count + 1.0
        )

        for action, child in (
            node.children.items()
        ):

            # ------------------------------------------------
            # child.value는 child.player 관점
            #
            # parent 관점에서는 반대
            # ------------------------------------------------

            q = -child.value

            # ------------------------------------------------
            # Virtual loss
            # ------------------------------------------------

            virtual_loss_penalty = (
                child.virtual_loss
                * 1.0
            )

            q -= virtual_loss_penalty

            # ------------------------------------------------
            # U
            # ------------------------------------------------

            u = (
                self.c_puct
                * child.prior
                * sqrt_parent_visits
                /
                (
                    1.0
                    +
                    child.visit_count
                )
            )

            score = q + u

            if score > best_score:

                best_score = score

                best_action = action

                best_child = child

        return (
            best_action,
            best_child
        )

    # ========================================================
    # Apply Virtual Loss
    # ========================================================

    def apply_virtual_loss(
        self,
        path
    ):

        for node in path:

            node.visit_count += 1

            node.value_sum -= 1.0

            node.virtual_loss += 1

    # ========================================================
    # Remove Virtual Loss
    # ========================================================

    def remove_virtual_loss(
        self,
        path
    ):

        for node in path:

            node.visit_count -= 1

            node.value_sum += 1.0

            node.virtual_loss -= 1

    # ========================================================
    # Select Leaf
    # ========================================================

    def select_leaf(
        self,
        root
    ):

        node = root

        path = [
            root
        ]

        # ----------------------------------------------------
        # Traverse tree
        # ----------------------------------------------------

        while node.expanded:

            # ------------------------------------------------
            # Terminal check
            # ------------------------------------------------

            terminal_value = (
                get_terminal_value(
                    node.board,
                    node.player
                )
            )

            if terminal_value is not None:

                return (
                    node,
                    path,
                    terminal_value
                )

            # ------------------------------------------------
            # No children
            # ------------------------------------------------

            if not node.children:

                return (
                    node,
                    path,
                    None
                )

            # ------------------------------------------------
            # PUCT
            # ------------------------------------------------

            _, child = (
                self.select_child(
                    node
                )
            )

            if child is None:

                return (
                    node,
                    path,
                    None
                )

            node = child

            path.append(
                node
            )

        # ----------------------------------------------------
        # Terminal check for unexpanded node
        # ----------------------------------------------------

        terminal_value = (
            get_terminal_value(
                node.board,
                node.player
            )
        )

        if terminal_value is not None:

            return (
                node,
                path,
                terminal_value
            )

        # ----------------------------------------------------
        # Apply virtual loss
        #
        # 다음 batch selection에서
        # 동일한 경로를 다시 선택할 가능성을 줄인다.
        # ----------------------------------------------------

        self.apply_virtual_loss(
            path
        )

        return (
            node,
            path,
            None
        )

    # ========================================================
    # Backup
    # ========================================================

    def backup(
        self,
        node,
        value
    ):

        current = node

        current_value = float(
            value
        )

        while current is not None:

            current.visit_count += 1

            current.value_sum += (
                current_value
            )

            current_value = (
                -current_value
            )

            current = current.parent

    # ========================================================
    # Backup Batch Result
    # ========================================================

    def backup_leaf(
        self,
        node,
        path,
        value
    ):

        # ----------------------------------------------------
        # 먼저 virtual loss 제거
        # ----------------------------------------------------

        self.remove_virtual_loss(
            path
        )

        # ----------------------------------------------------
        # 실제 value backup
        # ----------------------------------------------------

        self.backup(
            node,
            value
        )

    # ========================================================
    # Root Dirichlet Noise
    # ========================================================

    def add_root_noise(
        self,
        root
    ):

        actions = list(
            root.children.keys()
        )

        if not actions:
            return

        noise = np.random.dirichlet(
            np.full(
                len(actions),
                DIRICHLET_ALPHA,
                dtype=np.float32
            )
        )

        for i, action in enumerate(
            actions
        ):

            child = (
                root.children[action]
            )

            child.prior = (
                (
                    1.0
                    -
                    DIRICHLET_EPSILON
                )
                *
                child.prior
                +
                DIRICHLET_EPSILON
                *
                float(noise[i])
            )

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        board,
        player,
        add_noise=True
    ):

        # ----------------------------------------------------
        # Root
        # ----------------------------------------------------

        root_board = (
            Quoridor.CloneBoard(
                board
            )
        )

        root = MCTSNode(
            board=root_board,
            player=player
        )

        # ----------------------------------------------------
        # Root evaluation
        #
        # 최초 root는 batch에 넣을 필요가 없으므로
        # 한 번 평가한다.
        # ----------------------------------------------------

        root_policy_batch, root_value_batch = (
            self.evaluate_batch(
                [root]
            )
        )

        root_policy = (
            root_policy_batch[0]
        )

        root_value = (
            float(
                root_value_batch[0]
            )
        )

        self.expand(
            root,
            root_policy
        )

        # ----------------------------------------------------
        # Root noise
        # ----------------------------------------------------

        if add_noise:

            self.add_root_noise(
                root
            )

        # ----------------------------------------------------
        # Simulations
        # ----------------------------------------------------

        simulations_done = 0

        while (
            simulations_done
            <
            self.simulations
        ):

            current_batch_size = min(
                self.batch_size,
                self.simulations
                -
                simulations_done
            )

            leaves = []

            paths = []

            terminal_results = []

            # ------------------------------------------------
            # Select leaves
            # ------------------------------------------------

            for _ in range(
                current_batch_size
            ):

                leaf, path, terminal_value = (
                    self.select_leaf(
                        root
                    )
                )

                # --------------------------------------------
                # Terminal leaf
                # --------------------------------------------

                if terminal_value is not None:

                    # select_leaf에서 virtual loss를
                    # 적용하지 않은 terminal path이므로
                    # 바로 backup
                    self.backup(
                        leaf,
                        terminal_value
                    )

                    simulations_done += 1

                    continue

                # --------------------------------------------
                # Non-terminal leaf
                # --------------------------------------------

                leaves.append(
                    leaf
                )

                paths.append(
                    path
                )

            # ------------------------------------------------
            # Batch NN evaluation
            # ------------------------------------------------

            if leaves:

                policies, values = (
                    self.evaluate_batch(
                        leaves
                    )
                )

                # --------------------------------------------
                # Expand + Backup
                # --------------------------------------------

                for i, leaf in enumerate(
                    leaves
                ):

                    policy = (
                        policies[i]
                    )

                    value = float(
                        values[i]
                    )

                    # ----------------------------------------
                    # Expand
                    # ----------------------------------------

                    self.expand(
                        leaf,
                        policy
                    )

                    # ----------------------------------------
                    # Backup
                    # ----------------------------------------

                    self.backup_leaf(
                        leaf,
                        paths[i],
                        value
                    )

                    simulations_done += 1

        # ----------------------------------------------------
        # Visit count policy
        # ----------------------------------------------------

        visit_policy = np.zeros(
            ACTION_SIZE,
            dtype=np.float32
        )

        total_visits = sum(
            child.visit_count
            for child
            in root.children.values()
        )

        if total_visits <= 0:

            valid_actions = list(
                root.children.keys()
            )

            if valid_actions:

                visit_policy[
                    valid_actions
                ] = (
                    1.0 /
                    len(valid_actions)
                )

            return visit_policy

        for action, child in (
            root.children.items()
        ):

            visit_policy[action] = (
                child.visit_count
                /
                total_visits
            )

        return visit_policy

# ============================================================
# Replay Buffer
# ============================================================

replay_buffer = deque(
    maxlen=REPLAY_BUFFER_SIZE
)


# ============================================================
# Self Play
# ============================================================

# ============================================================
# Self Play
# ============================================================

def self_play_game(
    model,
    simulations=MCTS_SIMULATIONS,
    batch_size=MCTS_BATCH_SIZE
):

    board = Quoridor.Quoridor()

    history = []

    move_count = 0

    MAX_MOVES = 200

    # --------------------------------------------------------
    # winner
    #
    # 0  = 진행 중
    # 1  = P1 승리
    # 2  = P2 승리
    # -1 = 무승부
    # --------------------------------------------------------

    winner = 0

    # --------------------------------------------------------
    # MCTS
    #
    # 게임 전체에서 한 번만 생성
    # --------------------------------------------------------

    mcts = MCTS(
        model,
        simulations=simulations,
        batch_size=batch_size
    )

    while True:

        # ----------------------------------------------------
        # Current player
        # ----------------------------------------------------

        player = (
            1
            if bool(board.isP1Turn)
            else 2
        )

        # ----------------------------------------------------
        # Terminal check
        # ----------------------------------------------------

        winner = int(
            Quoridor.isGameOver(
                board
            )
        )

        if winner != 0:
            break

        # ----------------------------------------------------
        # Maximum moves
        # ----------------------------------------------------

        if move_count >= MAX_MOVES:

            winner = -1

            print(
                f"[SelfPlay] "
                f"Game reached "
                f"{MAX_MOVES} moves. "
                f"Draw."
            )

            break

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state = get_state(
            board,
            player
        )

        # ----------------------------------------------------
        # MCTS
        # ----------------------------------------------------

        mcts_start_time = time.time()

        visit_policy = mcts.search(
            board,
            player,
            add_noise=True
        )

        mcts_elapsed = (
            time.time()
            -
            mcts_start_time
        )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        if move_count < 20:

            temperature = 1.0

        else:

            temperature = 0.1

        # ----------------------------------------------------
        # Valid actions
        # ----------------------------------------------------

        valid_actions = [
            int(a)
            for a in Quoridor.GetValidActions(
                board
            )
        ]

        if not valid_actions:

            raise RuntimeError(
                "No valid actions available."
            )

        # ----------------------------------------------------
        # Temperature
        # ----------------------------------------------------

        probs = (
            visit_policy[
                valid_actions
            ]
            .astype(
                np.float64
            )
        )

        if temperature != 1.0:

            probs = np.power(
                probs,
                1.0 / temperature
            )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        prob_sum = probs.sum()

        if (
            not np.isfinite(
                prob_sum
            )
            or
            prob_sum <= 0
        ):

            probs = np.ones(
                len(valid_actions),
                dtype=np.float64
            )

            probs /= len(
                valid_actions
            )

        else:

            probs /= prob_sum

        # ----------------------------------------------------
        # Training sample
        # ----------------------------------------------------

        target_policy = np.zeros(
            ACTION_SIZE,
            dtype=np.float32
        )

        target_policy[
            valid_actions
        ] = probs.astype(
            np.float32
        )

        history.append(
            (
                state,
                target_policy,
                player
            )
        )

        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = int(
            np.random.choice(
                valid_actions,
                p=probs
            )
        )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        result = execute_action(
            board,
            action
        )

        if result != 0:

            raise RuntimeError(
                f"MCTS selected invalid "
                f"action: {action}"
            )

        move_count += 1

        # ----------------------------------------------------
        # Game over
        # ----------------------------------------------------

        winner = int(
            Quoridor.isGameOver(
                board
            )
        )

        if winner != 0:
            break

        # ----------------------------------------------------
        # Optional MCTS timing
        # ----------------------------------------------------
        '''
                print(
            f"[Move {move_count:3d}] "
            f"P{player} "
            f"MCTS={mcts_elapsed:.3f}s"
        )
        '''
        

    # ========================================================
    # Convert winner -> z
    # ========================================================

    training_data = []

    for state, policy, player in history:

        if winner == 1 or winner == 2:

            if player == winner:

                z = 1.0

            else:

                z = -1.0

        elif winner == -1:

            z = 0.0

        else:

            z = 0.0

        training_data.append(
            (
                state,
                policy,
                z
            )
        )

    return (
        training_data,
        winner
    )

# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    learning_rate=LEARNING_RATE
)


# ============================================================
# Training step
# ============================================================

@tf.function
def train_step(
    states,
    target_policy,
    target_value
):

    with tf.GradientTape() as tape:

        # ----------------------------------------------------
        # Network
        # ----------------------------------------------------

        policy_logits, value = model(
            states,
            training=True
        )


        # ----------------------------------------------------
        # Policy loss
        # ----------------------------------------------------

        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=target_policy,
            logits=policy_logits
        )


        policy_loss = tf.reduce_mean(
            policy_loss
        )


        # ----------------------------------------------------
        # Value loss
        # ----------------------------------------------------

        value_prediction = tf.squeeze(
            value,
            axis=-1
        )


        value_loss = tf.reduce_mean(
            tf.square(
                target_value
                -
                value_prediction
            )
        )


        # ----------------------------------------------------
        # L2
        # ----------------------------------------------------

        l2_terms = [

            tf.reduce_sum(
                tf.square(variable)
            )

            for variable
            in model.trainable_variables

            if (
                "bias" not in
                variable.name.lower()
            )
        ]


        if l2_terms:

            l2_loss = tf.add_n(
                l2_terms
            )

        else:

            l2_loss = tf.constant(
                0.0,
                dtype=tf.float32
            )


        l2_loss *= L2_WEIGHT


        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        loss = (
            policy_loss
            +
            value_loss
            +
            l2_loss
        )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    grads = tape.gradient(
        loss,
        model.trainable_variables
    )


    # --------------------------------------------------------
    # Finite check
    # --------------------------------------------------------

    safe_grads = []

    for grad in grads:

        if grad is None:

            safe_grads.append(
                None
            )

            continue


        grad = tf.clip_by_norm(
            grad,
            GRAD_CLIP_NORM
        )


        safe_grads.append(
            grad
        )


    # --------------------------------------------------------
    # Apply
    # --------------------------------------------------------

    optimizer.apply_gradients(
        zip(
            safe_grads,
            model.trainable_variables
        )
    )

    return (
        loss,
        policy_loss,
        value_loss
    )


# ============================================================
# Train from Replay Buffer
# ============================================================

def train_from_buffer(
    batch_size=BATCH_SIZE
):

    if len(replay_buffer) < batch_size:

        return None


    batch = random.sample(
        replay_buffer,
        batch_size
    )


    states = np.stack(
        [
            sample[0]
            for sample in batch
        ]
    ).astype(
        np.float32
    )


    policies = np.stack(
        [
            sample[1]
            for sample in batch
        ]
    ).astype(
        np.float32
    )


    values = np.array(
        [
            sample[2]
            for sample in batch
        ],
        dtype=np.float32
    )


    values = values.reshape(
        -1,
        1
    )


    return train_step(
        states,
        policies,
        values
    )


# ============================================================
# Create model
# ============================================================
with tf.device(GPU_DEVICE):

    model = AlphaZeroNetwork(
        num_blocks=NUM_RES_BLOCKS,
        filters=NUM_FILTERS
    )

    # ========================================================
    # Build model
    # ========================================================

    dummy_input = tf.zeros(
        (
            1,
            BOARD_SIZE,
            BOARD_SIZE,
            7
        ),
        dtype=tf.float32
    )

    policy_logits, value = model(
        dummy_input,
        training=False
    )

    # ========================================================
    # Load previous AlphaZero weights
    # ========================================================

    if LOAD_CHECKPOINT:

        if os.path.exists(CHECKPOINT_PATH):

            print()
            print("========================================")
            print("Loading AlphaZero checkpoint")
            print("========================================")
            print(
                f"Checkpoint: {CHECKPOINT_PATH}"
            )

            model.load_weights(
                CHECKPOINT_PATH
            )

            print("Checkpoint loaded successfully.")
            print("========================================")
            print()

        else:

            print()
            print("========================================")
            print("Checkpoint not found")
            print("========================================")
            print(
                f"Path: {CHECKPOINT_PATH}"
            )
            print("Training from scratch.")
            print("========================================")
            print()

print()
print("========================================")
print("AlphaZero Model")
print("========================================")
print(
    "Policy shape:",
    policy_logits.shape
)
print(
    "Value shape:",
    value.shape
)
print("========================================")
print()


# ============================================================
# Main Training Loop
# ============================================================

for game in range(NUM_GAMES):

    print()
    print(
        "========================================"
    )

    print(
        f"Self Play Game {game + 1}/{NUM_GAMES}"
    )

    print(
        "========================================"
    )


    # --------------------------------------------------------
    # Self play
    # --------------------------------------------------------

    game_data, winner = self_play_game(
        model,
        simulations=MCTS_SIMULATIONS,
        batch_size=MCTS_BATCH_SIZE
    )

    replay_buffer.extend(
        game_data
    )


    print(
        f"Winner = P{winner}"
    )

    print(
        f"Game data = {len(game_data)}"
    )

    print(
        f"Replay buffer = {len(replay_buffer)}"
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    if len(replay_buffer) >= BATCH_SIZE:

        for _ in range(4):

            result = train_from_buffer(
                BATCH_SIZE
            )


        loss, policy_loss, value_loss = result


        print(
            f"Loss = {float(loss):.6f}"
        )

        print(
            f"Policy = {float(policy_loss):.6f}"
        )

        print(
            f"Value = {float(value_loss):.6f}"
        )

        # ========================================================
        # NaN / Inf Weight Check
        # ========================================================

        for variable in model.trainable_variables:

            values = variable.numpy()

            if not np.all(np.isfinite(values)):

                print(
                    "NaN/Inf weight:",
                    variable.name
                )
    print_progress(
        game + 1,
        NUM_GAMES,
        training_start_time
        )
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if (
        (game + 1) % 100 == 0
        or
        (game + 1) == NUM_GAMES
    ):

        model.save_weights(
            CHECKPOINT_PATH
        )

        print(
            f"Saved: {CHECKPOINT_PATH}"
        )