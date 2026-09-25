import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision


# ============================================================
# 설정
# ============================================================

nA = 132

BOARD_SIZE = 9

# 학습할 때 사용한 Residual Block 수
NUM_RES_BLOCKS = 20

# 학습할 때 사용한 필터 수
NUM_FILTERS = 128


# ============================================================
# Quoridor C++ Module
# ============================================================

sys.path.append(
    "../TESLA_Quoridor_AI/build/Debug"
)

sys.path.append(
    "../TESLA_Quoridor_AI"
)

import Quoridor


# ============================================================
# TensorFlow
# ============================================================

mixed_precision.set_global_policy(
    "mixed_float16"
)


# ============================================================
# ResNet DQN
# ============================================================

class Network(keras.Model):

    def __init__(
        self,
        nS,
        nA,
        num_blocks=20,
        num_filters=128
    ):
        super().__init__()

        self.nS = nS
        self.nA = nA

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

        self.conv1 = layers.Conv2D(
            num_filters,
            kernel_size=3,
            padding="same",
            use_bias=False
        )

        self.bn1 = layers.BatchNormalization()

        self.act1 = layers.LeakyReLU(
            negative_slope=0.1
        )

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        self.res_blocks = []

        for _ in range(num_blocks):

            conv1 = layers.Conv2D(
                num_filters,
                kernel_size=3,
                padding="same",
                use_bias=False
            )

            bn1 = layers.BatchNormalization()

            act1 = layers.LeakyReLU(
                negative_slope=0.1
            )

            conv2 = layers.Conv2D(
                num_filters,
                kernel_size=3,
                padding="same",
                use_bias=False
            )

            bn2 = layers.BatchNormalization()

            self.res_blocks.append(
                (
                    conv1,
                    bn1,
                    act1,
                    conv2,
                    bn2
                )
            )

        self.res_act = layers.LeakyReLU(
            negative_slope=0.1
        )

        # ----------------------------------------------------
        # Fully Connected
        # ----------------------------------------------------

        self.fc1 = layers.Dense(
            256
        )

        self.fc_act = layers.LeakyReLU(
            negative_slope=0.1
        )

        # ----------------------------------------------------
        # Q value
        # ----------------------------------------------------

        self.q_head = layers.Dense(
            nA,
            dtype="float32"
        )

    # ========================================================
    # Forward
    # ========================================================

    def call(
        self,
        x,
        training=False
    ):

        # ----------------------------------------------------
        # Initial convolution
        # ----------------------------------------------------

        x = self.conv1(x)

        x = self.bn1(
            x,
            training=training
        )

        x = self.act1(x)

        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        for (
            conv1,
            bn1,
            act1,
            conv2,
            bn2
        ) in self.res_blocks:

            shortcut = x

            y = conv1(x)

            y = bn1(
                y,
                training=training
            )

            y = act1(y)

            y = conv2(y)

            y = bn2(
                y,
                training=training
            )

            x = y + shortcut

            x = self.res_act(x)

        # ----------------------------------------------------
        # Global Average Pooling
        # ----------------------------------------------------

        x = tf.reduce_mean(
            x,
            axis=[1, 2]
        )

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        x = self.fc1(x)

        x = self.fc_act(x)

        # ----------------------------------------------------
        # Q values
        # ----------------------------------------------------

        return self.q_head(x)


# ============================================================
# Model 생성
# ============================================================

model = Network(
    nS=567,
    nA=nA,
    num_blocks=NUM_RES_BLOCKS,
    num_filters=NUM_FILTERS
)


# ============================================================
# Model Build
# ============================================================

dummy_input = tf.zeros(
    (1, 9, 9, 7),
    dtype=tf.float32
)

model(
    dummy_input,
    training=False
)


# ============================================================
# Weight Load
# ============================================================

FILENAME = "./TESLA_Quoridor_AI/dqn_episode_100000.weights.h5"

if not os.path.exists(FILENAME):

    raise FileNotFoundError(
        f"\nWeight 파일을 찾을 수 없습니다.\n"
        f"{os.path.abspath(FILENAME)}"
    )

print()
print("Weight loading...")
print(FILENAME)

model.load_weights(
    FILENAME
)

print()
print("DQN model loaded successfully!")
print()

print("\n================================")
print("CHECK MODEL WEIGHTS")
print("================================")

for weight in model.weights:
    value = weight.numpy()

    print(
        f"{weight.name:50s} "
        f"shape={str(value.shape):20s} "
        f"min={np.nanmin(value):12.6f} "
        f"max={np.nanmax(value):12.6f} "
        f"nan={np.isnan(value).sum()} "
        f"inf={np.isinf(value).sum()}"
    )


# ============================================================
# State 생성
#
# 중요:
#
# 학습 모델은 현재 플레이어 기준으로 state를 정규화함.
#
# AI = P2
#
# channel 0 = P2
# channel 1 = P1
# channel 2 = Horizontal Wall
# channel 3 = Vertical Wall
# channel 4 = P2 turn
# channel 5 = P2 Wall Count
# channel 6 = P1 Wall Count
# ============================================================

def get_state(board):

    state = np.zeros(
        (9, 9, 7),
        dtype=np.float32
    )

    # ========================================================
    # AI = P2
    # ========================================================

    my_pos = board.P1
    opponent_pos = board.P2

    my_x = int(my_pos[0])
    my_y = int(my_pos[1])

    opponent_x = int(opponent_pos[0])
    opponent_y = int(opponent_pos[1])

    state[
        my_x,
        my_y,
        0
    ] = 1.0

    state[
        opponent_x,
        opponent_y,
        1
    ] = 1.0

    # ========================================================
    # Horizontal Wall
    #
    # 학습 코드:
    #
    # WallH[x] & (1 << y)
    # ========================================================

    for x in range(8):

        for y in range(8):

            if board.WallH[x] & (1 << y):

                state[
                    x,
                    y,
                    2
                ] = 1.0

    # ========================================================
    # Vertical Wall
    #
    # 학습 코드:
    #
    # WallVt[y] & (1 << x)
    # ========================================================

    for x in range(8):

        for y in range(8):

            if board.WallVt[y] & (1 << x):

                state[
                    x,
                    y,
                    3
                ] = 1.0

    # ========================================================
    # Turn
    #
    # P2 차례라면 1
    # ========================================================

    if not board.isP1Turn:

        state[:, :, 4] = 1.0

    # ========================================================
    # Wall Count
    #
    # P2 = 나
    # P1 = 상대
    # ========================================================

    state[:, :, 5] = (
        int(board.WallCount[1]) / 10.0
    )

    state[:, :, 6] = (
        int(board.WallCount[0]) / 10.0
    )

    return state


# ============================================================
# Action Decode
# ============================================================

def decode_action(action):

    action = int(action)

    # --------------------------------------------------------
    # Move
    # --------------------------------------------------------

    if action < 4:

        return (
            "move",
            action
        )

    # --------------------------------------------------------
    # Horizontal Wall
    # --------------------------------------------------------

    elif action < 68:

        index = action - 4

        x = index // 8
        y = index % 8

        return (
            "wall_h",
            (x, y)
        )

    # --------------------------------------------------------
    # Vertical Wall
    # --------------------------------------------------------

    else:

        index = action - 68

        x = index // 8
        y = index % 8

        return (
            "wall_v",
            (x, y)
        )


# ============================================================
# AI Action
# ============================================================

def ai_action(
    board,
    valid_actions
):

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = get_state(board)

    state = np.expand_dims(
        state,
        axis=0
    )

    # --------------------------------------------------------
    # Q values
    # --------------------------------------------------------

    q_values = model(
        state,
        training=False
    ).numpy()[0]

    # --------------------------------------------------------
    # Invalid action 제거
    # --------------------------------------------------------

    masked_q = np.full(
        nA,
        -np.inf,
        dtype=np.float32
    )

    for action in valid_actions:

        action = int(action)

        masked_q[action] = q_values[action]

    # --------------------------------------------------------
    # Greedy action
    # --------------------------------------------------------

    action = int(
        np.argmax(masked_q)
    )

    return action


# ============================================================
# AI Q-value 출력
# ============================================================

def print_ai_q_values(
    board,
    valid_actions
):

    state = get_state(board)

    state = np.expand_dims(
        state,
        axis=0
    )

    q_values = model(
        state,
        training=False
    ).numpy()[0]

    candidates = []

    for action in valid_actions:

        action = int(action)

        candidates.append(
            (
                action,
                float(q_values[action])
            )
        )

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print()
    print("--------------------------------")
    print("AI Q VALUES TOP 10")
    print("--------------------------------")

    for action, q in candidates[:10]:

        action_type, data = decode_action(
            action
        )

        print(
            f"{action:3d} | "
            f"Q = {q:9.4f} | "
            f"{action_type} {data}"
        )

    print("--------------------------------")


# ============================================================
# Action 실행
# ============================================================

def execute_action(
    board,
    action
):

    action_type, data = decode_action(
        action
    )

    # ========================================================
    # Move
    # ========================================================

    if action_type == "move":

        move = data

        result = Quoridor.MovePlayer(
            board,
            move
        )

        return result

    # ========================================================
    # Horizontal Wall
    # ========================================================

    elif action_type == "wall_h":

        x, y = data

        result = Quoridor.PlaceWallH(
            board,
            [x, y]
        )

        return result

    # ========================================================
    # Vertical Wall
    # ========================================================

    elif action_type == "wall_v":

        x, y = data

        result = Quoridor.PlaceWallV(
            board,
            [x, y]
        )

        return result

    return -1


# ============================================================
# Grid Position Input
#
# 기존 인터페이스 그대로
#
# 행: 1 ~ 8
# 열: 1 ~ 8
# ============================================================

def GridPosInput():

    while True:

        try:

            x = int(
                input("행: ")
            ) - 1

            y = int(
                input("열: ")
            ) - 1

            if (
                0 <= x <= 7
                and
                0 <= y <= 7
            ):

                return x, y

            print("범위 밖")

        except ValueError:

            print("유효하지 않은 입력")


# ============================================================
# Human Action
#
# 기존 인터페이스 그대로
# ============================================================

def human_action(board):

    print("당신의 차례입니다.")

    print("0: 위")
    print("1: 아래")
    print("2: 왼쪽")
    print("3: 오른쪽")
    print("4: 가로 벽")
    print("5: 세로 벽")

    command = input("입력: ")

    # --------------------------------------------------------
    # Move
    # --------------------------------------------------------

    if command in [
        "0",
        "1",
        "2",
        "3"
    ]:

        return int(command)

    # --------------------------------------------------------
    # Horizontal Wall
    # --------------------------------------------------------

    elif command == "4":

        x, y = GridPosInput()

        return 4 + x * 8 + y

    # --------------------------------------------------------
    # Vertical Wall
    # --------------------------------------------------------

    elif command == "5":

        x, y = GridPosInput()

        return 68 + x * 8 + y

    # --------------------------------------------------------
    # Invalid input
    # --------------------------------------------------------

    else:

        print(
            "잘못된 입력입니다."
        )

        board.PrintBoard()

        return human_action(
            board
        )

def setup_random_walls(self):
    """
    게임 시작 시 0~3 개의 랜덤한 합법 벽을 설치한다.

    설치된 벽은 PlaceWallH / PlaceWallV를 통해
    C++ 쪽 WallCount에서 자동으로 차감된다.

    벽 설치가 끝난 후에는 항상 P1 턴으로 복구한다.
    """

    # ------------------------------------------
    # 랜덤하게 설치할 벽 개수
    # ------------------------------------------

    wall_count = random.randint(0, 3)

    placed_count = 0

    # ------------------------------------------
    # 최대 시도 횟수
    # ------------------------------------------

    max_attempts = 200
    attempts = 0

    while placed_count < wall_count:

        attempts += 1

        if attempts > max_attempts:
            print(
                f"[RandomWall] "
                f"최대 시도 횟수 초과: "
                f"{placed_count}/{wall_count}"
            )
            break

        # --------------------------------------
        # 현재 가능한 행동
        # --------------------------------------

        valid_actions = list(
            Quoridor.GetValidActions(
                self.board
            )
        )

        # --------------------------------------
        # 벽 행동만 추출
        #
        # 4 ~ 67   : Horizontal
        # 68 ~ 131 : Vertical
        # --------------------------------------

        valid_wall_actions = [
            int(action)
            for action in valid_actions
            if 4 <= int(action) < 132
        ]

        if not valid_wall_actions:
            break

        # --------------------------------------
        # 랜덤 벽 선택
        # --------------------------------------

        action = random.choice(
            valid_wall_actions
        )

        # --------------------------------------
        # 벽 설치
        # --------------------------------------

        if action < 68:

            wall_action = action - 4

            x = wall_action // 8
            y = wall_action % 8

            result = Quoridor.PlaceWallH(
                self.board,
                [x, y]
            )

        else:

            wall_action = action - 68

            x = wall_action // 8
            y = wall_action % 8

            result = Quoridor.PlaceWallV(
                self.board,
                [x, y]
            )

        # --------------------------------------
        # 설치 성공
        # --------------------------------------

        if result == 0:
            placed_count += 1

    # ------------------------------------------
    # 시작 플레이어를 P1로 복구
    # ------------------------------------------

    self.board.isP1Turn = True
    if self.debug == True:
        print(
            f"[RandomWall] "
            f"Placed={placed_count} | "
            f"P1 Walls={self.board.WallCount[0]} | "
            f"P2 Walls={self.board.WallCount[1]} | "
            f"P1 Distance={Quoridor.AStar(self.board, True)} | "
            f"P2 Distance={Quoridor.AStar(self.board, False)}",
            flush=True
        )

    return placed_count

# ============================================================
# Main
# ============================================================

def main():

    board = Quoridor.Quoridor()

    print(
        "================================"
    )

    print(
        "       Quoridor AI Battle"
    )

    print(
        "================================"
    )

    print()

    print(
        "P1 = 당신"
    )

    print(
        "P2 = AI"
    )

    print()

    # ========================================================
    # Game Loop
    # ========================================================

    while True:

        # ----------------------------------------------------
        # 현재 보드
        # ----------------------------------------------------

        board.PrintBoard()

        # ----------------------------------------------------
        # 현재 턴
        # ----------------------------------------------------

        if board.isP1Turn:

            # =================================================
            # Human
            # =================================================

            valid_actions = (
                Quoridor.GetValidActions(
                    board
                )
            )

            while True:

                action = human_action(
                    board
                )

                # pybind 결과가 numpy/list일 수도 있으므로
                # int로 변환해서 비교
                valid_actions_int = [
                    int(a)
                    for a in valid_actions
                ]

                if action in valid_actions_int:

                    break

                print(
                    "그 수는 둘 수 없습니다."
                )

                board.PrintBoard()

        else:

            # =================================================
            # AI
            # =================================================

            valid_actions = (
                Quoridor.GetValidActions(
                    board
                )
            )

            valid_actions = [
                int(a)
                for a in valid_actions
            ]

            # -------------------------------------------------
            # AI Q-value 표시
            # -------------------------------------------------

            print()
            print("AI가 생각 중...")

            # 디버깅이 필요하면 이 줄을 유지
            print_ai_q_values(
                board,
                valid_actions
            )

            # -------------------------------------------------
            # AI action
            # -------------------------------------------------

            action = ai_action(
                board,
                valid_actions
            )

            action_type, data = decode_action(
                action
            )

            print(
                "AI action:",
                action,
                "|",
                action_type,
                data
            )

        # ====================================================
        # Action 실행
        # ====================================================

        result = execute_action(
            board,
            action
        )

        # ----------------------------------------------------
        # 실행 실패
        # ----------------------------------------------------

        if result != 0:

            print(
                "행동 실행 실패:",
                result
            )

            continue

        # ====================================================
        # Game Over
        # ====================================================

        winner = Quoridor.isGameOver(
            board
        )

        if winner != 0:

            board.PrintBoard()

            print()

            print(
                "================================"
            )

            print(
                f"승자: P{winner}"
            )

            if winner == 1:

                print(
                    "당신이 승리했습니다!"
                )

            elif winner == 2:

                print(
                    "AI가 승리했습니다!"
                )

            print(
                "================================"
            )

            return


# ============================================================
# Start
# ============================================================

if __name__ == "__main__":

    main()