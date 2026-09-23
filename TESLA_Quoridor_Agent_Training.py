import os

# TensorFlow import 전에 설정
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, mixed_precision
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

mixed_precision.set_global_policy("float32")
from collections import deque
import numpy as np
import random

import sys
sys.path.append('../TESLA_Quoridor_AI/build/Debug')  # Add the path to TESLA_Quoridor_AI module
sys.path.append('../TESLA_Quoridor_AI')
sys.path.insert(0, './build-linux')
#print(Quoridor("Board"))

import Quoridor

'''
Board = Quoridor.Quoridor()

#Board.P1 = [0, 0]
WallH = [0,0]
print(Quoridor.PlaceWallH(Board, WallH))
for i in range(4):
     Quoridor.PlaceWallH(Board, WallH)
     WallH[1] += 2

Quoridor.PlaceWallH(Board, [1,7])
Quoridor.PlaceWallV(Board, [1,6])

print(Quoridor.AStar(Board))
#Board.P2 = [8,5]
print(Quoridor.isGameOver(Board))
#print(Quoridor.canPlaceWallV(Board, [1, 6]))  # Example usage of the C++ function
Board.PrintBoard()

'''

#print(Quoridor.MovePlayer(Board, 255))
#print(Board.P1)
#print(Board.P2)
#print(Board.WallCount)
#print(Board.isP1Turn)

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

#아래는 chatgpt가 tensorflow 2.x 버전으로 바꾼 것
#맞는지는 보장 못함 ㅎ
class Network:

    @staticmethod
    def residual_block(x, filters=128, block_id=0):
        """
        Quoridor용 Residual Block

        x
        │
        ├───────────────┐
        │               │
        ▼               │
      Conv 3×3          │
      BatchNorm         │
      LeakyReLU         │
      Conv 3×3          │
      BatchNorm         │
        │               │
        └── Add ◄───────┘
             │
         LeakyReLU
        """

        shortcut = x

        # ------------------------------------------------------------
        # Conv 1
        # ------------------------------------------------------------

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"res_{block_id}_conv1"
        )(x)

        x = keras.layers.BatchNormalization(
            name=f"res_{block_id}_bn1"
        )(x)

        x = keras.layers.LeakyReLU(
            negative_slope=0.1,
            name=f"res_{block_id}_relu1"
        )(x)

        # ------------------------------------------------------------
        # Conv 2
        # ------------------------------------------------------------

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"res_{block_id}_conv2"
        )(x)

        x = keras.layers.BatchNormalization(
            name=f"res_{block_id}_bn2"
        )(x)

        # ------------------------------------------------------------
        # Skip Connection
        # ------------------------------------------------------------

        x = keras.layers.Add(
            name=f"res_{block_id}_add"
        )([
            shortcut,
            x
        ])

        x = keras.layers.LeakyReLU(
            negative_slope=0.1,
            name=f"res_{block_id}_relu2"
        )(x)

        return x

    def __init__(
        self,
        nS,
        nA,
        scope="estimator",
        learning_rate=0.0001,
        neural_architecture=None,
        global_step=None,
        summaries_dir=None
    ):

        self.nS = nS
        self.nA = nA
        self.learning_rate = learning_rate
        self.scope = scope

        with tf.device(GPU_DEVICE):

            # ========================================================
            # Input
            # ========================================================

            inputs = keras.layers.Input(
                shape=(9, 9, 7),
                name="board_input"
            )

            # ========================================================
            # Initial Convolution
            # ========================================================

            x = keras.layers.Conv2D(
                filters=128,
                kernel_size=3,
                padding="same",
                use_bias=False,
                name="initial_conv"
            )(inputs)

            x = keras.layers.BatchNormalization(
                name="initial_bn"
            )(x)

            x = keras.layers.LeakyReLU(
                negative_slope=0.1,
                name="initial_relu"
            )(x)

            # ========================================================
            # Residual Tower
            # ========================================================

            for i in range(20):

                x = self.residual_block(
                    x,
                    filters=128,
                    block_id=i
                )

            # ========================================================
            # Global Average Pooling
            # ========================================================

            x = keras.layers.GlobalAveragePooling2D(
                name="global_average_pooling"
            )(x)

            # ========================================================
            # Fully Connected
            # ========================================================

            x = keras.layers.Dense(
                256,
                name="dense_256"
            )(x)

            x = keras.layers.LeakyReLU(
                negative_slope=0.1,
                name="dense_relu"
            )(x)

            # ========================================================
            # Q Value Head
            # ========================================================

            outputs = keras.layers.Dense(
                nA,
                activation=None,
                dtype="float32",
                name="q_values"
            )(x)

            # ========================================================
            # Model
            # ========================================================

            self.model = keras.Model(
                inputs=inputs,
                outputs=outputs,
                name=f"Quoridor_ResNet_DQN_{scope}"
            )

            self.model.summary()

        # ============================================================
        # Optimizer
        # ============================================================

        self.optimizer = Adam(
            learning_rate=learning_rate
        )

        # ============================================================
        # Compile
        # ============================================================

        self.model.compile(
            optimizer=self.optimizer,
            loss="mse"
        )

    def predict(self, s):

        s = np.asarray(
            s,
            dtype=np.float32
        )

        return self.model(
            s,
            training=False
        ).numpy()

    def fit(self, s, target):

        s = np.asarray(
            s,
            dtype=np.float32
        )

        target = np.asarray(
            target,
            dtype=np.float32
        )

        return self.model.train_on_batch(
            s,
            target
        )

class Memory:
    '''
    dequeue를 기반으로 한 경험 재생 메모리 클래스
    '''
    def __init__(self, memory_size = 5000):
        self.memory = deque(maxlen = memory_size)
        
    def __len__(self):
        return len(self.memory)
    
    def add_memory(
        self,s,a,r,s_,status,next_valid_actions):
        self.memory.append((s,a,r,s_,status,next_valid_actions))
    
    def recall_memories(self):
        #메모리에서 무작위로 배치 샘플을 추출
        return list(self.memory)
    
    def sample(self, batch_size):
        return random.sample(
            self.memory,
            batch_size
        )
    
class DQNAgent:
    #아래는 chatgpt가 tensorflow 2.x 버전으로 바꾼 것
    #맞는지는 보장 못함 ㅎ
    def __init__(
        self,
        nS,
        nA,
        experiment_dir
    ):

        self.nS = nS
        self.nA = nA

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9998

        self.gamma = 0.9998

        self.learning_rate = 0.0001

        self.batch_size = 128

        self.memory = Memory(
            memory_size=500000
        )

        self.experiment_dir = os.path.abspath(
            experiment_dir
        )

        os.makedirs(
            self.experiment_dir,
            exist_ok=True
        )

        # Q network
        self.model = Network(
            nS=self.nS,
            nA=self.nA,
            scope="q",
            learning_rate=self.learning_rate
        )

        # Target network
        self.target_model = Network(
            nS=self.nS,
            nA=self.nA,
            scope="target_q",
            learning_rate=self.learning_rate
        )

        # Target network 초기화
        self.target_model.model.set_weights(
            self.model.model.get_weights()
        )
        
    def epsilon_update(self):
        #탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def save_weights(self, filename):

        self.model.model.save_weights(
            filename
        )

        print(
            f"Model weights saved: {filename}"
        )
        
    def load_weights(self, filename):

        self.model.model.load_weights(
            filename
        )

        print(
            f"Model weights loaded: {filename}"
        )

    def update_target_model(self):
        self.target_model.model.set_weights(self.model.model.get_weights())
    
    def act(self, s, valid_actions):

        valid_actions = np.asarray(
            valid_actions,
            dtype=np.int32
        ).reshape(-1)

        if len(valid_actions) == 0:

            raise RuntimeError(
                "Valid action이 없습니다."
            )

        # 탐험
        if np.random.rand() < self.epsilon:
            return int(
                random.choice(
                    valid_actions.tolist()
                )
            )

        # 상태
        s = np.asarray(
            s,
            dtype=np.float32
        )

        if s.ndim == 3:
            s = np.expand_dims(s, axis=0)

        q_values = self.model.model(s,training=False).numpy()[0]

        # valid action만 추출
        valid_q_values = q_values[valid_actions]

        # 가장 높은 Q값의 위치
        best_index = np.argmax(
            valid_q_values
        )

        # 실제 action 번호
        return int(
            valid_actions[best_index]
        )
        
    def act_batch(self, states, valid_actions_list):
        """
        여러 환경의 상태를 한 번에 GPU에 넣어서 행동을 결정한다.

        states:
            shape = (N, 9, 9, 7)

        valid_actions_list:
            각 환경에서 가능한 action 목록
        """

        batch_size = len(states)

        if batch_size == 0:
            return []

        states = np.asarray(
            states,
            dtype=np.float32
        )

        # -------------------------------------------------
        # Q값을 한 번에 계산
        # -------------------------------------------------
        with tf.device(GPU_DEVICE):
            q_values = self.model.model(
                states,
                training=False
            ).numpy()

        actions = []

        for i in range(batch_size):

            valid_actions = np.asarray(
                valid_actions_list[i],
                dtype=np.int32
            ).reshape(-1)

            if len(valid_actions) == 0:
                raise RuntimeError(
                    f"환경 {i}: Valid action이 없습니다."
                )

            # -------------------------------------------------
            # epsilon-greedy
            # -------------------------------------------------

            if np.random.rand() < self.epsilon:

                action = random.choice(
                    valid_actions.tolist()
                )

            else:

                valid_q_values = q_values[
                    i,
                    valid_actions
                ]

                best_index = np.argmax(
                    valid_q_values
                )

                action = valid_actions[
                    best_index
                ]

            actions.append(
                int(action)
            )

        return actions
    
    def replay(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(
            self.batch_size
        )

        # ============================================================
        # CPU에서 numpy batch 구성
        # ============================================================

        states = np.asarray(
            [x[0] for x in batch],
            dtype=np.float32
        )

        actions = np.asarray(
            [x[1] for x in batch],
            dtype=np.int32
        )

        rewards = np.asarray(
            [x[2] for x in batch],
            dtype=np.float32
        )

        next_states = np.asarray(
            [x[3] for x in batch],
            dtype=np.float32
        )

        dones = np.asarray(
            [x[4] for x in batch],
            dtype=np.float32
        )

        # ============================================================
        # Valid action mask
        # ============================================================

        next_action_masks = np.zeros(
            (len(batch), self.nA),
            dtype=np.float32
        )

        for i, transition in enumerate(batch):

            valid_actions = transition[5]

            if len(valid_actions) > 0:

                next_action_masks[
                    i,
                    valid_actions
                ] = 1.0

        # ============================================================
        # GPU에서 Tensor 생성 + DQN 학습
        # ============================================================

        with tf.device(GPU_DEVICE):

            states = tf.convert_to_tensor(
                states,
                dtype=tf.float32
            )

            actions = tf.convert_to_tensor(
                actions,
                dtype=tf.int32
            )

            rewards = tf.convert_to_tensor(
                rewards,
                dtype=tf.float32
            )

            next_states = tf.convert_to_tensor(
                next_states,
                dtype=tf.float32
            )

            dones = tf.convert_to_tensor(
                dones,
                dtype=tf.float32
            )

            next_action_masks = tf.convert_to_tensor(
                next_action_masks,
                dtype=tf.float32
            )

            loss = self.train_step(
                states,
                actions,
                rewards,
                next_states,
                dones,
                next_action_masks
            )

        return loss
            
        ###############################################################
    @tf.function
    def train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones,
        next_action_masks
    ):

        with tf.device(GPU_DEVICE):

            with tf.GradientTape() as tape:

                # ========================================================
                # Current Q
                # ========================================================

                current_q = self.model.model(
                    states,
                    training=True
                )

                # ========================================================
                # Next Q - Online Network
                # ========================================================

                next_online_q = self.model.model(
                    next_states,
                    training=False
                )

                # ========================================================
                # Invalid action -> -inf
                # ========================================================

                masked_next_online_q = tf.where(
                    next_action_masks > 0.0,
                    next_online_q,
                    tf.constant(
                        -1e9,
                        dtype=tf.float32
                    )
                )

                # ========================================================
                # Best valid action
                # ========================================================

                best_next_actions = tf.argmax(
                    masked_next_online_q,
                    axis=1,
                    output_type=tf.int32
                )

                # ========================================================
                # Target Network
                # ========================================================

                next_target_q = self.target_model.model(
                    next_states,
                    training=False
                )

                batch_indices = tf.range(
                    tf.shape(next_target_q)[0],
                    dtype=tf.int32
                )

                indices = tf.stack(
                    [
                        batch_indices,
                        best_next_actions
                    ],
                    axis=1
                )

                max_next_q = tf.gather_nd(
                    next_target_q,
                    indices
                )

                targets = tf.where(
                    dones > 0.0,
                    rewards,
                    rewards + self.gamma * max_next_q
                )

                targets = tf.stop_gradient(
                    targets
                )
                
                # ========================================================
                # Current action Q만 추출
                # ========================================================

                current_action_q = tf.gather_nd(
                    current_q,
                    indices=tf.stack(
                        [
                            tf.range(
                                tf.shape(current_q)[0],
                                dtype=tf.int32
                            ),
                            actions
                        ],
                        axis=1
                    )
                )

                # ========================================================
                # Huber Loss
                # ========================================================

                loss = tf.reduce_mean(
                    tf.keras.losses.huber(
                        targets,
                        current_action_q
                    )
                )

            # ============================================================
            # Gradient
            # ============================================================

            gradients = tape.gradient(
                loss,
                self.model.model.trainable_variables
            )

            self.model.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.model.trainable_variables
                )
            )

        return loss
class Environment:
    def __init__(self, debug=False):
        self.board = None
        self.done = False
        self.winner = None

        self.nA = 132
        self.debug = debug

        # ==========================================
        # 학습 모드
        # ==========================================
        # "astar"   : DQN vs A*
        # "selfplay": DQN vs DQN
        self.training_mode = "astar"

        # astar 모드에서 DQN이 맡는 플레이어
        # 1 = P1
        # 2 = P2
        self.dqn_player = 1
        
    def reset(self, training_mode=None, dqn_player=None):
        # ==========================================
        # 새로운 보드 생성
        # ==========================================

        self.board = Quoridor.Quoridor()

        self.done = False
        self.winner = None

        # ==========================================
        # 새로운 게임의 학습 모드 설정
        # ==========================================

        if training_mode is not None:
            self.training_mode = training_mode

        if dqn_player is not None:
            self.dqn_player = dqn_player

        # ==========================================
        # 랜덤 초기 벽 배치
        # ==========================================

        self.setup_random_walls()

        # ==========================================
        # 초기 최단거리
        # ==========================================

        self.p1_distance = Quoridor.AStar(
            self.board,
            True
        )

        self.p2_distance = Quoridor.AStar(
            self.board,
            False
        )

        # ==========================================
        # 초기 State
        # ==========================================

        if self.training_mode == "selfplay":
            # self-play에서는 현재 행동할 플레이어 관점
            current_player = (
                1 if self.board.isP1Turn else 2
            )

            return self.get_state(
                player=current_player
            )

        else:
            # DQN vs A*에서는 항상 DQN 플레이어 관점
            return self.get_state(
                player=self.dqn_player
            )
            
    def get_state(self, player=None):
        """
        DQN 플레이어의 관점으로 보드를 표현한다.

        P1 관점:
            row 0 = 상대 진영
            row 8 = 내 시작 진영

        P2 관점:
            실제 보드를 180도 회전
            -> row 0 = 상대 진영
            -> row 8 = 내 시작 진영

        따라서 CNN 입장에서는 항상:

            channel 0 = 나
            channel 1 = 상대
            row 8      = 내 시작 위치
            row 0      = 상대 목표 방향

        이 된다.
        """

        state = np.zeros(
            (9, 9, 7),
            dtype=np.float32
        )

        # =========================================================
        # 현재 관점의 플레이어 결정
        # =========================================================

        if player is None:
            player = 1 if self.board.isP1Turn else 2

        # =========================================================
        # 실제 보드에서 내 위치 / 상대 위치
        # =========================================================

        if player == 1:

            my_pos = self.board.P1
            opponent_pos = self.board.P2

            my_walls = self.board.WallCount[0]
            opponent_walls = self.board.WallCount[1]

            rotate = False

        else:

            my_pos = self.board.P2
            opponent_pos = self.board.P1

            my_walls = self.board.WallCount[1]
            opponent_walls = self.board.WallCount[0]

            # P2는 P1과 반대 방향으로 진행하므로
            # CNN 입력에서는 보드를 180도 회전한다.
            rotate = True

        # =========================================================
        # 좌표 변환 함수
        # =========================================================

        def transform_position(pos):
            r = int(pos[0])
            c = int(pos[1])

            if rotate:
                return (
                    8 - r,
                    8 - c
                )

            return (
                r,
                c
            )

        # =========================================================
        # 플레이어 위치
        # =========================================================

        my_r, my_c = transform_position(my_pos)

        opponent_r, opponent_c = transform_position(
            opponent_pos
        )

        # =========================================================
        # Channel 0
        # 나
        # =========================================================

        state[
            my_r,
            my_c,
            0
        ] = 1.0

        # =========================================================
        # Channel 1
        # 상대
        # =========================================================

        state[
            opponent_r,
            opponent_c,
            1
        ] = 1.0

        # =========================================================
        # Channel 2
        # Horizontal Walls
        #
        # WallH[x]의 bit y
        #
        # 180도 회전:
        #
        # 원본 (x, y)
        #     ↓
        # 회전 (7-x, 7-y)
        #
        # =========================================================

        for x in range(8):

            for y in range(8):

                if self.board.WallH[x] & (1 << y):

                    if rotate:

                        new_x = 7 - x
                        new_y = 7 - y

                    else:

                        new_x = x
                        new_y = y

                    state[
                        new_x,
                        new_y,
                        2
                    ] = 1.0

        # =========================================================
        # Channel 3
        # Vertical Walls
        #
        # 원본 (x, y)
        #     ↓
        # 회전 (7-x, 7-y)
        #
        # =========================================================

        for x in range(8):

            for y in range(8):

                if self.board.WallVt[y] & (1 << x):

                    if rotate:

                        new_x = 7 - x
                        new_y = 7 - y

                    else:

                        new_x = x
                        new_y = y

                    state[
                        new_x,
                        new_y,
                        3
                    ] = 1.0

        # =========================================================
        # Channel 4
        # 현재 턴인지
        #
        # DQN 관점에서:
        #
        #   1 = 지금 내가 행동할 차례
        #   0 = 상대가 행동할 차례
        #
        # =========================================================

        is_my_turn = (
            (player == 1 and self.board.isP1Turn)
            or
            (player == 2 and not self.board.isP1Turn)
        )

        if is_my_turn:

            state[
                :,
                :,
                4
            ] = 1.0

        # =========================================================
        # Channel 5
        # 내 Wall Count
        # =========================================================

        state[
            :,
            :,
            5
        ] = my_walls / 10.0

        # =========================================================
        # Channel 6
        # 상대 Wall Count
        # =========================================================

        state[
            :,
            :,
            6
        ] = opponent_walls / 10.0

        return state


    
    def step(self, action):
        reward = 0.0
        acting_player = 1 if self.board.isP1Turn else 2

        # ==========================================
        # 행동 전 최단거리
        # ==========================================

        prev_p1_distance = self.p1_distance
        prev_p2_distance = self.p2_distance

        # ==========================================
        # 행동
        # ==========================================

        self.execute_action(action)

        # ==========================================
        # 행동 후 최단거리
        # ==========================================

        self.p1_distance = Quoridor.AStar(
            self.board,
            True
        )

        self.p2_distance = Quoridor.AStar(
            self.board,
            False
        )

        # ==========================================
        # 최단거리 변화
        # ==========================================

        p1_progress = (
            prev_p1_distance - self.p1_distance
        )

        p2_progress = (
            prev_p2_distance - self.p2_distance
        )

        # ==========================================
        # 기본 행동 비용
        # ==========================================

        if action >= 4:

            # 벽 설치 비용
            reward += -0.5

        else:
            reward += 0.5 #기본적으로 이동을 선호하게끔
            
            # 이동 비용 (상대와 나의 거리차가 클수록 이동이 더 유리하기 때문)
            if acting_player == 1:
                reward += 0.8 * (self.p2_distance - self.p1_distance)
                reward += 2 * (p1_progress)
            else:
                reward += 0.8* (self.p1_distance - self.p2_distance)
                reward += 2 * (p2_progress)
        # ==========================================
        # 거리 기반 보상
        # ==========================================

        if acting_player == 1:

            reward += (
                0.5 * p1_progress
                + 0.7 * (-p2_progress)
            )

        else:

            reward += (
                0.5 * p2_progress
                + 0.7 * (-p1_progress)
            )

        # ==========================================
        # 승리 확인
        # ==========================================

        winner = Quoridor.isGameOver(
            self.board
        )

        done = winner != 0

        if done:

            if winner == acting_player:

                reward += 50.0

            else:

                reward -= 50.0

            self.winner = winner
            self.done = True

        # ==========================================
        # 다음 상태
        # ==========================================

        next_state = self.get_state(player=acting_player)
        
        return next_state, reward, done

    def execute_action(self, action):

        # 이동
        if action < 4:

            move = action

            result = Quoridor.MovePlayer(
                self.board,
                move
            ) #상,하,좌,우 이동

        # 가로벽
        elif action < 68:

            wall_action = action - 4

            x = wall_action // 8
            y = wall_action % 8

            result = Quoridor.PlaceWallH(
                self.board,
                [x, y]
            )
        # 세로벽
        else:

            wall_action = action - 68

            x = wall_action // 8
            y = wall_action % 8

            result = Quoridor.PlaceWallV(
                self.board,
                [x, y]
            )

        return result

    def check_done(self):
        winner = Quoridor.isGameOver(self.board)
        if winner != 0:
            self.winner = winner
            self.done = True
            return True
        return False
    def get_valid_actions(self):
        return tuple(
            int(action)
            for action in Quoridor.GetValidActions(
                self.board
            )
        )

    def get_dqn_valid_actions(self, player):
        actual_actions = self.get_valid_actions()

        return tuple(
            transform_action(
                action,
                player
            )
            for action in actual_actions
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

def transform_action(action, player):
    """
    DQN 관점의 action을 실제 C++ 보드 action으로 변환한다.

    Action:
        0 = 위
        1 = 아래
        2 = 왼쪽
        3 = 오른쪽

        4~67   = Horizontal Wall
        68~131 = Vertical Wall
    """

    action = int(action)

    # =========================================================
    # P1
    # =========================================================

    if player == 1:
        return action

    # =========================================================
    # P2
    #
    # 보드를 180도 회전했으므로
    #
    # 위 <-> 아래
    # 왼쪽 <-> 오른쪽
    #
    # =========================================================

    if action == 0:
        return 1

    if action == 1:
        return 0

    if action == 2:
        return 3

    if action == 3:
        return 2

    # =========================================================
    # Wall action
    #
    # 180도 회전한 벽 action을 실제 보드 좌표로 변환
    # =========================================================

    if 4 <= action < 68:

        wall_action = action - 4

        x = wall_action // 8
        y = wall_action % 8

        new_x = 7 - x
        new_y = 7 - y

        return 4 + new_x * 8 + new_y

    if 68 <= action < 132:

        wall_action = action - 68

        x = wall_action // 8
        y = wall_action % 8

        new_x = 7 - x
        new_y = 7 - y

        return 68 + new_x * 8 + new_y

    raise ValueError(
        f"Invalid action: {action}"
    )

def get_astar_action(env, player):
    valid_actions = [
        int(action)
        for action in env.get_valid_actions()
        if int(action) < 4
    ]

    if not valid_actions:
        raise RuntimeError(
            f"A* Player {player} has no valid movement action."
        )

    astar_action = int(
        Quoridor.AStarNextMove(
            env.board,
            player == 1
        )
    )

    if astar_action in valid_actions:
        return astar_action

    if astar_action == -1:
        # 가장 단순한 안전 대체 동작
        return valid_actions[0]

    raise RuntimeError(
        f"Unexpected A* action={astar_action}, "
        f"valid_actions={valid_actions}, "
        f"player={player}"
    )
    
def get_astar_wall_action(env):
    """
    현재 A* 플레이어가 설치할 수 있는 벽 중
    하나를 랜덤하게 선택한다.

    반환:
        0 ~ 131 : 벽 action
        None    : 설치 가능한 벽이 없음
    """

    valid_actions = list(
        Quoridor.GetValidActions(env.board)
    )

    valid_wall_actions = [
        int(action)
        for action in valid_actions
        if 4 <= int(action) < 132
    ]

    if not valid_wall_actions:
        return None

    return random.choice(valid_wall_actions)
######################################################################

######################################################################
# Parallel environments
######################################################################

######################################################################
# Parallel Training
######################################################################

######################################################################
# Parallel Self-Play Training
######################################################################

NUM_ENVS = 128

envs = [
    Environment(debug=False)
    for _ in range(NUM_ENVS)
]

agent = DQNAgent(
    nS=567,
    nA=132,
    experiment_dir="./TESLA_Quoridor_AI"
)

EPISODES = 100000 #~8시간 소요

RESUME_FROM = "./TESLA_Quoridor_AI/dqn_episode_61000.weights.h5"

RESUME = True

if RESUME:
    if not os.path.exists(RESUME_FROM):

        raise FileNotFoundError(
            f"\nWeight 파일을 찾을 수 없습니다.\n"
            f"{os.path.abspath(RESUME_FROM)}"
        )

    # ============================================================
    # 기존 Online Network 가중치 불러오기
    # ============================================================

    agent.model.model.load_weights(
        RESUME_FROM
    )

    # ============================================================
    # Target Network도 동일한 가중치로 초기화
    # ============================================================

    agent.target_model.model.set_weights(
        agent.model.model.get_weights()
    )
    agent.epsilon = 0.3

    print(
        f"DQN model loaded successfully!\n"
        f"Weight: {os.path.abspath(RESUME_FROM)}",
        flush=True
    )
    envs
    
# ============================================================
# 128개 환경 초기화
# ============================================================

for i, env in enumerate(envs):

    env.training_mode = "astar"

    # DQN을 P1/P2 중 랜덤 배정
    env.dqn_player = random.choice([1, 2])

    env.reset(
        training_mode="astar",
        dqn_player=env.dqn_player
    )

# ============================================================
# 학습 시간 측정
# ============================================================

training_start_time = time.time()

total_training_steps = 0
finished_episodes = 0

REPLAY_EVERY_BATCHES = 1
REPLAY_UPDATES_PER_BATCH = 1

training_batch_count = 0

last_saved_episode = 0

episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

episode_p1_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
episode_p2_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

# ==========================================================
# 각 환경에서 P1/P2의 마지막 transition을 저장
# ==========================================================

pending_transitions = [
    {
        1: None,
        2: None
    }
    for _ in range(NUM_ENVS)
]
episode_steps = np.zeros(
    NUM_ENVS,
    dtype=np.int32
)

######################################################################
# Self-play / A* Training
######################################################################

ASTAR_WALL_THRESHOLD = 0.85

while finished_episodes < EPISODES:

    ##################################################################
    # 1. 살아있는 환경 수집
    ##################################################################

    active_indices = []
    active_states = []
    active_valid_actions = []

    for i, env in enumerate(envs):
        if env.done:
            continue

        active_indices.append(i)
        acting_player = (
            1 if env.board.isP1Turn else 2
        )

        if env.training_mode == "selfplay":
            dqn_player = acting_player
        else:
            dqn_player = env.dqn_player

        active_states.append(
            env.get_state(
                player=dqn_player
            )
        )

        active_valid_actions.append(
            env.get_dqn_valid_actions(
                dqn_player
            )
        )
        

    ##################################################################
    # 2. 행동 결정
    ##################################################################

    actions = [None] * len(active_indices)
    
    # Replay Memory에 저장할 DQN 관점 action
    dqn_memory_actions = [None] * len(active_indices)

    dqn_batch_indices = []
    dqn_batch_states = []
    dqn_batch_valid_actions = []

    for batch_index, env_index in enumerate(active_indices):

        env = envs[env_index]

        acting_player = (
            1 if env.board.isP1Turn else 2
        )

        # ============================================================
        # DQN 차례인지 확인
        # ============================================================

        if env.training_mode == "selfplay":

            is_dqn_turn = True

        else:

            is_dqn_turn = (
                acting_player == env.dqn_player
            )

        # ============================================================
        # DQN
        # ============================================================

        if is_dqn_turn:

            dqn_batch_indices.append(
                batch_index
            )

            dqn_batch_states.append(
                active_states[batch_index]
            )

            dqn_batch_valid_actions.append(
                active_valid_actions[batch_index]
            )

        # ============================================================
        # A*
        # ============================================================

        else:
            # ============================================================
            # A* 턴
            # ============================================================

            # 난수 생성
            random_value = random.random()

            # ------------------------------------------------------------
            # 일정 확률로 랜덤 벽 설치
            # ------------------------------------------------------------

            if random_value >= ASTAR_WALL_THRESHOLD:

                wall_action = get_astar_wall_action(env)

                if wall_action is not None:
                    actions[batch_index] = wall_action

                    if env.debug:
                        print(
                            f"[A* WALL] "
                            f"Player={acting_player} | "
                            f"Random={random_value:.4f} | "
                            f"Action={wall_action}",
                            flush=True
                        )

                else:
                    # 설치 가능한 벽이 없으면 A* 이동
                    actions[batch_index] = get_astar_action(
                        env,
                        acting_player
                    )

            # ------------------------------------------------------------
            # 나머지는 정상적인 A* 이동
            # ------------------------------------------------------------

            else:

                actions[batch_index] = get_astar_action(
                    env,
                    acting_player
                )

    ##################################################################
    # 3. DQN Batch Inference
    ##################################################################
    if len(dqn_batch_states) > 0:

        dqn_actions = agent.act_batch(
            dqn_batch_states,
            dqn_batch_valid_actions
        )

        for local_index, batch_index in enumerate(
            dqn_batch_indices
        ):

            env_index = active_indices[batch_index]

            env = envs[env_index]

            dqn_action = int(
                dqn_actions[local_index]
            )

            # Replay Memory에는 DQN 관점 action 저장
            dqn_memory_actions[batch_index] = dqn_action

            # 실제 플레이어
            if env.training_mode == "astar":

                dqn_player = env.dqn_player

            else:

                dqn_player = (
                    1 if env.board.isP1Turn
                    else 2
                )

            # DQN 관점 → 실제 C++ 보드 관점
            actions[batch_index] = transform_action(
                dqn_action,
                dqn_player
            )
            
    ##################################################################
    # 4. 행동 검증
    ##################################################################

    for batch_index, action in enumerate(actions):

        if action is None:

            raise RuntimeError(
                f"Action이 지정되지 않았습니다. "
                f"batch_index={batch_index}"
            )

    ##################################################################
    # 5. StepBatch용 데이터
    ##################################################################

    active_boards = [
        envs[i].board
        for i in active_indices
    ]

    active_acting_players = [
        1 if envs[i].board.isP1Turn else 2
        for i in active_indices
    ]

    active_prev_p1_distances = [
        envs[i].p1_distance
        for i in active_indices
    ]

    active_prev_p2_distances = [
        envs[i].p2_distance
        for i in active_indices
    ]

    ##################################################################
    # 6. C++ StepBatch
    ##################################################################

    rewards, dones, p1_distances, p2_distances = (
        Quoridor.StepBatch(
            active_boards,
            actions,
            active_acting_players,
            active_prev_p1_distances,
            active_prev_p2_distances
        )
    )

    ##################################################################
    # 7. 환경 결과 처리
    ##################################################################

    for batch_index, env_index in enumerate(
        active_indices
    ):
        env = envs[env_index]

        # ============================================================
        # 행동 정보
        # ============================================================

        acting_player = active_acting_players[batch_index]

        action = int(
            actions[batch_index]
        )

        reward = float(
            rewards[batch_index]
        )

        done = bool(
            dones[batch_index]
        )

        # ============================================================
        # 행동 직전 state
        #
        # active_states는 StepBatch 실행 전에 생성되었으므로
        # 여기서는 이것을 사용해야 한다.
        # ============================================================

        current_state = active_states[batch_index]
        # ============================================================
        # DQN 차례인지
        # ============================================================

        if env.training_mode == "selfplay":

            is_dqn_turn = True

        else:

            is_dqn_turn = (
                acting_player == env.dqn_player
            )

        # ============================================================
        # A* 거리 업데이트
        # ============================================================

        env.p1_distance = int(
            p1_distances[batch_index]
        )

        env.p2_distance = int(
            p2_distances[batch_index]
        )

        # ============================================================
        # Episode step
        # ============================================================

        episode_steps[env_index] += 1

        # ============================================================
        # 200턴 무승부
        # ============================================================

        is_draw = False

        if (
            not done
            and episode_steps[env_index] >= 200
        ):

            #DRAW_PENALTY = -80.0
            is_draw = True

            done = True

            env.done = True
            env.winner = 0

        # ============================================================
        # done 저장
        # ============================================================

        env.done = done

        # ============================================================
        # Reward 누적
        # ============================================================

        episode_rewards[
            env_index
        ] += reward

        if acting_player == 1:

            episode_p1_rewards[
                env_index
            ] += reward

        else:

            episode_p2_rewards[
                env_index
            ] += reward

        # ========================================================
        # DQN transition 임시 저장
        #
        # 중요:
        # 여기서는 next_state를 저장하지 않는다.
        #
        # 현재 DQN 행동
        #       ↓
        # 상대방 행동
        #       ↓
        # 다시 DQN 차례
        #
        # 가 되었을 때 next_state를 결정한다.
        # ========================================================

        if is_dqn_turn:

            old_pending = pending_transitions[env_index][acting_player]

            # ========================================================
            # 이전 DQN 행동을 이번 state에서 종료
            #
            # P1 행동
            #   ↓
            # 상대 행동
            #   ↓
            # 다시 P1 턴
            #
            # 이 시점의 current_state가 이전 transition의 next_state
            # ========================================================

            if old_pending is not None:

                agent.memory.add_memory(
                    old_pending["state"],
                    old_pending["action"],
                    old_pending["reward"],
                    current_state.copy(),
                    False,
                    tuple(active_valid_actions[batch_index])
                )

            # ========================================================
            # 현재 DQN 행동을 pending으로 저장
            # ========================================================

            pending_transitions[env_index][acting_player] = {
                "state": current_state.copy(),
                "action": dqn_memory_actions[batch_index],

                "p1_distance": int(
                    active_prev_p1_distances[batch_index]
                ),

                "p2_distance": int(
                    active_prev_p2_distances[batch_index]
                ),

                "reward": reward
            }

        # ============================================================
        # Episode 종료
        # ============================================================

        if done:

            if is_draw:

                winner = 0

            else:

                winner = Quoridor.isGameOver(
                    env.board
                )

            env.winner = winner
         
            # ========================================================
            # 게임 종료 시 DQN 최종 transition 처리
            # ========================================================

            if env.training_mode == "selfplay":

                # P1, P2 모두 DQN
                players_to_finalize = (1, 2)

            else:

                # DQN vs A*
                players_to_finalize = (env.dqn_player,)


            for player in players_to_finalize:

                pending = (
                    pending_transitions[
                        env_index
                    ][player]
                )

                # 해당 플레이어의 pending transition이 없으면 넘어감
                if pending is None:
                    continue

                # ====================================================
                # Terminal reward
                # ====================================================

                if winner == 0:

                    # 200턴 무승부
                    terminal_reward = -80.0

                elif winner == player:

                    # DQN 승리
                    terminal_reward = 50.0

                else:

                    # DQN 패배
                    terminal_reward = -50.0
                    
                episode_rewards[env_index] += terminal_reward

                if player == 1:
                    episode_p1_rewards[env_index] += terminal_reward
                else:
                    episode_p2_rewards[env_index] += terminal_reward

                # ====================================================
                # 기존 reward + terminal reward
                # ====================================================

                final_reward = (
                    pending["reward"]
                    + terminal_reward
                )

                # ====================================================
                # Terminal state
                #
                # 게임이 끝났으므로 다음 상태는 의미가 없음.
                # ====================================================

                terminal_state = np.zeros_like(
                    pending["state"]
                )

                # ====================================================
                # Replay Memory 저장
                # ====================================================

                agent.memory.add_memory(
                    pending["state"],
                    pending["action"],
                    final_reward,
                    terminal_state,
                    True,
                    ()
                )

                # ====================================================
                # Pending transition 제거
                # ====================================================

                pending_transitions[
                    env_index
                ][player] = None

            # ========================================================
            # Episode count
            # ========================================================

            finished_episodes += 1

            # ========================================================
            # 시간 계산
            # ========================================================

            elapsed_time = (
                time.time()
                - training_start_time
            )

            avg_episode_time = (
                elapsed_time
                / max(
                    1,
                    finished_episodes
                )
            )

            remaining_episodes = (
                EPISODES
                - finished_episodes
            )

            estimated_remaining_time = (
                avg_episode_time
                * remaining_episodes
            )

            remaining_hours = int(
                estimated_remaining_time
                // 3600
            )

            remaining_minutes = int(
                (estimated_remaining_time % 3600)
                // 60
            )

            remaining_seconds = int(
                estimated_remaining_time
                % 60
            )

            # ========================================================
            # 모델 저장
            # ========================================================

            if finished_episodes % 1000 == 0:

                save_path = (
                    "./TESLA_Quoridor_AI/"
                    f"dqn_episode_{finished_episodes}.weights.h5"
                )

                agent.model.model.save_weights(
                    save_path
                )

                print(
                    f"모델 저장 완료: {save_path}",
                    flush=True
                )

            # ========================================================
            # Winner
            # ========================================================

            if winner == 0:

                winner_text = "--"
            else:
                if env.training_mode == "astar":
                    if winner == env.dqn_player: 
                        winner_text = "DQN" 
                    else: winner_text = "A*"
                else:
                    winner_text = f"P{winner}"

            # ========================================================
            # Reward 표시
            # ========================================================

            display_p1_reward = (
                episode_p1_rewards[
                    env_index
                ]
            )

            display_p2_reward = (
                episode_p2_rewards[
                    env_index
                ]
            )

            # ========================================================
            # Mode
            # ========================================================

            if env.training_mode == "astar":

                mode_text = (
                    f"DQN(P{env.dqn_player}) vs A*"
                )

            else:

                mode_text = (
                    "DQN(P1) vs DQN(P2)"
                )

            # ========================================================
            # 출력
            # ========================================================

            print(
                f"Episode: "
                f"{finished_episodes}/{EPISODES} | "

                f"Mode: "
                f"{mode_text} | "

                f"Winner: "
                f"{winner_text} | "

                f"Total Reward: "
                f"{episode_rewards[env_index]:.2f} | "

                f"P1 Reward: "
                f"{display_p1_reward:.2f} | "

                f"P2 Reward: "
                f"{display_p2_reward:.2f} | "

                f"Steps: "
                f"{episode_steps[env_index]} | "

                f"Avg Reward/Step: "
                f"{episode_rewards[env_index] / max(1, episode_steps[env_index]):.2f} | "

                f"Epsilon: "
                f"{agent.epsilon:.4f} | "

                f"예상 남은 시간: "
                f"{remaining_hours:02d}:"
                f"{remaining_minutes:02d}:"
                f"{remaining_seconds:02d}",

                flush=True
            )

            # ========================================================
            # 다음 게임의 학습 방식
            # ========================================================

            if finished_episodes < 12000:

                new_training_mode = "astar"

                new_dqn_player = random.choice(
                    [1, 2]
                )

            else:

                new_training_mode = "selfplay"

                new_dqn_player = 1

            # ========================================================
            # 새 게임
            # ========================================================

            env.reset(
                training_mode=new_training_mode,
                dqn_player=new_dqn_player
            )

            # ========================================================
            # Reward 초기화
            # ========================================================

            episode_rewards[
                env_index
            ] = 0.0

            episode_p1_rewards[
                env_index
            ] = 0.0

            episode_p2_rewards[
                env_index
            ] = 0.0

            episode_steps[
                env_index
            ] = 0

    ##################################################################
    # 8. Batch 전체 처리 완료
    ##################################################################
    batch_step_count = len(
        active_indices
    )

    total_training_steps += (
        batch_step_count
    )

    training_batch_count += 1

    if (
        training_batch_count
        % REPLAY_EVERY_BATCHES
        == 0
    ):

        for _ in range(
            REPLAY_UPDATES_PER_BATCH
        ):

            loss = agent.replay()


    ##################################################################
    # 10. Target Network
    ##################################################################

    previous_step_count = (
        total_training_steps
        - batch_step_count
    )

    if (
        total_training_steps // 1000
        >
        previous_step_count // 1000
    ):

        agent.update_target_model()

        print(
            f"[TRAIN] "
            f"steps={total_training_steps} "
            f"memory={len(agent.memory)} "
            f"epsilon={agent.epsilon:.4f}",
            flush=True
        )

    ##################################################################
    # 11. Epsilon
    ##################################################################

    agent.epsilon_update()
 
###################################################################################
    
######################################################################
# 학습 완료
######################################################################

training_end_time = time.time()

total_training_time = training_end_time - training_start_time

# 평균 episode 시간
avg_episode_time = (
    total_training_time / finished_episodes
)

# 최종 모델 저장
final_model_path = (
    "./TESLA_Quoridor_AI/"
    "dqn_final.weights.h5"
)

agent.model.model.save_weights(
    final_model_path
)

######################################################################
# 시간 계산
######################################################################

hours = int(total_training_time // 3600)
minutes = int((total_training_time % 3600) // 60)
seconds = int(total_training_time % 60)

avg_hours = int(avg_episode_time // 3600)
avg_minutes = int((avg_episode_time % 3600) // 60)
avg_seconds = int(avg_episode_time % 60)

print(
    "\n"
    "============================================================\n"
    "                 TRAINING COMPLETE\n"
    "============================================================",
    flush=True
)

print(
    f"총 Episode        : {finished_episodes:,} / {EPISODES:,}",
    flush=True
)

print(
    f"총 Training Step  : {total_training_steps:,}",
    flush=True
)

print(
    f"최종 Epsilon      : {agent.epsilon:.6f}",
    flush=True
)

print(
    f"총 학습 시간      : "
    f"{hours:02d}시간 "
    f"{minutes:02d}분 "
    f"{seconds:02d}초",
    flush=True
)

print(
    f"평균 Episode 시간 : "
    f"{avg_hours:02d}시간 "
    f"{avg_minutes:02d}분 "
    f"{avg_seconds:02d}초",
    flush=True
)

print(
    f"최종 모델 저장    : {final_model_path}",
    flush=True
)

print(
    "============================================================",
    flush=True
)

print(
    "학습이 정상적으로 종료되었습니다.",
    flush=True
)