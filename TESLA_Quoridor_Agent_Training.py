import os

# TensorFlow import 전에 설정
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, mixed_precision
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

mixed_precision.set_global_policy("mixed_float16")

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
            # DQN 신경망
            self.model = keras.Sequential([
                keras.layers.Input(shape=(9, 9, 7)),

                keras.layers.Conv2D(
                    64,
                    kernel_size=3,
                    padding="same",
                    activation="relu"
                ),

                keras.layers.Conv2D(
                    128,
                    kernel_size=3,
                    padding="same",
                    activation="relu"
                ),

                keras.layers.Conv2D(
                    128,
                    kernel_size=3,
                    padding="same",
                    activation="relu"
                ),

                keras.layers.Flatten(),

                Dense(512, activation="relu"),
                Dense(256, activation="relu"),

                Dense(nA, activation=None, dtype="float32")
            ])

        self.optimizer = Adam(
            learning_rate=learning_rate
        )

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

        return self.model.train_on_batch(s,target)
        

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

        self.gamma = 0.99

        self.learning_rate = 0.0001

        self.batch_size = 128

        self.memory = Memory(
            memory_size=250000
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
            print("\n========== NO VALID ACTION ==========")
            print(f"Environment: {i}")
            print(f"Board turn: {'P1' if env.board.isP1Turn else 'P2'}")
            print(f"P1 position: {env.board.P1}")
            print(f"P2 position: {env.board.P2}")
            print(f"P1 walls: {env.board.WallCount[0]}")
            print(f"P2 walls: {env.board.WallCount[1]}")
            print(f"State shape: {states[i].shape}")
            print(f"State min/max: {states[i].min()} / {states[i].max()}")
            print("====================================\n")

            raise RuntimeError(
                f"환경 {i}: Valid action이 없습니다."
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
            
    def get_valid_actions(self):
        return Quoridor.GetValidActions(self.Board)
    
        ###############################################################
    @tf.function(jit_compile=True)
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
    def __init__(self, debug = False):
        #환경 초기화
        self.board = None
        self.done = False
        self.winner = None
        self.nA = 132 #보드의 상태에서 취할 수 있는 행동의 최대 가짓수
        self.debug = debug
        
    def reset(self):
            self.board = Quoridor.Quoridor()

            self.done = False
            self.winner = None

            # 초기 최단거리
            self.p1_distance = Quoridor.AStar(
                self.board,
                True
            )

            self.p2_distance = Quoridor.AStar(
                self.board,
                False
            )

            return self.get_state()
        
    def get_state(self):
        state = np.zeros(
            (9, 9, 7),
            dtype=np.float32
        )

        # =========================================================
        # 현재 플레이어 기준으로 state를 표현
        # =========================================================

        if self.board.isP1Turn:

            # 나 = P1
            my_pos = self.board.P1
            opponent_pos = self.board.P2

            my_walls = self.board.WallCount[0]
            opponent_walls = self.board.WallCount[1]

        else:

            # 나 = P2
            my_pos = self.board.P2
            opponent_pos = self.board.P1

            my_walls = self.board.WallCount[1]
            opponent_walls = self.board.WallCount[0]


        # =========================================================
        # 현재 플레이어
        # =========================================================

        state[
            my_pos[0],
            my_pos[1],
            0
        ] = 1.0


        # =========================================================
        # 상대 플레이어
        # =========================================================

        state[
            opponent_pos[0],
            opponent_pos[1],
            1
        ] = 1.0


        # =========================================================
        # Horizontal walls
        # =========================================================

        for x in range(8):

            for y in range(8):

                if self.board.WallH[x] & (1 << y):

                    state[
                        x,
                        y,
                        2
                    ] = 1.0


        # =========================================================
        # Vertical walls
        # =========================================================

        for x in range(8):

            for y in range(8):

                if self.board.WallVt[y] & (1 << x):

                    state[
                        x,
                        y,
                        3
                    ] = 1.0


        # =========================================================
        # Turn
        #
        # 현재 플레이어 관점이므로 항상 1
        # =========================================================

        state[:, :, 4] = 1.0


        # =========================================================
        # Wall count
        # =========================================================

        state[:, :, 5] = (
            my_walls / 10.0
        )

        state[:, :, 6] = (
            opponent_walls / 10.0
        )


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

        next_state = self.get_state()

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
        return tuple(Quoridor.GetValidActions(self.board))
    
#####################################################################################################

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

EPISODES = 10000

# ============================================================
# 학습 시간 측정
# ============================================================

training_start_time = time.time()

total_training_steps = 0
finished_episodes = 0

last_saved_episode = 0

episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

episode_p1_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
episode_p2_rewards = np.zeros(NUM_ENVS, dtype=np.float32)

# ==========================================================
# 각 환경에서 P1/P2의 마지막 transition을 저장
# ==========================================================

last_transitions = [
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
# 초기화
######################################################################

states = []

for env in envs:

    states.append(
        env.reset()
    )


######################################################################
# Self-play
######################################################################

while finished_episodes < EPISODES:

    ##################################################################
    # 1. 현재 살아있는 환경 수집
    ##################################################################

    active_indices = []
    active_states = []
    active_valid_actions = []

    for i, env in enumerate(envs):

        if env.done:
            continue

        active_indices.append(i)

        active_states.append(
            states[i]
        )

        active_valid_actions.append(
            env.get_valid_actions()
        )


    ##################################################################
    # 모든 환경이 종료된 경우
    ##################################################################

    if len(active_indices) == 0:
        continue


    ##################################################################
    # 2. GPU에서 128개 환경의 action을 한 번에 계산
    ##################################################################

    actions = agent.act_batch(
        active_states,
        active_valid_actions
    )


    ##################################################################
    # 3. StepBatch용 데이터 생성
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
    # 4. C++ StepBatch
    #
    # 여기서 128개의 환경이 한 번에 C++로 들어감
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
    # 5. 각 환경의 결과 처리
    ##################################################################

    for batch_index, env_index in enumerate(active_indices):

        env = envs[env_index]

        state = states[env_index].copy()

        action = int(
            actions[batch_index]
        )

        reward = float(
            rewards[batch_index]
        )

        done = bool(
            dones[batch_index]
        )


        ################################################################
        # 행동한 플레이어
        ################################################################

        acting_player = active_acting_players[
            batch_index
        ]


        ################################################################
        # A* 거리 업데이트
        ################################################################

        env.p1_distance = int(
            p1_distances[batch_index]
        )

        env.p2_distance = int(
            p2_distances[batch_index]
        )


        ################################################################
        # Episode step 증가
        ################################################################

        episode_steps[env_index] += 1


        ################################################################
        # 200턴 무승부
        ################################################################

        is_draw = False

        if (
            not done
            and episode_steps[env_index] >= 200
        ):
            DRAW_PENALTY = -80.0

            is_draw = True

            # ==========================================================
            # 현재 행동한 플레이어
            # ==========================================================

            reward += DRAW_PENALTY

            done = True
            env.done = True
            env.winner = 0

            # ==========================================================
            # 상대 플레이어의 마지막 transition도
            # 무승부(-80) terminal transition으로 저장
            # ==========================================================

            opponent_player = 2 if acting_player == 1 else 1

            opponent_transition = (
                last_transitions[env_index][opponent_player]
            )

            if opponent_transition is not None:

                (
                    opp_state,
                    opp_action,
                    opp_reward,
                    opp_next_state,
                    opp_done,
                    opp_next_valid_actions
                ) = opponent_transition

                agent.memory.add_memory(
                    opp_state,
                    opp_action,
                    DRAW_PENALTY,
                    opp_next_state,
                    True,
                    ()
                )

        ################################################################
        # done 상태 반영
        ################################################################

        env.done = done


        ################################################################
        # 다음 상태
        ################################################################

        next_state = env.get_state()


        ################################################################
        # 다음 valid actions
        ################################################################

        if done:

            next_valid_actions = ()

        else:

            next_valid_actions = (
                env.get_valid_actions()
            )


        ################################################################
        # Reward 누적
        ################################################################

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


        ################################################################
        # Training step
        ################################################################

        total_training_steps += 1

        # ==========================================================
        # 현재 플레이어의 마지막 transition 저장
        # ==========================================================

        last_transitions[env_index][acting_player] = (
            state.copy(),
            action,
            reward,
            next_state.copy(),
            done,
            next_valid_actions
        )

        ################################################################
        # Replay Memory
        ################################################################
        
        agent.memory.add_memory(
            state,
            action,
            reward,
            next_state,
            done,
            next_valid_actions
        )


        ################################################################
        # 상태 업데이트
        ################################################################

        states[
            env_index
        ] = next_state


        ################################################################
        # Replay
        ################################################################

        if total_training_steps % 4 == 0:

            loss = agent.replay()


        ################################################################
        # Target Network
        ################################################################

        if total_training_steps % 1000 == 0:

            agent.update_target_model()

            print(
                f"[TRAIN] "
                f"steps={total_training_steps} "
                f"memory={len(agent.memory)} "
                f"epsilon={agent.epsilon:.4f}",
                flush=True
            )


        ################################################################
        # Winner 확인
        ################################################################

        if done:

            winner = Quoridor.isGameOver(env.board)

            env.winner = winner
        ################################################################
        # Episode 종료
        ################################################################

        if done:

            finished_episodes += 1

            elapsed_time = (
                time.time()
                - training_start_time
            )


            ################################################################
            # 평균 Episode 시간
            ################################################################

            avg_episode_time = (
                elapsed_time
                / max(1, finished_episodes)
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
                estimated_remaining_time % 60
            )


            ################################################################
            # 모델 저장
            ################################################################

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


            ################################################################
            # Winner
            ################################################################

            winner = env.winner

            if winner == 0:

                winner_text = "--"

            else:

                winner_text = f"P{winner}"


            ################################################################
            # Episode 결과
            ################################################################

            winner = env.winner

            if winner == 0:
                winner_text = "--"
            else:
                winner_text = f"P{winner}"
                
            display_p1_reward = episode_p1_rewards[env_index]
            display_p2_reward = episode_p2_rewards[env_index]
            
            if winner == 0:
                if acting_player == 1:
                    display_p2_reward -= 180.0
                else:
                    display_p1_reward -= 180.0

            print(
                f"Episode: "
                f"{finished_episodes}/{EPISODES} | "

                f"Winner: "
                f"{winner_text} | "

                f"Total Reward: "
                f"{episode_rewards[env_index]:.2f} | "

                f"P1 Reward: {display_p1_reward:.2f} | "
                f"P2 Reward: {display_p2_reward:.2f} | "

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


            ################################################################
            # 새 게임 시작
            ################################################################

            states[
                env_index
            ] = env.reset()


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
            
            # 새 게임이므로 이전 게임의 마지막 transition 삭제
            last_transitions[env_index][1] = None
            last_transitions[env_index][2] = None

    ##################################################################
    # Epsilon 감소
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