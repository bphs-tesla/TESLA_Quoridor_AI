import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

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



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
print("Build info:", tf.sysconfig.get_build_info())

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

        # DQN 신경망
        self.model = keras.Sequential([
            keras.layers.Input(shape=(nS,)),

            Dense(
                256,
                activation="relu"
            ),

            Dense(
                256,
                activation="relu"
            ),

            Dense(
                nA,
                activation=None
            )
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
    
    def add_memory(self, s, a, r, s_, status):
        #경험을 메모리에 추가
        self.memory.append((s, a, r, s_, status))
    
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
        ).reshape(1, -1)

        # Q값
        #q_values = self.model.predict(s)[0]
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
        여러 환경의 P2 상태를 한 번에 GPU에 넣어서 행동을 결정한다.

        states:
            list[np.ndarray]
            shape = (N, 567)

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
            return

        batch = self.memory.sample(
            self.batch_size
        )

        states = tf.convert_to_tensor(
            np.asarray(
                [x[0] for x in batch],
                dtype=np.float32
            )
        )

        actions = tf.convert_to_tensor(
            np.asarray(
                [x[1] for x in batch],
                dtype=np.int32
            )
        )

        rewards = tf.convert_to_tensor(
            np.asarray(
                [x[2] for x in batch],
                dtype=np.float32
            )
        )

        next_states = tf.convert_to_tensor(
            np.asarray(
                [x[3] for x in batch],
                dtype=np.float32
            )
        )

        dones = tf.convert_to_tensor(
            np.asarray(
                [x[4] for x in batch],
                dtype=np.float32
            )
        )

        loss = self.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

        return loss
            
    def get_valid_actions(self):
        return Quoridor.GetValidActions(self.Board)
    
    ###############################################################
    @tf.function
    def train_step(
        self,
        states,
        actions,
        rewards,
        next_states,
        dones):
        with tf.GradientTape() as tape:
            current_q = self.model.model(
                states,
                training=True
            )

            next_q = self.target_model.model(
                next_states,
                training=False
            )

            max_next_q = tf.reduce_max(
                next_q,
                axis=1
            )

            targets = (
                rewards
                - (1.0 - dones)
                * self.gamma
                * max_next_q
            )

            action_mask = tf.one_hot(
                actions,
                self.nA
            )

            target_q = (
                current_q * (1.0 - action_mask)
                + tf.expand_dims(targets, 1)
                * action_mask
            )

            loss = tf.reduce_mean(
                tf.square(current_q - tf.stop_gradient(target_q))
            )

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


        return state.flatten()
    
    def step(self, action):
        acting_player = 1 if self.board.isP1Turn else 2

        # 행동 전
        prev_p1_distance = self.p1_distance
        prev_p2_distance = self.p2_distance

        # 행동 실행
        self.execute_action(action)

        if (action > 4):
            #AI가 벽을 설치했을때
            reward -= 0.5 #한턴을 소모하는 것이기에, 기본적으로 차감
            # 행동 후
            self.p1_distance = Quoridor.AStar(
                self.board,
                True
            )

            self.p2_distance = Quoridor.AStar(
                self.board,
                False
            )

            # 변화량
            p1_progress = (
                prev_p1_distance - self.p1_distance
            )

            p2_progress = (
                prev_p2_distance - self.p2_distance
            )

            # 보상
            if acting_player == 1:
                reward = (
                    0.5 * p1_progress
                    + 0.5 * (-p2_progress)
                )

            else:
                reward = (
                    0.5 * p2_progress
                    + 0.5 * (-p1_progress)
                )
        else:
            reward -= 0.2
            if (acting_player == 1):
                reward += 0.3*(prev_p2_distance - prev_p1_distance)
            else:
                reward -= 0.3*(prev_p2_distance - prev_p1_distance)
            #플레이어가 움직일때, 상대보다 멀때는 패널티를, 상대보다 가까울때는 보상을 줌.
        
        # 게임 종료
        winner = Quoridor.isGameOver(
            self.board
        )

        done = winner != 0

        if done:

            if winner == acting_player:
                reward += 10.0
            else:
                reward -= 10.0

            self.winner = winner
            self.done = True

        # 디버깅
        if self.debug:

            print(
                f"P{acting_player} | "
                f"Action={action} | "
                f"P1: {prev_p1_distance}->{self.p1_distance} | "
                f"P2: {prev_p2_distance}->{self.p2_distance} | "
                f"P1 progress={p1_progress:+d} | "
                f"P2 progress={p2_progress:+d} | "
                f"Reward={reward:+.2f}",
                flush=True
            )

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

EPISODES = 50000

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
    # 현재 살아있는 모든 환경을 모음
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
    # 동시에 DQN으로 행동 결정
    ##################################################################

    if len(active_indices) == 0:
        continue

    actions = agent.act_batch(
        active_states,
        active_valid_actions
    )


    ##################################################################
    # 각 환경에서 행동 실행
    ##################################################################

    for batch_index, env_index in enumerate(
        active_indices
    ):

        env = envs[
            env_index
        ]

        state = states[
            env_index
        ].copy()

        action = actions[
            batch_index
        ]


        ##################################################################
        # 행동한 플레이어 확인
        ##################################################################

        acting_player = 1 if env.board.isP1Turn else 2


        ##################################################################
        # 행동
        ##################################################################

        next_state, reward, done = env.step(
            action
        )


        ##################################################################
        # 에피소드 보상 누적
        ##################################################################

        # 전체 보상
        episode_rewards[
            env_index
        ] += reward

        # 플레이어별 보상
        if acting_player == 1:

            episode_p1_rewards[
                env_index
            ] += reward

        else:

            episode_p2_rewards[
                env_index
            ] += reward


        ##################################################################
        # Replay Memory
        ##################################################################

        agent.memory.add_memory(
            state,
            action,
            reward,
            next_state,
            done
        )
        ##################################################################
        # 상태 업데이트
        ##################################################################

        states[
            env_index
        ] = next_state

        episode_steps[
            env_index
        ] += 1

        total_training_steps += 1


        ##################################################################
        # Replay
        ##################################################################

        if total_training_steps % 4 == 0:

            loss = agent.replay()


        ##################################################################
        # Target Network
        ##################################################################

        if total_training_steps % 1000 == 0:

            agent.update_target_model()

            print(
                f"[TRAIN] "
                f"steps={total_training_steps} "
                f"memory={len(agent.memory)} "
                f"epsilon={agent.epsilon:.4f}",
                flush=True
            )


        ##################################################################
        # Episode 종료
        ##################################################################

        if done:
            ##################################################################
            # 모델 저장
            ##################################################################
            finished_episodes += 1
            elapsed_time = time.time() - training_start_time

            if finished_episodes > 0:

                avg_episode_time = (
                    elapsed_time / finished_episodes
                )

                remaining_episodes = (
                    EPISODES - finished_episodes
                )

                estimated_remaining_time = (
                    avg_episode_time * remaining_episodes
                )

                remaining_hours = int(
                    estimated_remaining_time // 3600
                )

                remaining_minutes = int(
                    (estimated_remaining_time % 3600) // 60
                )

                remaining_seconds = int(
                    estimated_remaining_time % 60
                )
            if finished_episodes % 5000 == 0:
                save_path = (
                    "./TESLA_Quoridor_AI/"
                    f"dqn_episode_{finished_episodes}.weights.h5"
                )

                agent.model.model.save_weights(save_path)

                print(
                    f"모델 저장 완료: {save_path}",
                    flush=True
                )

            winner = env.winner

            print(
                f"Episode: {finished_episodes} | "
                f"Winner: P{winner} | "
                f"Total Reward: "
                f"{episode_rewards[env_index]:.2f} | "
                f"P1 Reward: "
                f"{episode_p1_rewards[env_index]:.2f} | "
                f"P2 Reward: "
                f"{episode_p2_rewards[env_index]:.2f} | "
                f"Steps: "
                f"{episode_steps[env_index]} | "
                f"Avg Reward/Step: "
                f"{episode_rewards[env_index] / max(1, episode_steps[env_index]):.2f} | "
                f"Epsilon: "
                f"{agent.epsilon:.4f}",
                f"예상 남은 시간: "
                f"{remaining_hours:02d}: "
                f"{remaining_minutes:02d}: "
                f"{remaining_seconds:02d}",
                flush=True
            )


            ##################################################################
            # 새 게임
            ##################################################################

            states[
                env_index
            ] = env.reset()

            episode_rewards[
                env_index
            ] = 0
            
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
        # 너무 긴 게임 방지
        ##################################################################

        elif episode_steps[
            env_index
        ] >= 200:
            elapsed_time = time.time() - training_start_time

            if finished_episodes > 0:

                avg_episode_time = (
                    elapsed_time / finished_episodes
                )

                remaining_episodes = (
                    EPISODES - finished_episodes
                )

                estimated_remaining_time = (
                    avg_episode_time * remaining_episodes
                )

                remaining_hours = int(
                    estimated_remaining_time // 3600
                )

                remaining_minutes = int(
                    (estimated_remaining_time % 3600) // 60
                )

                remaining_seconds = int(
                    estimated_remaining_time % 60
                )
            finished_episodes += 1
            print(
                f"Episode: {finished_episodes} | "
                f"Winner: None | "
                f"Total Reward: "
                f"{episode_rewards[env_index]:.2f} | "
                f"P1 Reward: "
                f"{episode_p1_rewards[env_index]:.2f} | "
                f"P2 Reward: "
                f"{episode_p2_rewards[env_index]:.2f} | "
                f"Steps: "
                f"{episode_steps[env_index]} | "
                f"Avg Reward/Step: "
                f"{episode_rewards[env_index] / max(1, episode_steps[env_index]):.2f} | "
                f"Epsilon: "
                f"{agent.epsilon:.4f}",
                f"예상 남은 시간: "
                f"{remaining_hours:02d}: "
                f"{remaining_minutes:02d}: "
                f"{remaining_seconds:02d}",
                flush=True
            )

            states[
                env_index
            ] = env.reset()

            episode_rewards[
                env_index
            ] = 0
            
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