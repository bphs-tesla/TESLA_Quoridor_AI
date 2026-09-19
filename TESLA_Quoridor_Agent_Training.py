import os
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



#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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

        return self.model.predict(
            s,
            verbose=0
        )

    def fit(self, s, target):

        s = np.asarray(
            s,
            dtype=np.float32
        )

        target = np.asarray(
            target,
            dtype=np.float32
        )

        return self.model.fit(
            s,
            target,
            epochs=1,
            verbose=0
        )
        

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
        self.epsilon_decay = 0.995

        self.gamma = 0.99

        self.learning_rate = 0.0001

        self.batch_size = 32

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
    '''
        def act(self, s, valid_actions):
            # Exploration
            if np.random.rand() < self.epsilon:
                return random.choice(valid_actions)

            # Q-value 계산
            s = np.asarray(
                s,
                dtype=np.float32
            ).reshape(1, -1)

            q = self.model.predict(s)[0]

            # 유효하지 않은 행동 제거
            mask = np.full(
                self.nA,
                -np.inf,
                dtype=np.float32
            )

            mask[valid_actions] = 0.0

            q = q + mask

            return int(np.argmax(q))
    '''
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
        ).reshape(1, -1)

        # Q값
        q_values = self.model.predict(s)[0]

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
 
    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(
            self.batch_size
        )

        states = np.array(
            [x[0] for x in batch],
            dtype=np.float32
        )

        actions = np.array(
            [x[1] for x in batch],
            dtype=np.int32
        )

        rewards = np.array(
            [x[2] for x in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [x[3] for x in batch],
            dtype=np.float32
        )

        dones = np.array(
            [x[4] for x in batch],
            dtype=np.float32
        )

        # 현재 Q값
        current_q = self.model.predict(
            states
        )

        # 다음 상태의 Q값
        next_q = self.target_model.predict(
            next_states
        )

        # Bellman equation
        max_next_q = np.max(
            next_q,
            axis=1
        )

        targets = rewards + (
            1.0 - dones
        ) * self.gamma * max_next_q

        # 선택된 action의 Q값만 target으로 변경
        current_q[
            np.arange(self.batch_size),
            actions
        ] = targets

        # 학습
        self.model.fit(
            states,
            current_q
        )
    
    def get_valid_actions(self):
        return Quoridor.GetValidActions(self.Board)
        
class Environment:
    def __init__(self):
        #환경 초기화
        self.board = None
        self.done = False
        self.winner = None
        self.nA = 132 #보드의 상태에서 취할 수 있는 행동의 최대 가짓수
        
    def reset(self):
        #보드 초기화
        self.board = Quoridor.Quoridor()

        self.done = False
        self.winner = None

        return self.get_state()

    def get_state(self):

        state = np.zeros((9, 9, 7), dtype=np.float32)

        # P1
        state[self.board.P1[0], self.board.P1[1], 0] = 1.0

        # P2
        state[self.board.P2[0], self.board.P2[1], 1] = 1.0

        # Horizontal walls
        for x in range(8):
            for y in range(8):
                if self.board.WallH[x] & (1 << y):
                    state[x, y, 2] = 1.0

        # Vertical walls (transposed)
        for x in range(8):
            for y in range(8):
                if self.board.WallVt[y] & (1 << x):
                    state[x, y, 3] = 1.0

        # Turn
        if self.board.isP1Turn:
            state[:, :, 4] = 1.0
        else:
            state[:, :, 4] = -1.0

        # Remaining walls
        state[:, :, 5] = self.board.WallCount[0] / 10.0 #P1 WallCount / 10
        state[:, :, 6] = self.board.WallCount[1] / 10.0 #P2 WallCount / 10

        return state.flatten() #nS = 9*9*7 = 567

    def step(self, action):
        if self.done:
            raise RuntimeError("Game is already finished.")

        # 행동 전 최단경로
        prev_p1_distance = Quoridor.AStar(self.board, True)
        prev_p2_distance = Quoridor.AStar(self.board, False)

        # 행동 실행
        result = self.execute_action(action)

        # 행동 후 최단경로
        new_p1_distance = Quoridor.AStar(self.board, True)
        new_p2_distance = Quoridor.AStar(self.board, False)

        # P2가 AI
        # P2의 경로가 짧아지면 +
        my_reward = prev_p2_distance - new_p2_distance

        # P1의 경로가 길어지면 +
        opponent_reward = new_p1_distance - prev_p1_distance

        reward = (
            0.5 * my_reward
            + 0.5 * opponent_reward
        )

        # 승패 보상
        winner = Quoridor.isGameOver(self.board)

        if winner == 2:
            reward += 10.0

        elif winner == 1:
            reward -= 10.0

        done = self.check_done()

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

    def calculate_reward(self, prev_my_distance, prev_opponent_distance):
        winner = Quoridor.isGameOver(self.board)

        # 승패 보상
        if winner == 2:
            return 10.0   # P2 승리

        elif winner == 1:
            return -10.0  # P2 패배

        # 현재 최단 경로
        my_distance = Quoridor.AStar(self.board, 2)
        opponent_distance = Quoridor.AStar(self.board, 1)

        # 내 경로가 줄어든 만큼 보상
        my_reward = prev_my_distance - my_distance

        # 상대 경로가 늘어난 만큼 보상
        opponent_reward = opponent_distance - prev_opponent_distance

        # 가중치
        reward = (
            0.5 * my_reward
            + 0.5 * opponent_reward
        )

        return reward

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
env = Environment()

agent = DQNAgent(nS=567, nA=132, experiment_dir="./TESLA_Quoridor_AI")

EPISODES = 20
for episode in range(EPISODES):
    print(f"\n===== Episode {episode} 시작 =====", flush=True)
    if (episode + 1) % 10 == 0:
        save_path = (
            f"./TESLA_Quoridor_AI/"
            f"dqn_episode_{episode + 1}.weights.h5"
        )

        agent.model.model.save_weights(
            save_path
        )

        print(
            f"모델 저장 완료: {save_path}",
            flush=True
        )

    state = env.reset()

    total_reward = 0
    step_count = 0

    while True:
        #if step_count % 10 == 0:
            #print(f"Episode {episode}, step {step_count}",flush=True)

        valid_actions = env.get_valid_actions()

        #print("valid actions:", valid_actions, flush=True)

        action = agent.act(
            state,
            valid_actions
        )

        #print("action:", action, flush=True)

        next_state, reward, done = env.step(action)

        #print(f"reward={reward}, done={done}",flush=True)

        agent.memory.add_memory(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward
        step_count += 1

        agent.replay()

        if done:
            break

        # 임시 안전장치
        if step_count >= 200:
            print(
                "200 step 초과, episode 강제종료",
                flush=True
            )
            break
        
        

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    if episode % 10 == 0:
        agent.update_target_model()

    print(
        f"Episode: {episode} | "
        f"Reward: {total_reward} | "
        f"Epsilon: {agent.epsilon:.4f} | "
        f"Steps: {step_count}",
        flush=True
    )
    