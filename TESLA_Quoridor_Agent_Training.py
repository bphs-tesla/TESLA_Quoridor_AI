
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import gymnasium as gym

import pybind11
import sys
sys.path.append('../TESLA_Quoridor_AI/build/Debug')  # Add the path to TESLA_Quoridor_AI module
#C:\Users\yujoo\Desktop\TESLA\TESLA_Quoridor_AI\build\Debug
#C:\Users\yujoo\Desktop\TESLA\TESLA_Quoridor_AI
#print(Quoridor("Board"))

import Quoridor
Board = Quoridor.Quoridor()

#Board.P1 = [0, 0]
WallH = [0,0]
print(Quoridor.PlaceWallH(Board, WallH))
for i in range(4):
     Quoridor.PlaceWallH(Board, WallH)
     WallH[1] += 2

Quoridor.PlaceWallH(Board, [1,7])
Quoridor.PlaceWallV(Board, [1,6])

#print(Quoridor.MovePlayer(Board, 255))
print(Board.P1)
#print(Board.P2)
#print(Board.WallCount)
print(Board.isP1Turn)
Board.PrintBoard()
print(Quoridor.AStar(Board))
print(Quoridor.canPlaceWallV(Board, [1, 6]))  # Example usage of the C++ function

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class Network:
    def __init__(self, nS, nA, scope = "estimator",
          learning_rate=0.0001,
          neural_architecture = None,
          global_step = None,
          summaries_dir = None
          ):
          self.nS = nS
          self.nA = nA
          self.global_step = global_step
          self.learning_rate = learning_rate

          if not neural_architecture:
              self.neural_architecture = self.two_layers_network
              
          with tf.variable_scope(scope):
               self.create_network(network = neural_architecture, learning_rate = self.learning_rate)
               if summaries_dir:
                    summary_dir = os.path.join(summaries_dir, "summaries_%s" % scope)
                    if not os.path.exists(summary_dir):
                         os.makedirs(summary_dir)
                    self.summary_writer = tf.summary.FileWriter(summary_dir)
               else:
                    self.summary_writer = None
                    
def two_layers_network(self, x, layer_1_nodes = 32, layer_2_nodes = 32):
    layer_1 = tf.contrib.layers.fully_connected(x, layer_1_nodes, activation_fn=tf.nn.leaky_relu)
    layer_2 = tf.contrib.layers.fully_connected(layer_1, layer_2_nodes, activation_fn=tf.nn.leaky_relu)
    return tf.contrib.layers.fully_connected(layer_2, self.nA, activation_fn=None)

def create_network(self, network, learning_rate = 0.0001):
    #상태 입력을 위한 플레이스홀더 정의
    self.X = tf.placeholder(shape = [None, self.nS], dtype = tf.float32, name = "X")
    #r 목표값
    self.y = tf.placeholder(shape = [None, self.nA], dtype = tf.float32, name = "y")
    
    #선택된 네트워크 적용
    self.predictions = network(self, self.X)
    
    #손실값 계산
    self.loss = tf.reduce_mean(tf.squared_difference(self.predictions, self.y))
    
    #Adam optimizer를 사용하여 손실값 최소화
    self.train_op = tf.contrib.layers.optimize_loss(
        loss = self.loss,
        global_step = tf.train.get_global_step(),
        learning_rate = learning_rate,
        optimizer = "Adam")
    
    #텐서보드를 위한 요약 정보 기록
    self.summaries = tf.summary.merge([
        tf.summary.scalar("loss", self.loss),
        tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions)),
        tf.summary.scalar("mean_q_value", tf.reduce_mean(self.predictions))])
    
def predict(self, sess, s):
    #주어진 상태 s에 대한 Q값 예측
    return sess.run(self.predictions, feed_dict = {self.X: s})

def fit(self, sess, s, r, epochs = 1):
    #Q 함수 추정값 업데이트
    feed_dict = {self.X: s, self.y: r}
    
    for epoch in range(epochs):
        res = sess.run([self.summaries, self.train_op, self.loss, self.predictions, tf.train.get_global_step()], feed_dict = feed_dict)
        summaries, train_op, loss, predictions, global_step = res
        
    if self.summary_writer:
        self.summary_writer.add_summary(summaries, global_step)
        
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
    
class DQNAgent:
    def __init__(self, nS, nA, experiment_dir):
        # 초기화
        self.nS = nS
        self.nA = nA
        self.epsilon = 1.0  # 초기 탐험률
        self.epsilon_min = 0.01  # 최소 탐험률
        self.epsilon_decay = 0.995  # 탐험률 감소율
        self.gamma = 0.99  # 할인율
        self.learning_rate = 0.0001  # 학습률
        self.epochs = 10  # 학습 에포크 수
        self.batch_size = 32  # 배치 크기
        self.memory = Memory(memory_size = 250000)  # 경험 재생 메모리 초기화
        
    #추정 모델 생성
        self.experiment_dir = os.path.abspath("./experiments/{}".format(experiment_dir))
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        self.model = Network(nS = self.nS,
            nA = self.nA,
            scope = "q",
            learning_rate = self.learning_rate,
            global_step = self.global_step,
            summaries_dir = self.experiment_dir)
        self.target_model = Network(nS = self.nS,
            nA = self.nA,
            scope = "target_q",
            learning_rate = self.learning_rate,
            global_step = self.global_step)
        
        #변수를 초기화하기 위한 박업 추가
        init_op = tf.global_variables_initializer()
        
        #모든 변수를 저징하고 복원하기 위한 작업 추가
        self.saver = tf.train.Saver()
        
        #세션 생성
        self.sess = tf.Session()
        self.sess.rn(init_op)
    
    def epsilon_update(self):
        #탐험률 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def save_weights(self, filename):
        #모델 가중치 저장
        save_path = self.saver.save(self.sess, "%s.ckpt" % filename)
        print("Model weights saved in path: %s" % save_path)
        
    def load_weights(self, filename):
        #모델 가중치 로드
        self.saver.restore(self.sess, "%s.ckpt" % filename)
        print("Model weights resotred from path: %s" % filename)

    def set_weights(self, model_1, model_2):
        #model_1: 
        #model_2: 
        model_1_params = [t for t in tf.trainable_variables() if t.name.startswith(model_1.scope)]
        model_2_params = [t for t in tf.trainable_variables() if t.name.startswith(model_2.scope)]
        model_1_params = sorted(model_1_params, key = lambda x: x.name)
        model_2_params = sorted(model_2_params, key = lambda x: x.name)
        
        #실행될 작업 열거
        operations = [coef_2.assign(coef_1) for coef_1, coef_2 in zip(model_1_params, model_2_params)]
        
        #실행될 작업 실행
        self.sess.run(operations)
        
    def target_model_update(self):
        #모델 가중치를 목표 모델의 가중치에 설정
        self.set_weights(self.model, self.target_model)
    
    def act(self, s):
        #엘실론 기반으로 에이전트가 학습된 Q* 함수를 기반으로 움직이거니 무작위 선택에 의해 움직이게 함
        #엡실론 기반으로 다음 행동을 예측하거나 임의로 다음 행동을 택함
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.nA)
        else:
            #모든 가능한 행동에 대한 q값 추정
            q = self.model.predict(self.sess, s)[0]
            #최선의 행동을 반환
            best_action = np.argmax(q)
            return best_action
        
    def replay(self):
        #메모리에서 랜덤 배치를 가져옴
        batch = np.array(random.sample(self.memory.recall_memories(), self.batch_size)
        #현재 상태의 시퀀스 가져옴
        s = np.vstack(batch[:, 0])
        
        #행동 시퀀스 재현
        a = np.array(batch[:, 1], dtype = int)
        
        #보상 재현
        r = np.copy(batch[:, 2])
        
        #결과 상태의 시퀀스 재현
        s_p = np.vstack(batch[:, 3])
        
        #보상이 최종 상태와 관련된 것인지 여부를 확인
        status = np.where(batch[:, 4] == False)
        
        #우리 모델과 목표 모델의 보상을 예측하기 위해 모델을 사용
        ''''
        
        
        
        미완성~~~~~~
        
        
        
        '''