'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
'''


import numpy as np
import random
import gymnasium as gym

import pybind11
import sys
sys.path.append('../TESLA_Quoridor_AI/build/Debug')  # Add the path to TESLA_Quoridor_AI module
#C:\Users\yujoo\Desktop\TESLA\TESLA_Quoridor_AI\build\Debug
#C:\Users\yujoo\Desktop\TESLA\TESLA_Quoridor_AI
import example  # Import the C++ module
#print(Quoridor("Board"))

import Quoridor

Board = Quoridor.Quoridor()
Board.PrintBoard()

print(Board.P1)
print(Board.P2)
print(Board.WallCount)
print(Board.isP1Turn)
# Call the initialize_game function from the C++ module



#print(example.add(3, 4))  # Call the add function from the C++ module
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))



class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 9):
        # The size of the square grid (9x9 by default)
        self.size = size

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([-1, 0]),   #위쪽으로 이동 (행 -1)
            1: np.array([1, 0]),   #아래쪽으로 이동 (행 +1)
            2: np.array([0, -1]),   #왼쪽으로 이동 (열 -1)
            3: np.array([0, 1]),   #오른쪽으로 이동 (열 +1)
        }