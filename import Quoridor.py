import sys

sys.path.append('../TESLA_Quoridor_AI/build/Debug')  # Add the path to TESLA_Quoridor_AI module
sys.path.append('../TESLA_Quoridor_AI')
#print(Quoridor("Board"))

import Quoridor

import Quoridor

Board = Quoridor.Quoridor()
'''
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
'''


Board.PrintBoard()
print(Quoridor.GetValidActions(Board))