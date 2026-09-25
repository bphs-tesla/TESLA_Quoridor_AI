import sys

sys.path.append('../TESLA_Quoridor_AI/build/Debug')  # Add the path to TESLA_Quoridor_AI module
#sys.path.append('../TESLA_Quoridor_AI')
#print(Quoridor("Board"))
sys.path.insert(0, './build-linux')

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
Quoridor.PlaceWallH(Board, [4,4])
Quoridor.PlaceWallH(Board, [1,0])
Quoridor.PlaceWallH(Board, [0,1])
Quoridor.PlaceWallV(Board, [1,1])
Quoridor.PlaceWallH(Board, [1,2])
Quoridor.PlaceWallH(Board, [0,4])
Quoridor.PlaceWallV(Board, [0,5])
Quoridor.PlaceWallH(Board, [0,6])
Quoridor.PlaceWallV(Board, [1,7])
Quoridor.PlaceWallH(Board, [2,7])
Quoridor.PlaceWallV(Board, [1,4])
Quoridor.PlaceWallV(Board, [2,3])
Quoridor.PlaceWallV(Board, [3,4])
Board.P1 = [4,4]
Board.P2 = [2,4]


Board.PrintBoard()
print(Quoridor.AStarNextMove(Board, False))
print(Quoridor.GetMoveDestination(Board, [3, 4], 1, [4, 5], False))
#print(Quoridor.GetValidActions(Board))