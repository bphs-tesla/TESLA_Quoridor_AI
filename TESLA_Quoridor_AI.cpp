#include <Quoridor.h>

int main() {
	struct Quoridor Board;
	uint8_t H[2] = { 2,7 };
	uint8_t H2[2] = { 0,2 };
	uint8_t V[2] = { 1,7 };
	uint8_t V2[2] = { 2,3 };
	//uint8_t V[2] = { 3,4 };
	PlaceWallH(&Board, H);
	PlaceWallV(&Board, V);
	PlaceWallH(&Board, H2);
	PlaceWallV(&Board, V2);
	uint8_t GridWallPos[2] = { 0,0 };
	for (int i = 0; i < 8; i++, GridWallPos[1]++) {
		PlaceWallH(&Board, GridWallPos);
	}
	//PlaceWallVt(&Board, V2);
	printf("Distance: %d\n\n", AStar(&Board, Board.isP1Turn));
	Board.PrintBoard();
	switch (isConnected(&Board)) {
	case false:
		printf("0\n");
		break;
	case true:
		printf("1\n");
		break;
	}
	/*

		switch (Board.isLegalMove(0b11010011)) {
	case false:
		printf("0\n");
		break;
	case true:
		printf("1\n");
		break;
	}
	

	//PlaceWallV(&Board, V);

	for (int k = 0; k < 4; k++) { //갈수있는 방향
		printf("\nk = %d\n", k);
		for (int i = 0; i < 9; i++) { //행
			printf("[ ");
			for (int j = 0; j < 9; j++) { //열
				switch (Board.Path[i][j][k]) {
				case 1:
					printf("1 ");
					break;
				case 0:
					printf("0 ");
					break;
				}
			}
			printf("]\n");
		}
	}
	printf("\nWallH\n");
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if ((Board.WallH[i] & (1 << j))) {
				printf("1 ");
			}
			else {
				printf("0 ");
			}
		}
		printf("\n");
	}

	printf("\nWallVt\n");
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			if ((Board.WallVt[j] & (1 << i))) {
				printf("1 ");
			}
			else {
				printf("0 ");
			}
		}
		printf("\n");
	}
	//printf("Path Size: %d", (int)sizeof(Board.Path));

	//PrintBoard(Board);//쿼리도 보드 출력
	//findLegalMoves(StartState, StartWallCount, false);//쿼리도 보드에서 합법적인 수 찾기
	*/
	return 0;
}