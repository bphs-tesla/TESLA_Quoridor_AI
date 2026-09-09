#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

struct Quoridor
{
	//쿼리도 기본 시작 상태로 initialize됨
	//쿼리도 초기 게임의 그래프를 선언 (81개의 상태 노드를 가진 그래프라고 해석시, 81*81차원의 행렬과 동치로 취급 가능)
	bool Path[9][9][4]; //Path[행][열][상/하/좌/우] 형식으로 배열 저장
	uint8_t WallH[8] = { NULL, }; //이거는 가로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
	uint8_t WallVt[8] = { NULL, }; //이거는 세로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
	/*
	WallH는 가로로 벽을 설치하기에, 상/하 움직임을 불가능하게 한다. ( k=0,1 )
	WallV는 세로로 벽을 설치하기에, 좌/우 움직임을 불가능하게 한다. ( k=2,3 )
	*/

	//위의 WallH나 WallV는 사실은 8*8 bool 행렬이지만, 
	//벽의 설치 여부가 보드의 유효성 판정에만 중요하기에, 그냥 bitpacking을 이용해, 8개의 열을 1 Byte의 char 로 압축했다. 
	//즉, Cpp의 std::vector<bool>랑 비슷한 방식으로 만들었다.
	// 
	//**이 이유때문에, (WallV[행] & (1 << 열{왼쪽부터 0 ~ 7}))이런 식으로 판정해야 함 ㅎㅎ**
	Quoridor() {
		//Quoridor 초기 보드 상태 설정
		std::fill(&Path[0][0][0], &Path[8][8][3], 1);
		for (int i = 0; i < 9; i++) {
			Path[0][i][0] = false; //맨 위쪽 줄에서 위쪽으로 갈수 없게
			Path[8][i][1] = false; //맨 아래쪽 줄에서 아래쪽으로 갈수 없게
			Path[i][0][2] = false; //맨 왼쪽 줄에서 왼쪽으로 갈수 없게
			Path[i][8][3] = false; //맨 오른쪽 줄에서 오른쪽으로 갈수 없게
		}
	}
	uint8_t P1[2] = { 8,4 }; //P1의 보드 위의 행/열
	uint8_t P2[2] = { 0,4 }; //P2의 보드 위의 행/열
	uint8_t WallCount[2] = { 10,10 }; //P1,P2의 남은 벽 개수(10이 기본이고, 벽을 쓸수록 감소함)
	bool isP1Turn = true; // P1의 차례이면, true, P2의 차례이면 false (P1이 선공이라서 true가 기본)

};

int PlaceWallH(Quoridor* Board, uint8_t Pos[2]) {
	//가로로 벽을 배치하는 코드 ( k=0,1 )
	//불가능하면 1 반환, 가능하면 0 반환
	if (Pos[0] >= 8) {return 1;} //유효하지 않은 좌표 예외 처리
	if (!((Board->WallH)[Pos[0]] & (1 << (Pos[1] - 1)))) {
		//Pos[0], Pos[1] - 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
		if (!((Board->WallH)[Pos[0]] & (1 << (Pos[1] + 1)))) {
			//Pos[0], Pos[1] + 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
			//벽 설치 코드
			(Board->WallH)[Pos[0]] = (Board->WallH)[Pos[0]] | (1 << (Pos[1])); //WallH 업데이트
			//Path 업데이트
			(Board->Path)[Pos[0] + 1][Pos[1]][0] = false;
			(Board->Path)[Pos[0] + 1][Pos[1] + 1][0] = false;
			(Board->Path)[Pos[0]][Pos[1]][1] = false;
			(Board->Path)[Pos[0]][Pos[1] + 1][1] = false;
			return 0; //정상적으로 Path & WallH 업데이트
		}
	}
	return 1; //예외 처리
}

int PlaceWallVt(Quoridor* Board, uint8_t Pos[2]) {
	//세로로 벽을 배치하는 코드 ( k=2,3 )
	//불가능하면 1 반환, 가능하면 0 반환
	if (Pos[0] >= 8) { return 1; } //유효하지 않은 좌표 예외 처리
	if (!((Board->WallVt)[Pos[1]] & (1 << (Pos[0] - 1)))) {
		//Pos[0], Pos[1] - 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
		if (!((Board->WallVt)[Pos[1]] & (1 << (Pos[0] + 1)))) {
			//Pos[0], Pos[1] + 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
			//벽 설치 코드
			(Board->WallVt)[Pos[1]] = (Board->WallVt)[Pos[1]] | (1 << (Pos[0])); //WallV 업데이트
			//Path 업데이트
			(Board->Path)[Pos[1]][Pos[0] + 1][2] = false;
			(Board->Path)[Pos[1] + 1][Pos[0] + 1][2] = false;
			(Board->Path)[Pos[1]][Pos[0]][3] = false;
			(Board->Path)[Pos[1] + 1][Pos[0]][3] = false;
			return 0; //정상적으로 Path & WallV 업데이트
		}
	}
	return 1; //예외 처리
}



void PrintBoard(struct Quoridor Board) {
	//Quoridor를 받아서, 그에 대응되는 쿼리도 보드를 출력함
	putchar('+'); for (int _ = 0; _ < 53; _++) { putchar('-'); }printf("+\n");

	for (int i = 0; i < 35; i++) {
		printf("|");
		if ((i % 4) == 3) {
			 //가로벽 설치
			for (int j = 0; j < 53; j++) {
				if ((j % 6) == 5) {
					printf("+");
				}
				else if (Board.WallH[i / 6] & (1 << (j))) {
					printf("-");
				}
				else { printf(" "); }
			}
		}
		else {
			//세로벽 설치
			for (int j = 0; j < 53; j++) {
				if ((j % 6) == 5) {
					if (Board.WallVt[i / 12] & (1 << (j / 12))) {
						printf("|");
					}
					else { printf(" "); }
				}else { printf(" "); }
			}
		}
		printf("|\n");
	}
	putchar('+'); for (int _ = 0; _ < 53; _++) { putchar('-'); }printf("+\n");

}

int HeuristicEstimation(Quoridor* Board) {
	/*
	if (Board->isP1Turn) {
		//P1의 차례
	}
	*/
	return 0; //이거 원래 HeuristicEstimation은 보드의 상태(의 주소를 간접참조해) 보드의 상태를 평가하는 함수
	/*재민이 화이팅 ㅎㅎ*/
}

int MakeChildNode(Quoridor Board, int SearchTreeDepth) {
	if (SearchTreeDepth <= 0) {
		//탐색 깊이 최대 도달
		return HeuristicEstimation(&Board);
	}
	/*
	
	마법

	*/

	

	
}


int main() {
	struct Quoridor Board;
	uint8_t H[2] = { 0,0 };
	uint8_t H2[2] = { 1,2 };
	uint8_t V[2] = { 1,2 };
	//uint8_t V[2] = { 3,4 };
	PlaceWallH(&Board, H);
	PlaceWallH(&Board, H2);
	PlaceWallVt(&Board, V);
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
			if ((Board.WallVt[i] & (1 << j))) {
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
	//findLegalMoves(StartState, StartWallCount, false);//쿼리도 보드에서 합법척인 수 찾기

	return 0;
}