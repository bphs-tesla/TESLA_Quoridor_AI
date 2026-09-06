#include <stdio.h>
#include <stdlib.h>

void printBoard(char State[17][17], char WallCount[2]) {
	//17*17 char 배열(State)과 WallCount(남은 벽의 수)를 받아서, 그에 대응되는 쿼리도 보드를 출력함
	printf("+"); for (int _ = 0; _ < 17 * 3; _++) { printf("-"); }printf("+\n");
	for (int i = 0; i < 17; i ++) {
		printf("|");
		if (i % 2 != 0) {
			//행이 벽만 있는 때 
			for (int j = 0; j < 17; j ++) {
				if (j % 2 != 0) {
					printf(" + ");
					continue;
				}
				switch (State[i][j]) {
				case 0x00://빈칸
					printf("   ");
					break;
				case 'W'://벽
					printf(" = ");
					break;
				default:
					printf(" X ");
					break;
				}

			}
		}
		else {
			for (int _ = 0; _ < 17 * 3; _++) { printf(" "); }printf("|\n|");
			for (int j = 0; j < 17; j++) {
				switch (State[i][j]) {
				case 0x00://빈칸
					printf("   ");
					break;
				case 1://플레이어 1
					printf(" 1 ");
					break;
				case 2://플레이어 2
					printf(" 2 ");
					break;
				case 'W'://벽
					printf(" | ");
					break;
				default:
					printf(" X ");
					break;
				}
			}
			
			printf("|\n|"); for (int _ = 0; _ < 17 * 3; _++) { printf(" "); }
		}

		printf("|\n");

	}
	printf("+"); for (int _ = 0; _ < 17 * 3; _++) { printf("-"); }printf("+\n");
	printf("P1: %d P2: %d\n",WallCount[0],WallCount[1]);
}

int findLegalMoves(char State[17][17], char WallCount[2],bool isP2turn) {
	//보드의 상태를 받아, 현재 상태에서 규칙상 허용되는 모든 수를 array에 담고, 그 array의 포인터를 반환
	//이것은 후에 트리 탐색을 구현할때 사용되어야 하기에, 최적화가 잘 되어있어야 함
	//그리고 그 이유로 이 함수는 *매우* 중요함
	//bool isP2turn은 False이면, P1의 차례, True이면 P2의 차례이다.
	//이 함수는 [][5] 크기의 배열 포인터를 반환한다.
	
	int BufferCapacity = 1024 * 5;//버퍼의 크기 정의( 이 이상이면 재할당시킴)
	int ByteCount = 0;//지금까지 사용된 버퍼 바이트 저장
	char* LegalMoves;
	LegalMoves = (char*)calloc(BufferCapacity, sizeof(char));

	if (LegalMoves == NULL) {
		//메모리 할당 실패
		printf("FindLegalMoves(): Memory Allocation Failure\n");
		free(LegalMoves);
		return 1;
	}

	//메모리 다 사용시, 메모리를 재할당하는 코드 
	if (ByteCount == BufferCapacity) {
		//LegalMoves의 공간을 다 사용했을때, 메모리 재할당
		printf("Buffer full, Reallocating...\n");
		BufferCapacity *= 2;
		char* BufferRealloc = (char*)realloc(LegalMoves, sizeof(char));
		if (BufferRealloc == NULL) {
			//메모리 할당 실패
			printf("FindLegalMoves(): Memory Reallocation Failure\n");
			free(LegalMoves);
			return 1;
		}
		else { LegalMoves = BufferRealloc; }
	}
	/////////////////////////////////////////
	//아래서부터 가능한 모든 수 계산하는 코드
	////////////////////////////////////////
	char WallsLeft = WallCount[(int)isP2turn];
	//이번 차례에 해당하는 Player의 좌표 찾기
	//Player의 위치를 저장하는 배열(이 x,y 좌표는 행렬의 행/열을 저장한 것이기에 x,y로 변환시켜야 한다{가독성은 떨어지지만, 연산할때는 이 방식이 더 효율적)
	char PlayerPos[2] = { 0,0 };
	for (int i = 0; i < 17; i++) {
		for (int j = 0; j < 17; j++) {
			if (State[i][j] == (int)isP2turn + 1) {
				//지금 차례에 해당하는 플레이어의 좌표를 State에서 추출
				PlayerPos[0] = i;
				PlayerPos[1] = j;
			}
		}
	}
	//현재 플레이어가 갈 수 있는 방향 계산
	
	printf("Player's Current Position(in Array): %d , %d \n", PlayerPos[0], PlayerPos[1]);

	return 0;

	
}

bool isLegalMove(char State[17][17], char WallCount[2], bool isP2turn, char PlayerPos[2], char Move[5]) {
	//이 함수는 쿼리도 보드의 상태와 하려는 동작을 받아서, 이 수가 가능하다면 true, 불가능하다면 false를 반환한다.
	switch (Move[0]) {
	case 'M':
		//플레이어가 이동
		char PlayerNewPos[2] = { PlayerPos[0], PlayerPos[1] };
		switch (Move[1]) {
		case 'U':
			//플레이어가 위로 이동
			PlayerNewPos[1] -= 2;
			if (PlayerNewPos[1] < 0) {
				return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
			}
			else if (State[PlayerNewPos[0]][PlayerNewPos[1]] == 'W') {
				return false;
			}
			else if ((State[PlayerNewPos[0]][PlayerNewPos[1]] == '1') || (State[PlayerNewPos[0]][PlayerNewPos[1]] == '2')) {
				PlayerNewPos[1] -= 2;
				if (PlayerNewPos[1] < 0) {
					return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
				}return true;
			}
		case 'D':
			//아래로 이동
			PlayerNewPos[1] += 2;
			if (PlayerNewPos[1] >= 17) {
				return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
			}
			else if (State[PlayerNewPos[0]][PlayerNewPos[1]] == 'W') {
				return false;
			}
			else if ((State[PlayerNewPos[0]][PlayerNewPos[1]] == '1') || (State[PlayerNewPos[0]][PlayerNewPos[1]] == '2')) {
				PlayerNewPos[1] += 2;
				if (PlayerNewPos[1] >= 17) {
					return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
				}return true;
		case 'L':
			//왼쪽으로 이동
			PlayerNewPos[0] -= 2;
			if (PlayerNewPos[0] < 0) {
				return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
			}
			else if (State[PlayerNewPos[0]][PlayerNewPos[1]] == 'W') {
				return false;
			}
			else if ((State[PlayerNewPos[0]][PlayerNewPos[1]] == '1') || (State[PlayerNewPos[0]][PlayerNewPos[1]] == '2')) {
				PlayerNewPos[0] -= 2;
				if (PlayerNewPos[0] < 0) {
					return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
				}return true;
		case 'R':
			//오른쪽으로 이동
			PlayerNewPos[0] += 2;
			if (PlayerNewPos[0] >= 17) {
				return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
			}
			else if (State[PlayerNewPos[0]][PlayerNewPos[1]] == 'W') {
				return false;
			}
			else if ((State[PlayerNewPos[0]][PlayerNewPos[1]] == '1') || (State[PlayerNewPos[0]][PlayerNewPos[1]] == '2')) {
				PlayerNewPos[0] += 2;
				if (PlayerNewPos[0] >= 17) {
					return false;//플레이어가 가려는 방향이 보드 밖인지 확인(안하면 배열 밖의 원소 접근으로 undefined 활동)
				}return true;

	case 'W':
		//벽을 설치하는 명령
		for (int i = 0; i < 2; i++) {
			if ((0 <= Move[i]) && (Move[i] <= 17) && (0 <= Move[i+1]) && (Move[i+1] <= 17)) {
				//바운더리 체크
				if (State[Move[i]][Move[i + 1]] == 'W') {

				}


			}
		}
	}

}







int main() {
	char StartState[17][17];
	for (int i = 0; i < 17; i++) {
		for (int j = 0; j < 17; j++) {
			StartState[i][j] = 0;
		}
	}
	StartState[0][8] = 2;
	StartState[16][8] = 1;
	char StartWallCount[2] = {10,10};
	//이게 쿼리도 기본 시작 상태
	//StartState[16][15] = 'W';
	printf("Player1's turn\n");
	printBoard(StartState,StartWallCount);//쿼리도 보드 출력
	findLegalMoves(StartState, StartWallCount, false);//쿼리도 보드에서 합법척인 수 찾기
	return 0;
}