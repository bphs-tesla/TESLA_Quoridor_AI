#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <queue>
#include <array>

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

	int PlaceWallH(uint8_t GridPos[2]) {
		//가로로 벽을 배치하는 코드 ( k=0,1 )
		//불가능하면 1 반환, 가능하면 0 반환
		if ((GridPos[0] >= 8) || (GridPos[1] >= 8)) { return 1; } //유효하지 않은 좌표 예외 처리
		if (isWall(GridPos)) { return 1; }
		if ((this->WallH[GridPos[0]] | 1 << GridPos[1]) & (((this->WallH[GridPos[0]] | 1 << GridPos[1]) >> 1) | ((this->WallH[GridPos[0]] | 1 << GridPos[1]) << 1))) { return 1; }//인접한 곳에 벽이 있어 충돌판정
		//Pos[0], Pos[1] - 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
		//Pos[0], Pos[1] + 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
		//벽 설치 코드
		(this->WallH)[GridPos[0]] = (this->WallH)[GridPos[0]] | (1 << (GridPos[1])); //WallH 업데이트
		//Path 업데이트
		(this->Path)[GridPos[0] + 1][GridPos[1]][0] = false;
		(this->Path)[GridPos[0] + 1][GridPos[1] + 1][0] = false;
		(this->Path)[GridPos[0]][GridPos[1]][1] = false;
		(this->Path)[GridPos[0]][GridPos[1] + 1][1] = false;
		return 0; //정상적으로 Path & WallH 업데이트

	}
	//Quoridor 보드(의 메모리 주소)를 받아서 벽을 배치하는 함수
	//WallVt는 전치되어서 t가 붙음
	int PlaceWallVt(uint8_t GridPos[2]) {
		//세로로 벽을 배치하는 코드 ( k=2,3 )
		//불가능하면 1 반환, 가능하면 0 반환
		if ((GridPos[0] >= 8) || (GridPos[1] >= 8)) { return 1; } //유효하지 않은 좌표 예외 처리
		if (isWall(GridPos)) { return 1; }
		if ((this->WallVt[GridPos[1]] | 1 << GridPos[0]) & (((this->WallVt[GridPos[1]] | 1 << GridPos[0]) >> 1) | ((this->WallVt[GridPos[1]] | 1 << GridPos[0]) << 1))) { return 1; }//인접한 곳에 벽이 있어 충돌판정
		//Pos[0], Pos[1] +- 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
		//벽 설치 코드
		(this->WallVt)[GridPos[1]] = (this->WallVt)[GridPos[1]] | (1 << (GridPos[0])); //WallVt 업데이트
		//Path 업데이트
		(this->Path)[GridPos[1]][GridPos[0] + 1][2] = false;
		(this->Path)[GridPos[1] + 1][GridPos[0] + 1][2] = false;
		(this->Path)[GridPos[1]][GridPos[0]][3] = false;
		(this->Path)[GridPos[1] + 1][GridPos[0]][3] = false;
		return 0; //정상적으로 Path & WallV 업데이트
	}

	bool isWall(uint8_t GridPos[2]) {
		if ((this->WallVt)[GridPos[1]] & (1 << GridPos[0])) { return true; } //그 자리에 이미 벽이 존재
		if ((this->WallH)[GridPos[0]] & (1 << GridPos[1])) { return true; } //그 자리에 이미 벽이 존재
		return false;
	}

	bool isConnected() {
		//GridPos에 벽을 설치하는것이 가능하다는 전제로 이 함수가 실행됨(체크 별도로 해야함)
		return false;
	}

	//(지금은) 작동 안됨
	void PrintBoard(){
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
					else if (this->WallH[i / 6] & (1 << (j))) {
						printf("-");
					}
					else { printf(" "); }
				}
			}
			else {
				//세로벽 설치
				for (int j = 0; j < 53; j++) {
					if ((j % 6) == 5) {
						if (this->WallVt[i / 12] & (1 << (j / 12))) {
							printf("|");
						}
						else { printf(" "); }
					}
					else { printf(" "); }
				}
			}
			printf("|\n");
		}
		putchar('+'); for (int _ = 0; _ < 53; _++) { putchar('-'); }printf("+\n");

	}

	bool isLegalMove(uint8_t Move) {
		//0b[플레이어/벽설치][가로/세로][행(3비트)][열(3비트)] 가로: 0, 세로: 1
		//0b0[점프 유무][명령1(3비트)][명령2(3비트)] 점프시 1, 아니면 0
		//0b1[가로/세로][행(3비트)][열(3비트)]
		uint8_t MoveVariable[2];
		MoveVariable[0] = (Move & (0b111 << 3)) >> 3;
		MoveVariable[1] = Move & 0b111;
		//MoveVariable은 Move에서 2개의 3비트 인수를 2개의 uint8_t 배열로 바꿔 저장한것임(안그러면 가독성이 너무 구려서)
		switch (Move & (0b11 << 6)) {
		case (0b00 << 6):
			//0b00[명령1(3비트)][명령2(3비트)] 점프 X
			/*상/하/좌/우 인지에 따라, 명령 1이 0 ~ 3까지를 표시한다*/
			if (MoveVariable[0] > 3) { return false; } //정의되지 않은 범위
			if (this->isP1Turn) {
				//P1의 차례(P1이 이동할때)
				if (this->Path[this->P1[0]][this->P1[1]][MoveVariable[0]]) {
					//이동 가능
					return true;
				}
			}
			else {
				//P2의 차례
				if (this->Path[this->P2[0]][this->P2[1]][MoveVariable[0]]) {
					//이동 가능
					return true;
				}
				return false; //예외 처리(Path에 1이 존재하지 않음, 즉 벽에 가로막힘)
		case (0b01 << 6):
			//0b01[명령1(3비트)][명령2(3비트)] 점프 O
			if (this->Path[this->P2[0]][this->P2[1]][MoveVariable[0]]) {
				//이동 가능

				return true;
			}



			break;
		case (0b10 << 6):
			//0b10[행(3비트)][열(3비트)] 가로 벽 설치
			if ((this->WallH[MoveVariable[0]] | 1 << MoveVariable[1]) & (((this->WallH[MoveVariable[0]] | 1 << MoveVariable[1]) >> 1) | ((this->WallH[MoveVariable[0]] | 1 << MoveVariable[1]) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
			return !isWall(MoveVariable); //가로 벽 설치 가능 여부 판단(bitshift때문에, 범위 판정은 안해도 됨)
		case (0b11 << 6):
			//0b11[행(3비트)][열(3비트)] 세로 벽 설치
			if ((this->WallVt[MoveVariable[1]] | 1 << MoveVariable[0]) & (((this->WallVt[MoveVariable[1]] | 1 << MoveVariable[0]) >> 1) | ((this->WallVt[MoveVariable[1]] | 1 << MoveVariable[0]) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
			return !isWall(MoveVariable); //세로 벽 설치 가능 여부 판단(bitshift때문에, 범위 판정은 안해도 됨)
			}
		}
		return false; //어떻게 여기까지 올 수 있는지는 모르겠지만????? 만일을 위해서 예외 처리
	}

	int HeuristicEstimation(Quoridor* Board) {
		/*
		if (Board->isP1Turn) {
			//P1의 차례
		}
		*/
		return 0; //이거 원래 HeuristicEstimation은 보드의 상태(의 주소를 간접참조해) 보드의 상태를 평가하는 함수
		/*재민이 화이팅 ㅎㅎ*/
		/*

		*/
	}

	int MakeChildNode(Quoridor* Board, int SearchTreeDepth) {
		if (SearchTreeDepth <= 0) {
			//탐색 깊이 최대 도달
			return HeuristicEstimation(Board);
		}
		/*

		마법

		*/
		//std::sort()
		/*
		어떤
		*/


		return 0;
	}

	std::vector<uint8_t> MakeLegalMoves(Quoridor Board) {
		std::vector<uint8_t> LegalMoves; //동적으로 크기 설정 가능해서 vector<uint8_t> 이용했음
		//0b[플레이어/벽설치][가로/세로][행(3비트)][열(3비트)]
		//0b0[점프 유무][명령1(3비트)][명령2(3비트)] 점프시 1, 아니면 0
		//0b1[가로/세로][행(3비트)][열(3비트)]

		return LegalMoves;
	}
};

struct AStar_Node {
	int Cost; //총 비용(f)
	int HeuristicCost; //휴리스틱 비용(h)
	int AccumulatedCost; //누적 비용(g)

	std::array<uint8_t, 2> Position; //노드의 위치
	bool operator <(const AStar_Node & other) const {
		return this->Cost > other.Cost;
	}
};

////////////////////////////////////////////////////////////////////

inline int AStar_HeuristicCost(AStar_Node* Node, int Destination) {
	//휴리스틱비용
	//return (Board->isP1Turn ? 0 : 8) > PlayerPos[0] ? (Board->isP1Turn ? 0 : 8) - PlayerPos[0] : PlayerPos[0] - (Board->isP1Turn ? 0 : 8);
	return abs(Destination - Node->Position[0]);
}

int AStar_MakeChildNode(Quoridor* Board, std::priority_queue<AStar_Node>* AStar_Nodes, std::vector<std::array<uint8_t, 2>>* AStar_ClosedNodes, int Destination) {
	/*
	자식 노드를 비용이 최소인 부모 노드를 기반으로 만들고, 부모 노드를 닫힌 리스트로 이동시키는 함수
	자식 생성후, 그 노드가 목표가 아니면 0, 목표라면 출발-도착지까지의 거리 반환
	*/
	//Parent의 노드를 받고, 그 상태에서 이동 가능한 경로 탐색 후 자식 노드 생성

	//f = 비용g(n) + 휴리스틱h(n)
	struct AStar_Node ParentNode = AStar_Nodes->top(); //부모 노드 선택
	AStar_ClosedNodes->emplace_back(ParentNode.Position);
	AStar_Nodes->pop();

	struct AStar_Node ChildNode;
	ChildNode.AccumulatedCost = ParentNode.AccumulatedCost + 1;//누적 비용은 부모 함수 + 1(1번 더 이동해야 자식이므로)

	//자식 노드 생성 & queue 에 push()
	for (int i = 0; i < 4; i++) {
		if (Board->Path[ParentNode.Position[0]][ParentNode.Position[1]][i]) {
			//이동 가능한 수인지 확인
			ChildNode.Position = ParentNode.Position; //초기에 이동 전에는 부모 노드와 같음(즉, 초기 선언)
			ChildNode.Position[i / 2] += ((i % 2) == 0) ? -1 : 1; //상하좌우에 따라 ChildNode의 위치를 +-1(이동)
			if (ChildNode.Position[0] == Destination) { return ChildNode.AccumulatedCost; } //자식 노드가 목표면 누적비용(거리) 반환
			if (std::find(AStar_ClosedNodes->begin(), AStar_ClosedNodes->end(), ChildNode.Position) != AStar_ClosedNodes->end()) {
				//AStar_ClosedNodes에 존재하지 않음(새로운 노드 >> 자식 생성)
				ChildNode.HeuristicCost = AStar_HeuristicCost(&ChildNode, Destination); //휴리스틱 비용 계산
				ChildNode.Cost = ChildNode.AccumulatedCost + ChildNode.HeuristicCost;
				AStar_Nodes->emplace(ChildNode);
			}
		}
	}
	return 0;
}

int AStar(Quoridor* Board) {
	/*
	지금 차례의 플레이어가 목적지로 갈때까지의 거리 반환
	목적지로 도달하는 것이 불가능하다면, 0 반환
	*/
	
	std::priority_queue<AStar_Node> AStar_Nodes; //열린 리스트 선언
	std::vector<std::array<uint8_t, 2>> AStar_ClosedNodes; //닫힌 리스트 선언(닫힌 리스트는 비용 고려를 할 필요가 없으니, 벡터 행렬로 압축
	int Destination = Board->isP1Turn ? 0 : 8; // 목적지 설정
	struct AStar_Node RootNode = { 0, };
	RootNode.Position[0] = Board->isP1Turn ? Board->P1[0] : Board->P2[0]; // 루트 노드 위치 (플레이어의 위치)
	RootNode.Position[1] = Board->isP1Turn ? Board->P1[1] : Board->P2[1];

	RootNode.HeuristicCost = AStar_HeuristicCost(&RootNode, Destination); //루트 노드 휴리스틱
	RootNode.AccumulatedCost = 0; //루트 노드라서 누적 비용 = 0

	int Distance;
	while (!AStar_Nodes.empty()) {
		//AStar_Nodes에 원소 존재
		Distance = AStar_MakeChildNode(Board, &AStar_Nodes, &AStar_ClosedNodes, Destination);
		if (Distance) {
			// 목적지 도달하는 자식노드 발견
			return Distance;
		}
	}
	return 0;//목적지로 가는 경로 존재 X
}



int main() {
	struct Quoridor Board;
	uint8_t H[2] = { 0,0 };
	uint8_t H2[2] = { 1,2 };
	uint8_t V[2] = { 1,3 };
	uint8_t V2[2] = { 2,3 };
	//uint8_t V[2] = { 3,4 };
	Board.PlaceWallH(H);
	Board.PlaceWallVt(V);
	//PlaceWallVt(&Board, V2);
	
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

	printf("\nWallV\n");
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

	return 0;
}