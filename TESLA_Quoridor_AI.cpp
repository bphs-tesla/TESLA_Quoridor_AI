#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <queue>
#include <array>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


struct Quoridor
{
	//쿼리도 기본 시작 상태로 initialize됨
	//쿼리도 초기 게임의 그래프를 선언 (81개의 상태 노드를 가진 그래프라고 해석시, 81*81차원의 행렬과 동치로 취급 가능)
	std::array<std::array<std::array<bool, 4>, 9>, 9> Path; //Path[행][열][상/하/좌/우] 형식으로 배열 저장
	std::array<uint8_t, 8> WallH = { NULL, }; //이거는 가로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
	std::array<uint8_t, 8> WallVt = { NULL, }; //이거는 세로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
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
			Path.at(0).at(i).at(0) = false; //맨 위쪽 줄에서 위쪽으로 갈수 없게
			Path.at(8).at(i).at(1) = false; //맨 아래쪽 줄에서 아래쪽으로 갈수 없게
			Path.at(i).at(0).at(2) = false; //맨 왼쪽 줄에서 왼쪽으로 갈수 없게
			Path.at(i).at(8).at(3) = false; //맨 오른쪽 줄에서 오른쪽으로 갈수 없게
		}
	}
	std::array<uint8_t, 2> P1 = { 8,4 }; //P1의 보드 위의 행/열
	std::array<uint8_t, 2> P2 = { 0,4 }; //P2의 보드 위의 행/열
	std::array<uint8_t, 2> WallCount = { 10,10 }; //P1,P2의 남은 벽 개수(10이 기본이고, 벽을 쓸수록 감소함)
	bool isP1Turn = true; // P1의 차례이면, true, P2의 차례이면 false (P1이 선공이라서 true가 기본)

	void PrintBoard() {
		//이거는 UI관련이라 대놓고 개같이 짤거임 ㅅㄱ
		//Quoridor를 받아서, 그에 대응되는 쿼리도 보드를 출력함
		putchar('+'); for (int _ = 0; _ < 7 * 9 + 8; _++) { putchar('-'); }printf("+\n");
		for (int i = 0; i < 9 + 8; i++) {
			if (i % 2 == 0) {
				//짝수(세로벽 설치)
				for (int _i = 0; _i < 3; _i++) {
					printf("|");
					for (int j = 0; j < 9 + 8; j++) {
						if (j % 2 == 0) {
							//칸
							if (_i == 1) {
								if ((this->P1[0] == i / 2) && (this->P1[1] == j / 2)) {
									printf("   1   ");
								}
								else if ((this->P2[0] == i / 2) && (this->P2[1] == j / 2)) {
									printf("   2   ");
								}
								else {
									printf("       ");
								}
							}
							else {
								printf("       ");
							}
						}
						else {
							//세로벽
							if ((this->Path[i / 2][j / 2][3] == 0) && (this->Path[i / 2][j / 2 + 1][2] == 0)) {
								//벽 존재
								printf("|");
								continue;
							}
							printf(" ");
						}
					}
					printf("|\n");
				}
			}
			else {
				//홀수(가로벽 설치)
				printf("|");
				for (int j = 0; j < 9 + 8; j++) {
					if (j % 2 == 0) {
						//가로벽
						if ((this->Path[i / 2][j / 2][1] == 0) && (this->Path[i / 2 + 1][(j / 2)][0] == 0)) {
							//벽 존재
							printf("-------");
							continue;
						}
						printf("       ");

					}
					else {
						//격자점
						printf("+");

					}
				}
				printf("|\n");
			}


		}
		putchar('+'); for (int _ = 0; _ < 7 * 9 + 8; _++) { putchar('-'); }printf("+\n");
	}
};

bool isWall(Quoridor* Board, std::array<uint8_t, 2> GridPos) {
	if ((Board->WallVt)[GridPos.at(1)] & (1 << GridPos.at(0))) { return true; } //그 자리에 이미 벽이 존재
	if ((Board->WallH)[GridPos.at(0)] & (1 << GridPos.at(1))) { return true; } //그 자리에 이미 벽이 존재
	return false;
}

struct AStar_Node {
	int Cost; //총 비용(f)
	int HeuristicCost; //휴리스틱 비용(h)
	int AccumulatedCost; //누적 비용(g)

	std::array<uint8_t, 2> Position; //노드의 위치
	bool operator <(const AStar_Node& other) const {
		return this->Cost > other.Cost;
	}
};

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
			if (std::find(AStar_ClosedNodes->begin(), AStar_ClosedNodes->end(), ChildNode.Position) == AStar_ClosedNodes->end()) {
				//AStar_ClosedNodes에 존재하지 않음(새로운 노드 >> 자식 생성)
				ChildNode.HeuristicCost = AStar_HeuristicCost(&ChildNode, Destination); //휴리스틱 비용 계산
				ChildNode.Cost = ChildNode.AccumulatedCost + ChildNode.HeuristicCost;
				AStar_Nodes->emplace(ChildNode);
			}
		}
	}
	//printf("AStar_Nodes Size: %d\n", (int)AStar_Nodes->size());
	//printf("AStar_ClosedNodes Size: %d\n", (int)AStar_ClosedNodes->size());
	return 0;
}

int AStar(Quoridor* Board) {
	/*
	지금 차례의 플레이어가 목적지로 갈때까지의 거리 반환
	목적지로 도달하는 것이 불가능하다면, -1 반환
	*/

	std::priority_queue<AStar_Node> AStar_Nodes; //열린 리스트 선언
	std::vector<std::array<uint8_t, 2>> AStar_ClosedNodes; //닫힌 리스트 선언(닫힌 리스트는 비용 고려를 할 필요가 없으니, 벡터 행렬로 압축
	int Destination = Board->isP1Turn ? 0 : 8; // 목적지 설정
	struct AStar_Node RootNode = { 0, };
	RootNode.Position[0] = Board->isP1Turn ? Board->P1[0] : Board->P2[0]; // 루트 노드 위치 (플레이어의 위치)
	RootNode.Position[1] = Board->isP1Turn ? Board->P1[1] : Board->P2[1];

	RootNode.HeuristicCost = AStar_HeuristicCost(&RootNode, Destination); //루트 노드 휴리스틱
	RootNode.AccumulatedCost = 0; //루트 노드라서 누적 비용 = 0
	AStar_Nodes.emplace(RootNode);

	int Distance;
	while (!AStar_Nodes.empty()) {
		//AStar_Nodes에 원소 존재
		Distance = AStar_MakeChildNode(Board, &AStar_Nodes, &AStar_ClosedNodes, Destination);

		if (Distance) {
			// 목적지 도달하는 자식노드 발견
			return Distance;
		}
	}
	return -1;//목적지로 가는 경로 존재 X
}

bool isConnected(Quoridor Board) {
	//보드에서 두 플레이어 모두 목적지로 가는 것이 가능한지 체크
	if (AStar(&Board) != -1) {
		Board.isP1Turn = !Board.isP1Turn; //상대방도 연결되어있는지 체크
		if (AStar(&Board) != -1) {
			return true;
		}
	}
	return false;
}

//bool isLegalMove(Quoridor* Board, uint8_t Move) {
//	//0b[플레이어/벽설치][가로/세로][행(3비트)][열(3비트)] 가로: 0, 세로: 1
//	//0b0[점프 유무][명령1(3비트)][명령2(3비트)] 점프시 1, 아니면 0
//	//0b1[가로/세로][행(3비트)][열(3비트)]
//	std::array<uint8_t, 2> MoveVariable;
//	MoveVariable.at(0) = (Move & (0b111 << 3)) >> 3;
//	MoveVariable.at(1) = Move & 0b111;
//	//MoveVariable은 Move에서 2개의 3비트 인수를 2개의 uint8_t 배열로 바꿔 저장한것임(안그러면 가독성이 너무 구려서)
//	switch (Move & (0b11 << 6)) {
//	case (0b00 << 6):
//		//0b00[명령1(3비트)][명령2(3비트)] 점프 X
//		/*상/하/좌/우 인지에 따라, 명령 1이 0 ~ 3까지를 표시한다*/
//		if (MoveVariable[0] > 3) { return false; } //정의되지 않은 범위
//		if (Board->isP1Turn) {
//			//P1의 차례(P1이 이동할때)
//			if (Board->Path[Board->P1[0]][Board->P1[1]][MoveVariable.at(0)]) {
//				//이동 가능
//				return true;
//			}
//		}
//		else {
//			//P2의 차례
//			if (Board->Path[Board->P2[0]][Board->P2[1]][MoveVariable.at(0)]) {
//				//이동 가능
//				return true;
//			}
//			return false; //예외 처리(Path에 1이 존재하지 않음, 즉 벽에 가로막힘)
//	case (0b01 << 6):
//		//0b01[명령1(3비트)][명령2(3비트)] 점프 O
//		if (Board->Path[Board->P2[0]][Board->P2[1]][MoveVariable.at(0)]) {
//			//이동 가능
//
//			return true;
//		}
//
//
//
//		break;
//	case (0b10 << 6):
//		//0b10[행(3비트)][열(3비트)] 가로 벽 설치
//		if ((Board->WallH[MoveVariable.at(0)] | 1 << MoveVariable.at(1)) & (((Board->WallH[MoveVariable.at(0)] | 1 << MoveVariable.at(1)) >> 1) | ((Board->WallH[MoveVariable.at(0)] | 1 << MoveVariable.at(1)) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
//		return !isWall(Board, &MoveVariable); //가로 벽 설치 가능 여부 판단(bitshift때문에, 범위 판정은 안해도 됨)
//	case (0b11 << 6):
//		//0b11[행(3비트)][열(3비트)] 세로 벽 설치
//		if ((Board->WallVt[MoveVariable.at(1)] | 1 << MoveVariable.at(0)) & (((Board->WallVt[MoveVariable.at(1)] | 1 << MoveVariable.at(0)) >> 1) | ((Board->WallVt[MoveVariable.at(1)] | 1 << MoveVariable.at(0)) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
//		return !isWall(Board, &MoveVariable); //세로 벽 설치 가능 여부 판단(bitshift때문에, 범위 판정은 안해도 됨)
//		}
//	}
//	return false; //어떻게 여기까지 올 수 있는지는 모르겠지만????? 만일을 위해서 예외 처리
//}
//
//int HeuristicEstimation(Quoridor* Board) {
//	/*
//	if (Board->isP1Turn) {
//		//P1의 차례
//	}
//	*/
//	return 0; //이거 원래 HeuristicEstimation은 보드의 상태(의 주소를 간접참조해) 보드의 상태를 평가하는 함수
//	/*재민이 화이팅 ㅎㅎ*/
//	/*
//
//	*/
//}
//
//int MakeChildNode(Quoridor* Board, int SearchTreeDepth) {
//	if (SearchTreeDepth <= 0) {
//		//탐색 깊이 최대 도달
//		return HeuristicEstimation(Board);
//	}
//	/*
//
//	마법
//
//	*/
//	//std::sort()
//	/*
//	어떤
//	*/
//
//
//	return 0;
//}
//
//std::vector<uint8_t> MakeLegalMoves(Quoridor* Board) {
//	std::vector<uint8_t> LegalMoves; //동적으로 크기 설정 가능해서 vector<uint8_t> 이용했음
//	//0b[플레이어/벽설치][가로/세로][행(3비트)][열(3비트)]
//	//0b0[점프 유무][명령1(3비트)][명령2(3비트)] 점프시 1, 아니면 0
//	//0b1[가로/세로][행(3비트)][열(3비트)]
//
//	return LegalMoves;
//}

bool canPlaceWallH(Quoridor* Board, std::array<uint8_t, 2> GridPos) {

	if ((GridPos.at(0) >= 8) || (GridPos.at(1) >= 8)) { return false; } //유효하지 않은 좌표 예외 처리
	if (isWall(Board, GridPos)) { return false; }
	if ((Board->WallH[GridPos.at(0)] | 1 << GridPos.at(1)) & (((Board->WallH[GridPos.at(0)] | 1 << GridPos.at(1)) >> 1) | ((Board->WallH[GridPos.at(0)] | 1 << GridPos.at(1)) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
	//Pos[0], Pos[1] - 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
	//Pos[0], Pos[1] + 1 위치의 가로 벽이 있는지 체크(없으면 이쪽 실행)
	Quoridor tempBoard = *Board;
	//Path 업데이트
	try {
		(tempBoard.Path).at(GridPos.at(0) + 1).at(GridPos.at(1)).at(0) = false;
		(tempBoard.Path).at(GridPos.at(0) + 1).at(GridPos.at(1) + 1).at(0) = false;
		(tempBoard.Path).at(GridPos.at(0)).at(GridPos.at(1)).at(1) = false;
		(tempBoard.Path).at(GridPos.at(0)).at(GridPos.at(1) + 1).at(1) = false;
	}
	catch (std::out_of_range) {
		return false;
	}
	if (!isConnected(tempBoard)) {
		return false; //두 플레이어중 최소 한명이 목적지에 도달 불가능
	}
	return true; //가로 벽 설치 가능
}

bool canPlaceWallV(Quoridor* Board, std::array<uint8_t, 2> GridPos) {
	if ((GridPos.at(0) >= 8) || (GridPos.at(1) >= 8)) { return false; } //유효하지 않은 좌표 예외 처리
	if (isWall(Board, GridPos)) { return false; }
	if ((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) & (((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) >> 1) | ((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
	//Pos[0], Pos[1] - 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
	//Pos[0], Pos[1] + 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
	Quoridor tempBoard = *Board;
	//Path 업데이트
	try {
		(tempBoard.Path)[GridPos.at(0)][GridPos.at(1) + 1][2] = false;
		(tempBoard.Path)[GridPos.at(0) + 1][GridPos.at(1) + 1][2] = false;
		(tempBoard.Path)[GridPos.at(0)][GridPos.at(1)][3] = false;
		(tempBoard.Path)[GridPos.at(0) + 1][GridPos.at(1)][3] = false;
	}
	catch (std::out_of_range) {
		return false;
	}
	if (!isConnected(tempBoard)) {
		return false; //두 플레이어중 최소 한명이 목적지에 도달 불가능
	}
	return true; //세로 벽 설치 가능
}

bool canMovePlayer(Quoridor* Board, uint8_t Move) {
	std::array<uint8_t, 2>* Player = (Board->isP1Turn) ? &Board->P1 : &Board->P2;
	if (Board->Path.at(Player->at(0)).at(Player->at(1)).at(Move)) {
		return true;
	}
	return false;
}

int PlaceWallH(Quoridor* Board, std::array<uint8_t, 2> GridPos) {
	//가로로 벽을 배치하는 코드 ( k=0,1 )
	//불가능하면 1 반환, 가능하면 0 반환
	if (canPlaceWallH(Board, GridPos)) {
		//가로벽 설치 가능

		//벽 설치 코드
		(Board->WallH).at(GridPos.at(0)) = (Board->WallH).at(GridPos.at(0)) | (1 << (GridPos.at(1))); //WallH 업데이트
		//Path 업데이트
		(Board->Path).at(GridPos.at(0) + 1).at(GridPos.at(1)).at(0) = false;
		(Board->Path).at(GridPos.at(0) + 1).at(GridPos.at(1) + 1).at(0) = false;
		(Board->Path).at(GridPos.at(0)).at(GridPos.at(1)).at(1) = false;
		(Board->Path).at(GridPos.at(0)).at(GridPos.at(1) + 1).at(1) = false;
		return 0; //정상적으로 Path & WallH 업데이트
	}
	return 1; //예외 처리
}
//Quoridor 보드(의 메모리 주소)를 받아서 벽을 배치하는 함수
int PlaceWallV(Quoridor* Board, std::array<uint8_t, 2> GridPos) {
	//세로로 벽을 배치하는 코드 ( k=2,3 )
	//불가능하면 1 반환, 가능하면 0 반환
	if (canPlaceWallV(Board, GridPos)) {
		//세로벽 설치 가능

		//벽 설치 코드
		(Board->WallVt).at(GridPos.at(1)) = (Board->WallVt).at(GridPos.at(1)) | (1 << (GridPos.at(0))); //WallVt 업데이트
		//Path 업데이트
		(Board->Path)[GridPos.at(0)][GridPos.at(1) + 1][2] = false;
		(Board->Path)[GridPos.at(0) + 1][GridPos.at(1) + 1][2] = false;
		(Board->Path)[GridPos.at(0)][GridPos.at(1)][3] = false;
		(Board->Path)[GridPos.at(0) + 1][GridPos.at(1)][3] = false;
		return 0; //정상적으로 Path & WallV 업데이트
	}
	return 1; //예외 처리
}

int MovePlayer(Quoridor* Board, uint8_t Move) {
	//점프 명령이면, -1
	//정상적으로 이동 후, 턴을 넘기면, 0
	//예외면 1
	std::array<uint8_t, 2>* Player = (Board->isP1Turn) ? &Board->P1 : &Board->P2;
	std::array<uint8_t, 2> Opponent = (Board->isP1Turn) ? Board->P2 : Board->P1;
	if (canMovePlayer(Board, Move)) {
		//이동 가능(Path에서 1)
		switch (Move) {
		case 0:
			Player->at(0) -= 1;
			break;
		case 1:
			Player->at(0) += 1;
			break;
		case 2:
			Player->at(1) -= 1;
			break;
		case 3:
			Player->at(1) += 1;
			break;
		}
		if (*Player == Opponent) {
			//상대랑 위치 겹침(점프) => 턴 넘김 X
			return -1;
		}
		else {
			//성공적으로 플레이어 이동 => 턴 넘김
			Board->isP1Turn = !Board->isP1Turn;
			//printf("turn passed\n");
			return 0;
		}
	}
	return 1; //예외처리
}

int isGameOver(Quoridor* Board) {
	//게임 끝났는지 알려주는 함수(이거는 파이썬 가독성&속도 위한거라 별거 없음)
	if (Board->P1.at(0) == 0) {
		return 1;
	}
	else if (Board->P2.at(0) == 8) {
		return 2;
	}
	else {
		return 0;
	}
}

int main() {
	struct Quoridor Board;
	std::array<uint8_t, 2> H = { 2,7 };
	std::array<uint8_t, 2>  H2 = { 0,2 };
	std::array<uint8_t, 2>  V = { 1,7 };
	std::array<uint8_t, 2>  V2 = { 2,3 };
	//uint8_t V[2] = { 3,4 };
	//Board.PlaceWallH(&H);
	//PlaceWallV(&Board, &V);
	//PlaceWallH(&Board, &H2);
	//PlaceWallV(&Board, &V2);
	std::array<uint8_t, 2> GridWallPos = { 0,0 };
	for (int i = 0; i < 8; i++, GridWallPos.at(1)++) {
		PlaceWallH(&Board, GridWallPos);
	}
	//PlaceWallVt(&Board, V2);
	printf("Distance: %d\n\n", AStar(&Board));
	Board.PrintBoard();
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


//pybind11 모듈 바인딩
PYBIND11_MODULE(Quoridor, m) {
	m.doc() = "Quoridor Cpp library";
	pybind11::class_<Quoridor>(m, "Quoridor")
		.def(pybind11::init<>())// 생성자 바인딩 (필요 시)
		.def("PrintBoard", &Quoridor::PrintBoard)
		.def_readwrite("WallH", &Quoridor::WallH)
		.def_readwrite("WallVt", &Quoridor::WallVt)
		.def_readwrite("P1", &Quoridor::P1)
		.def_readwrite("P2", &Quoridor::P2)
		.def_readwrite("WallCount", &Quoridor::WallCount)
		.def_readwrite("isP1Turn", &Quoridor::isP1Turn);

	m.def("isWall", &isWall, "checks if the following GridPos has a wall");
	m.def("isConected", &isConnected, "checks if the players are possible to reach destination");
	m.def("canPlaceWallH", &canPlaceWallH, "checks if placing a horizontal wall is valid");
	m.def("canPlaceWallV", &canPlaceWallV, "checks if placing a vertical wall is valid");
	m.def("canMovePlayer", &canMovePlayer, "checks if moving a player");
	m.def("PlaceWallH", &PlaceWallH, "places horizontal wall");
	m.def("PlaceWallV", &PlaceWallV, "places vertical wall");
	m.def("PrintBoard", &Quoridor::PrintBoard, "prints current Board");
	m.def("MovePlayer", &MovePlayer, "moves player");
	m.def("AStar", &AStar, "finds the distance between Player and destination");
	m.def("isGameOver", &isGameOver, "checks if the game is over and returns the winner");
}