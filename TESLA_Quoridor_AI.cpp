#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <array>
#include <tuple>

#include <stdexcept> // std::out_of_range를 사용하기 위해 필수
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <omp.h>

struct Quoridor	
{
	//쿼리도 기본 시작 상태로 initialize됨
	//쿼리도 초기 게임의 그래프를 선언 (81개의 상태 노드를 가진 그래프라고 해석시, 81*81차원의 행렬과 동치로 취급 가능)
	std::array<std::array<std::array<bool, 4>, 9>, 9> Path; //Path[행][열][상/하/좌/우] 형식으로 배열 저장
	std::array<uint8_t, 8> WallH = { 0 }; //이거는 가로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
	std::array<uint8_t, 8> WallVt = { 0 }; //이거는 세로로 배치되는 벽의 격자점 위의 존재 여부를 저장한다.
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
							if ((this->Path.at(i / 2).at(j / 2).at(3) == 0) && (this->Path.at(i / 2).at(j / 2 + 1).at(2) == 0)) {
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


// Move 방향
// 0 = 위
// 1 = 아래
// 2 = 왼쪽
// 3 = 오른쪽
bool GetMoveDestination(
	Quoridor* Board,
	std::array<uint8_t, 2> Start,
	uint8_t Move,
	std::array<uint8_t, 2>& Destination,
	bool movingP1
)
{
	if (Move >= 4)
		return false;

	const std::array<uint8_t, 2>& Opponent =
		movingP1 ? Board->P2 : Board->P1;

	// 0 = 위
	// 1 = 아래
	// 2 = 왼쪽
	// 3 = 오른쪽

	static constexpr int dr[4] = {
		-1, 1, 0, 0
	};

	static constexpr int dc[4] = {
		0, 0, -1, 1
	};

	const int r = static_cast<int>(Start[0]);
	const int c = static_cast<int>(Start[1]);

	// 현재 위치에서 해당 방향으로 이동 가능한지
	if (!Board->Path[r][c][Move])
		return false;

	const int nr = r + dr[Move];
	const int nc = c + dc[Move];

	// 보드 밖
	if (nr < 0 || nr >= 9 ||
		nc < 0 || nc >= 9)
	{
		return false;
	}

	std::array<uint8_t, 2> Next = {
		static_cast<uint8_t>(nr),
		static_cast<uint8_t>(nc)
	};

	// =========================================================
	// 1. 바로 앞에 상대방이 없는 경우
	// =========================================================

	if (Next != Opponent)
	{
		Destination = Next;
		return true;
	}

	// =========================================================
	// 2. 상대방이 바로 앞에 있음
	//    → 뒤로 점프 가능한지 확인
	// =========================================================

	const int jr = nr + dr[Move];
	const int jc = nc + dc[Move];

	if (jr >= 0 && jr < 9 &&
		jc >= 0 && jc < 9 &&
		Board->Path[Opponent[0]][Opponent[1]][Move])
	{
		Destination = {
			static_cast<uint8_t>(jr),
			static_cast<uint8_t>(jc)
		};

		return true;
	}

	// =========================================================
	// 3. 뒤로 점프할 수 없음
	//    → 상대방 옆의 대각선 칸으로 이동
	//
	//    위/아래 이동 중이었다면 → 왼쪽/오른쪽
	//    왼쪽/오른쪽 이동 중이었다면 → 위/아래
	// =========================================================

	std::array<uint8_t, 2> SideMoves;

	if (Move == 0 || Move == 1)
	{
		// 위/아래 → 왼쪽/오른쪽
		SideMoves = { 2, 3 };
	}
	else
	{
		// 왼쪽/오른쪽 → 위/아래
		SideMoves = { 0, 1 };
	}

	for (uint8_t SideMove : SideMoves)
	{
		if (!Board->Path[Opponent[0]][Opponent[1]][SideMove])
			continue;

		const int sr =
			static_cast<int>(Opponent[0]) +
			dr[SideMove];

		const int sc =
			static_cast<int>(Opponent[1]) +
			dc[SideMove];

		if (sr < 0 || sr >= 9 ||
			sc < 0 || sc >= 9)
		{
			continue;
		}

		Destination = {
			static_cast<uint8_t>(sr),
			static_cast<uint8_t>(sc)
		};

		return true;
	}

	return false;
}

struct AStar_Node
{
	int Cost;
	int HeuristicCost;
	int AccumulatedCost;

	std::array<uint8_t, 2> Position;

	bool operator<(const AStar_Node& other) const
	{
		return Cost > other.Cost;
	}
	std::vector<std::array<uint8_t, 2>> TakePaths;
};

inline int AStar_HeuristicCost(AStar_Node* Node, int Destination) {
	//휴리스틱비용
	//return (Board->isP1Turn ? 0 : 8) > PlayerPos[0] ? (Board->isP1Turn ? 0 : 8) - PlayerPos[0] : PlayerPos[0] - (Board->isP1Turn ? 0 : 8);
	return abs(Destination - Node->Position[0]);
}
int AStar_MakeChildNode(
	Quoridor* Board,
	std::priority_queue<AStar_Node>& AStar_Nodes,
	bool Closed[9][9],
	int Destination,
	bool movingP1
)
{
	if (AStar_Nodes.empty())
		return -1;

	AStar_Node ParentNode = AStar_Nodes.top();
	AStar_Nodes.pop();

	const int pr = ParentNode.Position[0];
	const int pc = ParentNode.Position[1];

	// 이미 처리한 노드라면 무시
	if (Closed[pr][pc])
		return 0;

	Closed[pr][pc] = true;

	for (uint8_t Move = 0; Move < 4; Move++)
	{
		std::array<uint8_t, 2> NextPosition;

		if (!GetMoveDestination(
			Board,
			ParentNode.Position,
			Move,
			NextPosition,
			movingP1))
		{
			continue;
		}

		const int nr = NextPosition[0];
		const int nc = NextPosition[1];

		// 이미 Closed
		if (Closed[nr][nc])
			continue;

		const int NewG =
			ParentNode.AccumulatedCost + 1;

		// 목적지 도착
		if (nr == Destination)
		{
			return NewG;
		}

		AStar_Node ChildNode{};

		ChildNode.Position = NextPosition;
		ChildNode.AccumulatedCost = NewG;

		ChildNode.HeuristicCost =
			abs(Destination - nr);

		ChildNode.Cost =
			ChildNode.AccumulatedCost +
			ChildNode.HeuristicCost;
		ChildNode.TakePaths.push_back(ParentNode.Position);
		AStar_Nodes.emplace(ChildNode);
	}

	return 0;
}

int AStar(Quoridor* Board, bool isP1Turn)
{
	const int Destination =
		isP1Turn ? 0 : 8;

	const std::array<uint8_t, 2>& Start =
		isP1Turn ? Board->P1 : Board->P2;

	// 이미 목표 행
	if (Start[0] == Destination)
		return 0;

	std::priority_queue<AStar_Node> AStar_Nodes;

	bool Closed[9][9] = {};

	AStar_Node RootNode{};

	RootNode.Position = Start;

	RootNode.AccumulatedCost = 0;

	RootNode.HeuristicCost =
		abs(Destination - static_cast<int>(Start[0]));

	RootNode.Cost =
		RootNode.AccumulatedCost +
		RootNode.HeuristicCost;

	AStar_Nodes.emplace(RootNode);

	while (!AStar_Nodes.empty())
	{
		int Distance =
			AStar_MakeChildNode(
				Board,
				AStar_Nodes,
				Closed,
				Destination,
				isP1Turn
			);

		if (Distance > 0)
			return Distance;
	}

	return -1;
}

std::vector<std::array<uint8_t, 2>> AStar_MakeChildNode_withPath(
	Quoridor* Board,
	std::priority_queue<AStar_Node>& AStar_Nodes,
	bool Closed[9][9],
	int Destination,
	bool movingP1
)
{
	// 경로가 없다는 의미
	if (AStar_Nodes.empty())
		return {};

	AStar_Node ParentNode = AStar_Nodes.top();
	AStar_Nodes.pop();

	const int pr = ParentNode.Position[0];
	const int pc = ParentNode.Position[1];

	// 이미 처리한 노드라면 무시
	if (Closed[pr][pc])
		return {};

	Closed[pr][pc] = true;

	for (uint8_t Move = 0; Move < 4; Move++)
	{
		std::array<uint8_t, 2> NextPosition;

		if (!GetMoveDestination(
			Board,
			ParentNode.Position,
			Move,
			NextPosition,
			movingP1))
		{
			continue;
		}

		const int nr = NextPosition[0];
		const int nc = NextPosition[1];

		// 이미 Closed
		if (Closed[nr][nc])
			continue;

		const int NewG =
			ParentNode.AccumulatedCost + 1;

		AStar_Node ChildNode{};

		ChildNode.Position = NextPosition;
		ChildNode.AccumulatedCost = NewG;

		ChildNode.HeuristicCost =
			abs(Destination - nr);

		ChildNode.Cost =
			ChildNode.AccumulatedCost +
			ChildNode.HeuristicCost;

		// 부모가 가지고 있던 경로를 복사
		ChildNode.TakePaths = ParentNode.TakePaths;

		// 현재 이동한 위치 추가
		ChildNode.TakePaths.push_back(NextPosition);

		// 목적지 도착
		if (nr == Destination)
		{
			return ChildNode.TakePaths;
		}

		AStar_Nodes.emplace(std::move(ChildNode));
	}

	return {};
}

std::vector<std::array<uint8_t, 2>> AStar_withPath(
	Quoridor* Board,
	bool isP1Turn
)
{
	const int Destination =
		isP1Turn ? 0 : 8;

	const std::array<uint8_t, 2>& Start =
		isP1Turn ? Board->P1 : Board->P2;

	// 이미 목적지에 있다면 현재 위치만 반환
	if (Start[0] == Destination)
	{
		return { Start };
	}

	std::priority_queue<AStar_Node> AStar_Nodes;

	bool Closed[9][9] = {};

	AStar_Node RootNode{};

	RootNode.Position = Start;

	RootNode.AccumulatedCost = 0;

	RootNode.HeuristicCost =
		abs(Destination - static_cast<int>(Start[0]));

	RootNode.Cost =
		RootNode.AccumulatedCost +
		RootNode.HeuristicCost;

	// 시작점을 경로에 넣음
	RootNode.TakePaths.push_back(Start);

	AStar_Nodes.emplace(std::move(RootNode));

	while (!AStar_Nodes.empty())
	{
		std::vector<std::array<uint8_t, 2>> Path =
			AStar_MakeChildNode_withPath(
				Board,
				AStar_Nodes,
				Closed,
				Destination,
				isP1Turn
			);

		// 빈 벡터가 아니면 목적지까지의 경로를 찾은 것
		if (!Path.empty())
		{
			return Path;
		}
	}

	// 경로 없음
	return {};
}

int AStarNextMove(
	Quoridor* Board,
	bool movingP1
)
{
	std::vector<std::array<uint8_t, 2>> Path =
		AStar_withPath(Board, movingP1);

	if (Path.size() < 2)
		return -1;

	const auto& Current = Path[0];
	const auto& Next = Path[1];

	int dr =
		static_cast<int>(Next[0]) -
		static_cast<int>(Current[0]);

	int dc =
		static_cast<int>(Next[1]) -
		static_cast<int>(Current[1]);

	if (dr == -1) return 0;
	if (dr == 1) return 1;
	if (dc == -1) return 2;
	if (dc == 1) return 3;

	return -1;
}
bool isConnected(Quoridor& Board)
{
	return
		AStar(&Board, true) != -1 &&
		AStar(&Board, false) != -1;
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
	if ((Board->isP1Turn ? Board->WallCount.at(0) : Board->WallCount.at(1)) <= 0) { return false; } //벽 개수 없을때
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
	if ((Board->isP1Turn ? Board->WallCount.at(0) : Board->WallCount.at(1)) <= 0) { return false; } //벽 개수 없을때
	if ((GridPos.at(0) >= 8) || (GridPos.at(1) >= 8)) { return false; } //유효하지 않은 좌표 예외 처리
	if (isWall(Board, GridPos)) { return false; }
	if ((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) & (((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) >> 1) | ((Board->WallVt[GridPos.at(1)] | 1 << GridPos.at(0)) << 1))) { return false; }//인접한 곳에 벽이 있어 충돌판정
	//Pos[0], Pos[1] - 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
	//Pos[0], Pos[1] + 1 위치의 세로 벽이 있는지 체크(없으면 이쪽 실행)
	Quoridor tempBoard = *Board;
	//Path 업데이트
	try {
		tempBoard.Path.at(GridPos.at(0)).at(GridPos.at(1) + 1).at(2) = false;
		tempBoard.Path.at(GridPos.at(0) + 1).at(GridPos.at(1) + 1).at(2) = false;
		tempBoard.Path.at(GridPos.at(0)).at(GridPos.at(1)).at(3) = false;
		tempBoard.Path.at(GridPos.at(0) + 1).at(GridPos.at(1)).at(3) = false;
	}
	catch (std::out_of_range) {
		return false;
	}
	if (!isConnected(tempBoard)) {
		return false; //두 플레이어중 최소 한명이 목적지에 도달 불가능
	}
	return true; //세로 벽 설치 가능
}

bool canMovePlayer(Quoridor* Board, uint8_t Move)
{
	std::array<uint8_t, 2> Player =
		Board->isP1Turn ? Board->P1 : Board->P2;

	std::array<uint8_t, 2> Destination;

	return GetMoveDestination(
		Board,
		Player,
		Move,
		Destination,
		Board->isP1Turn
	);
}

inline void PassTurn(Quoridor* Board) {
	Board->isP1Turn = !Board->isP1Turn; //isP1Turn을 반대로 저장
	return;
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
		uint8_t Player = Board->isP1Turn ? 0 : 1; 
		Board->WallCount[Player]--;
		PassTurn(Board);
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
		(Board->Path).at(GridPos.at(0)).at(GridPos.at(1) + 1).at(2) = false;
		(Board->Path).at(GridPos.at(0) + 1).at(GridPos.at(1) + 1).at(2) = false;
		(Board->Path).at(GridPos.at(0)).at(GridPos.at(1)).at(3) = false;
		(Board->Path).at(GridPos.at(0) + 1).at(GridPos.at(1)).at(3) = false;
		uint8_t Player = Board->isP1Turn ? 0 : 1; 
		Board->WallCount[Player]--;
		PassTurn(Board);
		return 0; //정상적으로 Path & WallV 업데이트
	}
	return 1; //예외 처리
}

int MovePlayer(Quoridor* Board, uint8_t Move)
{
	std::array<uint8_t, 2>* Player =
		Board->isP1Turn ? &Board->P1 : &Board->P2;

	std::array<uint8_t, 2> Destination;

	if (!GetMoveDestination(
		Board,
		*Player,
		Move,
		Destination,
		Board->isP1Turn))
	{
		return 1;
	}

	*Player = Destination;

	PassTurn(Board);

	return 0;
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
std::tuple<
	std::vector<float>,
	std::vector<int>,
	std::vector<int>,
	std::vector<int>
>
StepBatchImpl(
	pybind11::list boards,
	const std::vector<int>& actions,
	const std::vector<int>& actingPlayers,
	const std::vector<int>& prevP1Distances,
	const std::vector<int>& prevP2Distances
)
{
	const size_t N = boards.size();

	if (actions.size() != N ||
		actingPlayers.size() != N ||
		prevP1Distances.size() != N ||
		prevP2Distances.size() != N)
	{
		throw std::runtime_error(
			"StepBatch: input sizes do not match"
		);
	}

	// ============================================================
	// Python Quoridor 객체의 실제 주소
	// ============================================================

	std::vector<Quoridor*> boardPtrs(N);

	for (size_t i = 0; i < N; ++i)
	{
		boardPtrs[i] =
			&pybind11::cast<Quoridor&>(boards[i]);
	}

	// ============================================================
	// 결과 배열
	// ============================================================

	std::vector<float> rewards(N, 0.0f);
	std::vector<int> dones(N, 0);

	std::vector<int> p1Distances(N, -1);
	std::vector<int> p2Distances(N, -1);

	// ============================================================
	// GIL release
	// ============================================================

	{
		pybind11::gil_scoped_release release;

#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(N); ++i)
		{
			Quoridor* board = boardPtrs[i];

			const int action = actions[i];
			const int actingPlayer = actingPlayers[i];

			float reward = 0.0f;

			// ====================================================
			// 1. 잘못된 actingPlayer 방어
			// ====================================================

			if (actingPlayer != 1 && actingPlayer != 2)
			{
				rewards[i] = -2.0f;

				p1Distances[i] = AStar(board, true);
				p2Distances[i] = AStar(board, false);

				dones[i] = 0;

				continue;
			}

			// ====================================================
			// 2. 행동 실행
			// ====================================================

			int result = 1;

			if (action < 0 || action >= 132)
			{
				// 존재하지 않는 행동
				reward = -2.0f;

				p1Distances[i] = AStar(board, true);
				p2Distances[i] = AStar(board, false);

				rewards[i] = reward;
				dones[i] = 0;

				continue;
			}

			// ----------------------------------------------------
			// 플레이어 이동
			// ----------------------------------------------------

			if (action < 4)
			{
				result = MovePlayer(
					board,
					static_cast<uint8_t>(action)
				);
			}

			// ----------------------------------------------------
			// 가로 벽
			// ----------------------------------------------------

			else if (action < 68)
			{
				const int wallIndex = action - 4;

				std::array<uint8_t, 2> gridPos =
				{
					static_cast<uint8_t>(wallIndex / 8),
					static_cast<uint8_t>(wallIndex % 8)
				};

				result = PlaceWallH(
					board,
					gridPos
				);
			}

			// ----------------------------------------------------
			// 세로 벽
			// ----------------------------------------------------

			else
			{
				const int wallIndex = action - 68;

				std::array<uint8_t, 2> gridPos =
				{
					static_cast<uint8_t>(wallIndex / 8),
					static_cast<uint8_t>(wallIndex % 8)
				};

				result = PlaceWallV(
					board,
					gridPos
				);
			}

			// ====================================================
			// 3. 불법 행동
			// ====================================================

			if (result != 0)
			{
				reward = -2.0f;

				p1Distances[i] = AStar(
					board,
					true
				);

				p2Distances[i] = AStar(
					board,
					false
				);

				rewards[i] = reward;
				dones[i] = 0;

				continue;
			}

			// ====================================================
			// 4. 행동 후 최단거리 계산
			// ====================================================

			p1Distances[i] = AStar(
				board,
				true
			);

			p2Distances[i] = AStar(
				board,
				false
			);

			// ====================================================
			// 5. 현재 플레이어 / 상대방 거리 선택
			// ====================================================

			const int currentMyDistance =
				(actingPlayer == 1)
				? p1Distances[i]
				: p2Distances[i];

			const int currentOpponentDistance =
				(actingPlayer == 1)
				? p2Distances[i]
				: p1Distances[i];

			const int previousMyDistance =
				(actingPlayer == 1)
				? prevP1Distances[i]
				: prevP2Distances[i];

			const int previousOpponentDistance =
				(actingPlayer == 1)
				? prevP2Distances[i]
				: prevP1Distances[i];

			// ====================================================
			// 내 거리 감소     -> +
			// 내 거리 증가     -> -
			//
			// 상대 거리 증가   -> +
			// 상대 거리 감소   -> -
			// ====================================================

			// =========================================================
			// 거리 기반 보상
			// =========================================================

			float myProgress = 0.0f;
			float opponentProgress = 0.0f;

			if (previousMyDistance >= 0 &&
				currentMyDistance >= 0)
			{
				// 내 최단경로가 짧아지면 +보상
				myProgress =
					static_cast<float>(
						previousMyDistance - currentMyDistance
						);
			}

			if (previousOpponentDistance >= 0 &&
				currentOpponentDistance >= 0)
			{
				// 상대 최단경로가 길어지면 +보상
				opponentProgress =
					static_cast<float>(
						currentOpponentDistance -
						previousOpponentDistance
						);
			}

			// 너무 큰 거리 변화가 보상을 폭발시키지 않도록 제한
			myProgress =
				std::clamp(myProgress, -5.0f, 5.0f);

			opponentProgress =
				std::clamp(opponentProgress, -5.0f, 5.0f);

			// 내 진행을 더 중요하게
			reward += 1.0f * myProgress;

			// 상대방 방해
			reward += 0.5f * opponentProgress;

			// 벽 설치에는 작은 비용
			if (action >= 4)
			{
				reward -= 0.2f;
			}

			// ====================================================
			// 9. 게임 종료 판정
			// ====================================================

			const int winner = isGameOver(board);

			if (winner != 0)
			{
				// ------------------------------------------------
				// 승리
				// ------------------------------------------------

				if (winner == actingPlayer)
				{
					reward += 150.0f;
				}

				// ------------------------------------------------
				// 상대방 승리
				//
				// 일반적으로 이 경우는 상대가 이전 행동에서
				// 이미 게임을 끝냈어야 하므로 자주 발생하지 않음.
				// 그래도 안전하게 처리.
				// ------------------------------------------------

				else
				{
					reward -= 150.0f;
				}

				dones[i] = 1;
			}

			// ====================================================
			// 10. 게임 진행 중
			// ====================================================

			else
			{
				dones[i] = 0;
			}

			// ====================================================
			// 11. 최종 reward 저장
			// ====================================================

			rewards[i] = reward;
		}
	}

	// ============================================================
	// 결과 반환
	// ============================================================

	return std::make_tuple(
		rewards,
		dones,
		p1Distances,
		p2Distances
	);
}

std::vector<int> GetValidActions(Quoridor* Board) {
	//std::array<uint8_t, 2> Player = Board->isP1Turn ? Board->P1 : Board->P2;
	int MoveCount = 0;
	int WallHCount = 0;
	int WallVCount = 0;
	std::vector<int> ValidActions;
	for (int i = 0; i < 4; i++) {
		if (canMovePlayer(Board, i)) {
			//이동 가능
			ValidActions.emplace_back(i);
			//printf("Move Possible: %d\n", i);
		}
	}
	std::array<uint8_t, 2> GridPos = { 0,0 };
	for (uint8_t i = 0; i < 64; i++) {
		GridPos.at(0) = i / 8;
		GridPos.at(1) = i % 8;
		if (canPlaceWallH(Board, GridPos)) {
			//가로벽 설치 가능
			ValidActions.emplace_back(i + 4);
			//printf("WallPLaceH Possible: %d\n", i + 4);
		}
		if (canPlaceWallV(Board, GridPos)) {
			//세로벽 설치 가능
			ValidActions.emplace_back(i + 68);
			//printf("WallPLaceVt Possible: %d\n", i + 68);
		}
	}

    if (ValidActions.empty()) {
		Board->PrintBoard();
        printf("\n");
        printf("========== NO VALID ACTION ==========\n");

        printf("Turn: P%d\n",
            Board->isP1Turn ? 1 : 2);

        printf("P1: (%d, %d)\n",
            Board->P1[0],
            Board->P1[1]);

        printf("P2: (%d, %d)\n",
            Board->P2[0],
            Board->P2[1]);

        printf("WallCount: P1=%d P2=%d\n",
            Board->WallCount[0],
            Board->WallCount[1]);

        printf("MoveCount = %d\n", MoveCount);
        printf("WallHCount = %d\n", WallHCount);
        printf("WallVCount = %d\n", WallVCount);

        printf("Path around current player:\n");

        std::array<uint8_t, 2> Player =
            Board->isP1Turn ? Board->P1 : Board->P2;

        printf("Position = (%d, %d)\n",
            Player[0],
            Player[1]);

        printf("Path[up]    = %d\n",
            Board->Path[Player[0]][Player[1]][0]);

        printf("Path[down]  = %d\n",
            Board->Path[Player[0]][Player[1]][1]);

        printf("Path[left]  = %d\n",
            Board->Path[Player[0]][Player[1]][2]);

        printf("Path[right] = %d\n",
            Board->Path[Player[0]][Player[1]][3]);

        printf("=====================================\n\n");
    }

	return ValidActions;
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
	printf("Distance: %d\n\n", AStar(&Board, Board.isP1Turn));
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
	omp_set_dynamic(0);
	omp_set_num_threads(16);
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
	m.def("isConnected", &isConnected, "checks if the players are possible to reach destination");
	m.def("canPlaceWallH", &canPlaceWallH, "checks if placing a horizontal wall is valid");
	m.def("canPlaceWallV", &canPlaceWallV, "checks if placing a vertical wall is valid");
	m.def("canMovePlayer", &canMovePlayer, "checks if moving a player");
	m.def("PlaceWallH", &PlaceWallH, "places horizontal wall");
	m.def("PlaceWallV", &PlaceWallV, "places vertical wall");
	m.def("PrintBoard", &Quoridor::PrintBoard, "prints current Board");
	m.def("MovePlayer", &MovePlayer, "moves player");
	m.def("AStar", &AStar, "finds the distance between Player and destination");
	m.def("AStar_withPath", &AStar_withPath, "finds the nodes between Player and destination");
	m.def("isGameOver", &isGameOver, "checks if the game is over and returns the winner");
	m.def("GetValidActions", &GetValidActions, "gets a vector of valid actions");
	m.def("AStarNextMove", &AStarNextMove);
	m.def(
		"StepBatch",
		[](pybind11::list boards,
			const std::vector<int>& actions,
			const std::vector<int>& actingPlayers,
			const std::vector<int>& prevP1Distances,
			const std::vector<int>& prevP2Distances)
		{
			return StepBatchImpl(
				boards,
				actions,
				actingPlayers,
				prevP1Distances,
				prevP2Distances
			);
		},
		"Execute multiple Quoridor environments in parallel"
	);
}