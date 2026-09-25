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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

//#include <BaseTsd.h>
#include <cstddef>

std::ptrdiff_t SSIZE_T;

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
		// 모든 이동 경로를 우선 true로 초기화
		std::fill(
			&Path[0][0][0],
			&Path[0][0][0] + 9 * 9 * 4,
			true
		);

		// 보드 경계
		for (int i = 0; i < 9; i++) {
			Path[0][i][0] = false; // 위
			Path[8][i][1] = false; // 아래
			Path[i][0][2] = false; // 왼쪽
			Path[i][8][3] = false; // 오른쪽
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

	static constexpr int dr[4] = { -1, 1, 0, 0 };
	static constexpr int dc[4] = { 0, 0, -1, 1 };

	const int r = static_cast<int>(Start[0]);
	const int c = static_cast<int>(Start[1]);

	// 현재 칸에서 해당 방향으로 이동 가능한지
	if (!Board->Path[r][c][Move])
		return false;

	const int nr = r + dr[Move];
	const int nc = c + dc[Move];

	if (nr < 0 || nr >= 9 || nc < 0 || nc >= 9)
		return false;

	const std::array<uint8_t, 2> Next = {
		static_cast<uint8_t>(nr),
		static_cast<uint8_t>(nc)
	};

	const std::array<uint8_t, 2>& Opponent =
		movingP1 ? Board->P2 : Board->P1;


	// =========================================================
	// 1. 일반 이동
	// =========================================================

	if (Next != Opponent)
	{
		Destination = Next;
		return true;
	}


	// =========================================================
	// 2. 상대방이 바로 앞에 있음
	//    -> 상대방을 뛰어넘을 수 있는지 확인
	// =========================================================

	const int jr = static_cast<int>(Opponent[0]) + dr[Move];
	const int jc = static_cast<int>(Opponent[1]) + dc[Move];

	// 상대방 뒤쪽이 보드 밖이면 직선 점프 불가능
	if (jr >= 0 && jr < 9 &&
		jc >= 0 && jc < 9)
	{
		// 상대방 -> 뒤쪽 칸으로 이동 가능
		if (Board->Path[Opponent[0]][Opponent[1]][Move])
		{
			Destination = {
				static_cast<uint8_t>(jr),
				static_cast<uint8_t>(jc)
			};

			return true;
		}
	}


	// =========================================================
	// 3. 직선 점프가 막혀 있음
	//    -> 대각선 우회
	// =========================================================

	uint8_t SideMoves[2];

	if (Move == 0 || Move == 1)
	{
		// 위/아래로 접근했으면
		// 왼쪽/오른쪽으로 우회
		SideMoves[0] = 2;
		SideMoves[1] = 3;
	}
	else
	{
		// 왼쪽/오른쪽으로 접근했으면
		// 위/아래로 우회
		SideMoves[0] = 0;
		SideMoves[1] = 1;
	}


	for (int i = 0; i < 2; ++i)
	{
		const uint8_t SideMove = SideMoves[i];

		// 상대방 위치에서 옆으로 갈 수 있는가?
		if (!Board->Path[Opponent[0]][Opponent[1]][SideMove])
			continue;

		const int sr =
			static_cast<int>(Opponent[0]) + dr[SideMove];

		const int sc =
			static_cast<int>(Opponent[1]) + dc[SideMove];

		if (sr < 0 || sr >= 9 ||
			sc < 0 || sc >= 9)
			continue;

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

	if (Path.empty())
	{
		std::cout
			<< "\n========== AStarNextMove FAILED ==========\n";

		std::cout
			<< "Player: "
			<< (movingP1 ? 1 : 2)
			<< "\n";

		Board->PrintBoard();

		std::cout
			<< "===========================================\n";

		return -1;
	}

	if (Path.size() < 2)
	{
		return -1;
	}

	const auto& Current = Path[0];
	const auto& Next = Path[1];

	int dr =
		static_cast<int>(Next[0]) -
		static_cast<int>(Current[0]);

	int dc =
		static_cast<int>(Next[1]) -
		static_cast<int>(Current[1]);

	/*
		일반적인 1칸 이동
	*/

	if (dr == -1 && dc == 0)
		return 0;   // UP

	if (dr == 1 && dc == 0)
		return 1;   // DOWN

	if (dr == 0 && dc == -1)
		return 2;   // LEFT

	if (dr == 0 && dc == 1)
		return 3;   // RIGHT


	/*
		2칸 점프
	*/

	if (dr < 0 && dc == 0)
		return 0;

	if (dr > 0 && dc == 0)
		return 1;

	if (dr == 0 && dc < 0)
		return 2;

	if (dr == 0 && dc > 0)
		return 3;

	return -1;
}

bool isConnected(Quoridor& Board)
{
	return
		AStar(&Board, true) != -1 &&
		AStar(&Board, false) != -1;
}

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

// action:
// 0 = UP
// 1 = DOWN
// 2 = LEFT
// 3 = RIGHT
//
// AStar_withPath가 반환하는 path는
// {현재 위치, 다음 위치, 다음 위치, ...} 형태라고 가정
int GetAStarDirectionReward(
	Quoridor* Board,
	bool movingP1,
	int action
)
{
	// 이동 행동이 아니면 보상 없음
	if (action < 0 || action >= 4)
		return 0;

	// 현재 A* 최단 경로
	std::vector<std::array<uint8_t, 2>> Path =
		AStar_withPath(Board, movingP1);

	if (Path.size() < 2)
		return 0;

	const auto& AStarNext = Path[1];

	// 현재 플레이어 위치
	const std::array<uint8_t, 2> Current =
		movingP1 ? Board->P1 : Board->P2;

	// 이 action을 실제로 했을 때 도착하는 위치
	std::array<uint8_t, 2> ActionDestination;

	if (!GetMoveDestination(
		Board,
		Current,
		static_cast<uint8_t>(action),
		ActionDestination,
		movingP1))
	{
		return 0;
	}

	// A*가 선택한 다음 위치와
	// DQN action의 실제 도착 위치가 같으면
	// A* 방향을 따른 것
	if (ActionDestination == AStarNext)
		return 1;

	return 0;
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

struct MCTSNode
{
	Quoridor Board;

	int parent;
	int action_from_parent;

	std::vector<int> actions;

	std::vector<float> P;
	std::vector<float> Q;
	std::vector<int> N;

	bool expanded;
	bool terminal;
};

Quoridor CloneBoard(const Quoridor& Board)
{
	return Board;
}

std::vector<float> MCTS_Search(
	Quoridor* Board,
	int simulations
);
pybind11::array_t<float> GetStateBatch(
	const std::vector<Quoridor>& Boards,
	const std::vector<int>& Players
)
{
	const ssize_t N = static_cast<ssize_t>(Boards.size());

	if (Players.size() != Boards.size())
	{
		throw std::runtime_error(
			"GetStateBatch: Boards.size() != Players.size()"
		);
	}

	// ========================================================
	// Output
	//
	// Shape = (N, 9, 9, 7)
	//
	// Channel
	// 0 : My Player
	// 1 : Opponent
	// 2 : Horizontal Wall
	// 3 : Vertical Wall
	// 4 : My Turn
	// 5 : My Wall Count
	// 6 : Opponent Wall Count
	// ========================================================

	pybind11::array_t<float> result(
		{
			N,
			static_cast<ssize_t>(9),
			static_cast<ssize_t>(9),
			static_cast<ssize_t>(7)
		}
	);

	auto buffer = result.request();

	float* data = static_cast<float*>(buffer.ptr);

	// ========================================================
	// Batch
	// ========================================================

#pragma omp parallel for
	for (ssize_t n = 0; n < N; ++n)
	{
		const Quoridor& Board = Boards[n];

		const int Player = Players[n];

		// 잘못된 Player
		if (Player != 1 && Player != 2)
		{
			continue;
		}

		// ====================================================
		// Player 기준으로 보드 해석
		// ====================================================

		const std::array<uint8_t, 2>& MyPosition =
			(Player == 1)
			? Board.P1
			: Board.P2;

		const std::array<uint8_t, 2>& OpponentPosition =
			(Player == 1)
			? Board.P2
			: Board.P1;

		const uint8_t MyWallCount =
			(Player == 1)
			? Board.WallCount[0]
			: Board.WallCount[1];

		const uint8_t OpponentWallCount =
			(Player == 1)
			? Board.WallCount[1]
			: Board.WallCount[0];

		const float MyTurn =
			(Board.isP1Turn == (Player == 1))
			? 1.0f
			: 0.0f;

		// ====================================================
		// Batch 하나의 시작 위치
		//
		// [9][9][7]
		// ====================================================

		const ssize_t base =
			n * 9 * 9 * 7;

		// ====================================================
		// 전체 0 초기화
		// ====================================================

		for (int x = 0; x < 9; ++x)
		{
			for (int y = 0; y < 9; ++y)
			{
				for (int c = 0; c < 7; ++c)
				{
					data[
						base
						+ x * 9 * 7
						+ y * 7
						+ c
					] = 0.0f;
				}
			}
		}

		// ====================================================
		// Channel 0
		//
		// My Player
		// ====================================================

		data[
			base
			+ MyPosition[0] * 9 * 7
			+ MyPosition[1] * 7
			+ 0
		] = 1.0f;

		// ====================================================
		// Channel 1
		//
		// Opponent
		// ====================================================

		data[
			base
			+ OpponentPosition[0] * 9 * 7
			+ OpponentPosition[1] * 7
			+ 1
		] = 1.0f;

		// ====================================================
		// Channel 2
		//
		// Horizontal Wall
		//
		// 실제 저장 방식:
		//
		// WallH[x] |= (1 << y)
		//
		// 따라서:
		//
		// WallH[x] & (1 << y)
		//
		// 로 검사해야 함.
		// ====================================================

		for (int x = 0; x < 8; ++x)
		{
			for (int y = 0; y < 8; ++y)
			{
				const uint8_t mask =
					static_cast<uint8_t>(1u << y);

				if ((Board.WallH[x] & mask) != 0)
				{
					data[
						base
						+ x * 9 * 7
						+ y * 7
						+ 2
					] = 1.0f;
				}
			}
		}

		// ====================================================
		// Channel 3
		//
		// Vertical Wall
		//
		// 실제 저장 방식:
		//
		// WallVt[y] |= (1 << x)
		//
		// 따라서:
		//
		// WallVt[y] & (1 << x)
		// ====================================================

		for (int x = 0; x < 8; ++x)
		{
			for (int y = 0; y < 8; ++y)
			{
				const uint8_t mask =
					static_cast<uint8_t>(1u << x);

				if ((Board.WallVt[y] & mask) != 0)
				{
					data[
						base
						+ x * 9 * 7
						+ y * 7
						+ 3
					] = 1.0f;
				}
			}
		}

		// ====================================================
		// Channel 4
		//
		// My Turn
		//
		// 9x9 전체에 broadcast
		// ====================================================

		for (int x = 0; x < 9; ++x)
		{
			for (int y = 0; y < 9; ++y)
			{
				data[
					base
					+ x * 9 * 7
					+ y * 7
					+ 4
				] = MyTurn;
			}
		}

		// ====================================================
		// Channel 5
		//
		// My Wall Count
		// ====================================================

		const float MyWall =
			static_cast<float>(MyWallCount) / 10.0f;

		for (int x = 0; x < 9; ++x)
		{
			for (int y = 0; y < 9; ++y)
			{
				data[
					base
					+ x * 9 * 7
					+ y * 7
					+ 5
				] = MyWall;
			}
		}

		// ====================================================
		// Channel 6
		//
		// Opponent Wall Count
		// ====================================================

		const float OppWall =
			static_cast<float>(OpponentWallCount) / 10.0f;

		for (int x = 0; x < 9; ++x)
		{
			for (int y = 0; y < 9; ++y)
			{
				data[
					base
					+ x * 9 * 7
					+ y * 7
					+ 6
				] = OppWall;
			}
		}
	}

	return result;
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

	m.def("MovePlayer", &MovePlayer);
	m.def("PlaceWallH", &PlaceWallH);
	m.def("PlaceWallV", &PlaceWallV);
	m.def("GetValidActions", &GetValidActions);
	m.def("isGameOver", &isGameOver);
	m.def("CloneBoard", &CloneBoard);
	m.def(
		"GetStateBatch",
		&GetStateBatch,
		pybind11::arg("Boards"),
		pybind11::arg("Players")
	);
}