def SelfPlay(
    model,
    simulations=800
):

    board = Quoridor.Quoridor()

    network = NeuralNetwork(
        model
    )

    game_history = []

    MAX_MOVES = 200
    move_count = 0

    # 0  = 게임 진행 중
    # 1  = P1 승리
    # 2  = P2 승리
    # -1 = 무승부 (200수 초과)
    winner = 0

    while True:

        # ----------------------------------------------------
        # 현재 플레이어
        # ----------------------------------------------------

        player = (
            1
            if board.isP1Turn
            else 2
        )

        # ----------------------------------------------------
        # 먼저 현재 게임이 끝났는지 확인
        # ----------------------------------------------------

        winner = int(
            Quoridor.isGameOver(
                board
            )
        )

        if winner != 0:

            break

        # ----------------------------------------------------
        # 200수 제한
        # ----------------------------------------------------

        if move_count >= MAX_MOVES:

            winner = -1

            print(
                f"[SelfPlay] "
                f"Game reached {MAX_MOVES} moves. "
                f"Draw."
            )

            break

        # ----------------------------------------------------
        # 현재 state
        # ----------------------------------------------------

        state = np.asarray(
            Quoridor.GetStateBatch(
                [board],
                [player]
            )[0],
            dtype=np.float32
        )
        # ----------------------------------------------------
        # MCTS
        # ----------------------------------------------------

        policy, root = MCTS_Search(
            board,
            network,
            simulations
        )

        # ----------------------------------------------------
        # Policy 확인
        # ----------------------------------------------------

        policy = np.asarray(
            policy,
            dtype=np.float64
        )

        # 혹시 수치 오차로 합이 1이 아닌 경우
        policy_sum = policy.sum()

        if (
            not np.isfinite(policy_sum)
            or policy_sum <= 0
        ):

            valid_actions = [
                int(a)
                for a in Quoridor.GetValidActions(
                    board
                )
            ]

            policy = np.zeros(
                ACTION_SIZE,
                dtype=np.float64
            )

            policy[
                valid_actions
            ] = 1.0 / len(valid_actions)

        else:

            policy /= policy_sum

        # ----------------------------------------------------
        # 데이터 저장
        # ----------------------------------------------------

        game_history.append(
            (
                state,
                policy.astype(np.float32),
                player
            )
        )

        # ----------------------------------------------------
        # Action 선택
        # ----------------------------------------------------

        action = np.random.choice(
            ACTION_SIZE,
            p=policy
        )

        action = int(
            action
        )

        # ----------------------------------------------------
        # Action 적용
        # ----------------------------------------------------

        result = apply_action(
            board,
            action
        )

        if result != 0:

            raise RuntimeError(
                f"Invalid action selected by MCTS: "
                f"{action}"
            )

        # ----------------------------------------------------
        # 수 증가
        # ----------------------------------------------------

        move_count += 1

        # ----------------------------------------------------
        # 게임 종료 확인
        #
        # 반드시 승리 여부를 먼저 확인한 뒤
        # 200수 제한을 확인한다.
        #
        # 따라서 정확히 200번째 수로 승리하면
        # 정상적으로 승리 처리된다.
        # ----------------------------------------------------

        winner = int(
            Quoridor.isGameOver(
                board
            )
        )

        if winner != 0:

            break

        if move_count >= MAX_MOVES:

            winner = -1

            print(
                f"[SelfPlay] "
                f"Game reached {MAX_MOVES} moves. "
                f"Draw."
            )

            break

    # ========================================================
    # 결과 z 생성
    # ========================================================

    training_data = []

    for state, policy, player in game_history:

        # ----------------------------------------------------
        # 정상적인 승리
        # ----------------------------------------------------

        if winner == 1 or winner == 2:

            if player == winner:

                z = 1.0

            else:

                z = -1.0

        # ----------------------------------------------------
        # 200수 무승부
        # ----------------------------------------------------

        elif winner == -1:

            z = 0.0

        # ----------------------------------------------------
        # 예외적인 경우
        # ----------------------------------------------------

        else:

            z = 0.0

        training_data.append(
            (
                state,
                policy,
                z
            )
        )

    # --------------------------------------------------------
    # 결과 출력
    # --------------------------------------------------------

    if winner == -1:

        print(
            f"[SelfPlay] "
            f"Draw after {move_count} moves."
        )

    else:

        print(
            f"[SelfPlay] "
            f"Player {winner} wins "
            f"after {move_count} moves."
        )

    return training_data