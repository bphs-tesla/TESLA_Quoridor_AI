import math
import numpy as np
import Quoridor


# ============================================================
# Configuration
# ============================================================

ACTION_SIZE = 132

C_PUCT = 1.5

MCTS_BATCH_SIZE = 32


# ============================================================
# Action
# ============================================================

def apply_action(board, action):

    action = int(action)

    # --------------------------------------------------------
    # Move
    # --------------------------------------------------------

    if 0 <= action < 4:

        result = Quoridor.MovePlayer(
            board,
            action
        )

        if result != 0:
            raise RuntimeError(
                f"Illegal move: {action}"
            )

        return

    # --------------------------------------------------------
    # Horizontal wall
    # --------------------------------------------------------

    if 4 <= action < 68:

        index = action - 4

        row = index // 8
        col = index % 8

        result = Quoridor.PlaceWallH(
            board,
            [row, col]
        )

        if result != 0:
            raise RuntimeError(
                f"Illegal horizontal wall: {action}"
            )

        return

    # --------------------------------------------------------
    # Vertical wall
    # --------------------------------------------------------

    if 68 <= action < 132:

        index = action - 68

        row = index // 8
        col = index % 8

        result = Quoridor.PlaceWallV(
            board,
            [row, col]
        )

        if result != 0:
            raise RuntimeError(
                f"Illegal vertical wall: {action}"
            )

        return

    raise ValueError(
        f"Invalid action: {action}"
    )


# ============================================================
# Terminal
# ============================================================

def terminal_value(board, player):

    winner = int(
        Quoridor.isGameOver(board)
    )

    if winner == 0:
        return None

    if winner == player:
        return 1.0

    return -1.0


# ============================================================
# Node
# ============================================================
# ============================================================
# MCTS Node
# ============================================================
# ============================================================
# MCTS Node
# ============================================================
# ============================================================
# MCTS Node
# ============================================================

class MCTSNode:

    def __init__(
        self,
        board,
        player,
        parent=None,
        action=None,
        prior=0.0
    ):

        self.board = board

        # 이 node에서 행동할 player
        self.player = int(player)

        self.parent = parent

        # 이 node로 들어오기 위해 부모에서 선택한 action
        self.action = action

        # Neural Network policy prior
        self.prior = float(prior)

        # MCTS 방문 통계
        self.visit_count = 0
        self.value_sum = 0.0

        # 자식 node
        self.children = {}

        # 현재 node가 expansion 되었는지
        self.expanded = False

        # GetValidActions() 결과 캐시
        self.valid_actions = None

        # Batched MCTS virtual loss
        self.virtual_loss = 0

        # ====================================================
        # Batched MCTS용
        # ====================================================

        # GetValidActions() 결과를 캐싱
        #
        # None:
        #   아직 GetValidActions()를 호출하지 않음
        #
        # list:
        #   이미 해당 board의 valid action을 계산함
        self.valid_actions = None

        # Batched MCTS에서 서로 다른 simulation이
        # 같은 leaf를 동시에 선택하는 것을 방지하기 위한
        # virtual loss
        self.virtual_loss = 0

    @property
    def value(self):

        if self.visit_count == 0:
            return 0.0

        return (
            self.value_sum /
            self.visit_count
        )
        
# ============================================================
# Neural Network
# ============================================================

class NeuralNetwork:

    def __init__(self, model):

        self.model = model

    def predict(self, boards):

        if len(boards) == 0:

            return (
                np.empty(
                    (0, ACTION_SIZE),
                    dtype=np.float32
                ),
                np.empty(
                    (0,),
                    dtype=np.float32
                )
            )

        # ----------------------------------------------------
        # 각 board의 현재 player
        # ----------------------------------------------------

        players = np.asarray(
            [
                1 if board.isP1Turn else 2
                for board in boards
            ],
            dtype=np.int32
        )

        # ----------------------------------------------------
        # C++ batch state
        # ----------------------------------------------------

        states = Quoridor.GetStateBatch(
            boards,
            players.tolist()
        )

        states = np.asarray(
            states,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # ResNet
        # ----------------------------------------------------

        policy_logits, values = self.model(
            states,
            training=False
        )

        policy_logits = np.asarray(
            policy_logits,
            dtype=np.float32
        )

        values = np.asarray(
            values,
            dtype=np.float32
        ).reshape(-1)

        return (
            policy_logits,
            values
        )


# ============================================================
# Masked Softmax
# ============================================================

def masked_softmax(
    logits,
    valid_actions
):

    valid_actions = np.asarray(
        valid_actions,
        dtype=np.int32
    )

    result = np.zeros(
        ACTION_SIZE,
        dtype=np.float32
    )

    if len(valid_actions) == 0:
        return result

    valid_logits = (
        logits[valid_actions]
    )

    max_logit = np.max(
        valid_logits
    )

    exp_logits = np.exp(
        valid_logits - max_logit
    )

    sum_exp = np.sum(
        exp_logits
    )

    if sum_exp <= 0:

        result[valid_actions] = (
            1.0 / len(valid_actions)
        )

    else:

        result[valid_actions] = (
            exp_logits / sum_exp
        )

    return result


# ============================================================
# Expand
# ============================================================

def expand_node(
    node,
    policy_logits
):

    valid_actions = Quoridor.GetValidActions(
        node.board
    )

    valid_actions = [
        int(action)
        for action in valid_actions
    ]

    if len(valid_actions) == 0:

        raise RuntimeError(
            "No valid actions."
        )

    policy = masked_softmax(
        policy_logits,
        valid_actions
    )

    for action in valid_actions:

        child_board = Quoridor.CloneBoard(
            node.board
        )

        apply_action(
            child_board,
            action
        )

        child = MCTSNode(
            child_board,
            parent=node,
            action=action,
            prior=policy[action]
        )

        node.children[action] = child

    node.expanded = True


# ============================================================
# PUCT Selection
# ============================================================

def select_child(node):

    best_score = -float("inf")

    best_action = None

    best_child = None

    sqrt_parent_visits = math.sqrt(
        max(1, node.visit_count)
    )

    for action, child in node.children.items():

        # ----------------------------------------------------
        # Parent 관점의 Q
        # ----------------------------------------------------

        q = -child.value

        # ----------------------------------------------------
        # U
        # ----------------------------------------------------

        u = (
            C_PUCT
            * child.prior
            * sqrt_parent_visits
            / (1 + child.visit_count)
        )

        score = q + u

        if score > best_score:

            best_score = score
            best_action = action
            best_child = child

    return (
        best_action,
        best_child
    )


# ============================================================
# Leaf Selection
# ============================================================

def select_leaf(root):

    node = root

    search_path = [node]

    # --------------------------------------------------------
    # Tree traversal
    # --------------------------------------------------------

    while node.expanded:

        if len(node.children) == 0:
            break

        _, node = select_child(node)

        search_path.append(node)

    return node, search_path


# ============================================================
# Backpropagation
# ============================================================

def backpropagate(
    search_path,
    value
):

    for node in reversed(
        search_path
    ):

        node.visit_count += 1

        node.value_sum += value

        value = -value


# ============================================================
# Batched MCTS
# ============================================================

def MCTS_Search(
    board,
    network,
    simulations=800,
    batch_size=MCTS_BATCH_SIZE
):

    # ========================================================
    # Root
    # ========================================================

    root_board = Quoridor.CloneBoard(
        board
    )

    root = MCTSNode(
        root_board
    )

    # ========================================================
    # Root Evaluation
    # ========================================================

    policy_logits, values = network.predict(
        [root.board]
    )

    expand_node(
        root,
        policy_logits[0]
    )

    # ========================================================
    # Simulation
    # ========================================================

    completed = 0

    while completed < simulations:

        current_batch_size = min(
            batch_size,
            simulations - completed
        )

        leaves = []

        search_paths = []

        # ----------------------------------------------------
        # 여러 leaf 수집
        # ----------------------------------------------------

        for _ in range(
            current_batch_size
        ):

            leaf, search_path = select_leaf(
                root
            )

            # Terminal이면 즉시 backprop
            terminal = terminal_value(
                leaf.board,
                leaf.player
            )

            if terminal is not None:

                backpropagate(
                    search_path,
                    terminal
                )

                completed += 1

                continue

            leaves.append(
                leaf
            )

            search_paths.append(
                search_path
            )

        # ----------------------------------------------------
        # Leaf가 없다면 계속
        # ----------------------------------------------------

        if len(leaves) == 0:
            continue

        # ----------------------------------------------------
        # ★ 핵심
        #
        # 여러 leaf를 한 번에 Neural Network에 전달
        # ----------------------------------------------------

        policy_logits, values = network.predict(
            [
                leaf.board
                for leaf in leaves
            ]
        )

        # ----------------------------------------------------
        # Expand + Backprop
        # ----------------------------------------------------

        for i, leaf in enumerate(
            leaves
        ):

            expand_node(
                leaf,
                policy_logits[i]
            )

            value = float(
                values[i]
            )

            backpropagate(
                search_paths[i],
                value
            )

            completed += 1

    # ========================================================
    # Visit count
    # ========================================================

    visit_counts = np.zeros(
        ACTION_SIZE,
        dtype=np.float32
    )

    for action, child in root.children.items():

        visit_counts[
            int(action)
        ] = float(
            child.visit_count
        )

    # ========================================================
    # Normalize
    # ========================================================

    total_visits = np.sum(
        visit_counts
    )

    if total_visits > 0:

        policy = (
            visit_counts
            / total_visits
        )

    else:

        policy = np.zeros(
            ACTION_SIZE,
            dtype=np.float32
        )

    return (
        policy,
        root
    )