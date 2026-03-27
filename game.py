"""
Quoridor (Barricade) Game Engine
High-performance game logic with Zobrist hashing for transposition tables.
"""

import random
from collections import deque
from copy import deepcopy
from typing import List, Tuple, Optional, Set

# Board is 9x9 for pawns
BOARD_SIZE = 9
# Walls exist on the gaps between cells: 8x8 grid of possible wall centers
WALL_GRID = BOARD_SIZE - 1
WALLS_PER_PLAYER = 10

# Directions: (row_delta, col_delta)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
DIR_NAMES = ['up', 'down', 'left', 'right']

# Zobrist keys - initialized once
random.seed(42)
ZOBRIST_PAWN = [[random.getrandbits(64) for _ in range(BOARD_SIZE * BOARD_SIZE)] for _ in range(2)]
ZOBRIST_HWALL = [random.getrandbits(64) for _ in range(WALL_GRID * WALL_GRID)]
ZOBRIST_VWALL = [random.getrandbits(64) for _ in range(WALL_GRID * WALL_GRID)]
ZOBRIST_TURN = random.getrandbits(64)
ZOBRIST_WALLS_LEFT = [[random.getrandbits(64) for _ in range(WALLS_PER_PLAYER + 1)] for _ in range(2)]


class Move:
    """Represents a game move - either a pawn move or wall placement."""
    __slots__ = ['move_type', 'row', 'col', 'orientation']

    PAWN = 0
    WALL = 1

    def __init__(self, move_type: int, row: int, col: int, orientation: str = ''):
        self.move_type = move_type
        self.row = row
        self.col = col
        self.orientation = orientation  # 'h' or 'v' for walls

    def __repr__(self):
        if self.move_type == Move.PAWN:
            return f"Pawn({self.row},{self.col})"
        return f"Wall({self.row},{self.col},{self.orientation})"

    def __eq__(self, other):
        if other is None:
            return False
        return (self.move_type == other.move_type and self.row == other.row
                and self.col == other.col and self.orientation == other.orientation)

    def __hash__(self):
        return hash((self.move_type, self.row, self.col, self.orientation))


class QuoridorState:
    """
    Full game state for Quoridor.

    Board coordinates:
    - Pawns: (row, col) on 9x9 grid, row 0 = top, row 8 = bottom
    - Walls: (row, col) is the top-left cell of the 2x2 block the wall occupies
      - Horizontal wall at (r,c) blocks movement between rows r and r+1 at columns c and c+1
      - Vertical wall at (r,c) blocks movement between columns c and c+1 at rows r and r+1

    Player 0 starts at row 0 (top), goal is row 8 (bottom)
    Player 1 starts at row 8 (bottom), goal is row 0 (top)
    """

    def __init__(self):
        # Pawn positions: [row, col] for each player
        self.pawns = [[0, 4], [8, 4]]
        # Goal rows
        self.goals = [8, 0]
        # Walls remaining
        self.walls_left = [WALLS_PER_PLAYER, WALLS_PER_PLAYER]
        # Placed walls: set of (row, col, orientation)
        self.h_walls: Set[Tuple[int, int]] = set()  # horizontal walls
        self.v_walls: Set[Tuple[int, int]] = set()  # vertical walls
        # Current player (0 or 1)
        self.current_player = 0
        # Move history for unmake
        self.history: List = []
        # Zobrist hash
        self.zobrist = self._compute_zobrist()
        # Game over flag
        self.winner = -1

    def _compute_zobrist(self) -> int:
        h = 0
        for p in range(2):
            idx = self.pawns[p][0] * BOARD_SIZE + self.pawns[p][1]
            h ^= ZOBRIST_PAWN[p][idx]
            h ^= ZOBRIST_WALLS_LEFT[p][self.walls_left[p]]
        for (r, c) in self.h_walls:
            h ^= ZOBRIST_HWALL[r * WALL_GRID + c]
        for (r, c) in self.v_walls:
            h ^= ZOBRIST_VWALL[r * WALL_GRID + c]
        if self.current_player == 1:
            h ^= ZOBRIST_TURN
        return h

    def copy(self) -> 'QuoridorState':
        s = QuoridorState.__new__(QuoridorState)
        s.pawns = [list(self.pawns[0]), list(self.pawns[1])]
        s.goals = list(self.goals)
        s.walls_left = list(self.walls_left)
        s.h_walls = set(self.h_walls)
        s.v_walls = set(self.v_walls)
        s.current_player = self.current_player
        s.history = []
        s.zobrist = self.zobrist
        s.winner = self.winner
        return s

    def _is_blocked(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if movement from (r1,c1) to (r2,c2) is blocked by a wall."""
        dr = r2 - r1
        dc = c2 - c1

        if dr == -1 and dc == 0:  # moving up
            # Blocked by horizontal wall at (r1-1, c1) or (r1-1, c1-1)
            if (r1 - 1, c1) in self.h_walls:
                return True
            if c1 > 0 and (r1 - 1, c1 - 1) in self.h_walls:
                return True
        elif dr == 1 and dc == 0:  # moving down
            # Blocked by horizontal wall at (r1, c1) or (r1, c1-1)
            if (r1, c1) in self.h_walls:
                return True
            if c1 > 0 and (r1, c1 - 1) in self.h_walls:
                return True
        elif dc == -1 and dr == 0:  # moving left
            # Blocked by vertical wall at (r1, c1-1) or (r1-1, c1-1)
            if (r1, c1 - 1) in self.v_walls:
                return True
            if r1 > 0 and (r1 - 1, c1 - 1) in self.v_walls:
                return True
        elif dc == 1 and dr == 0:  # moving right
            # Blocked by vertical wall at (r1, c1) or (r1-1, c1)
            if (r1, c1) in self.v_walls:
                return True
            if r1 > 0 and (r1 - 1, c1) in self.v_walls:
                return True
        return False

    def _can_move(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if a single step from (r1,c1) to (r2,c2) is possible (in bounds, no wall)."""
        if r2 < 0 or r2 >= BOARD_SIZE or c2 < 0 or c2 >= BOARD_SIZE:
            return False
        return not self._is_blocked(r1, c1, r2, c2)

    def get_pawn_moves(self) -> List[Move]:
        """Get all legal pawn moves for the current player, including jumps."""
        moves = []
        p = self.current_player
        opp = 1 - p
        r, c = self.pawns[p]
        or_, oc = self.pawns[opp]

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self._can_move(r, c, nr, nc):
                continue

            if nr == or_ and nc == oc:
                # Opponent is adjacent — try to jump over
                jr, jc = nr + dr, nc + dc
                if self._can_move(nr, nc, jr, jc):
                    moves.append(Move(Move.PAWN, jr, jc))
                else:
                    # Can't jump straight — try diagonal jumps
                    for dr2, dc2 in DIRECTIONS:
                        if (dr2, dc2) == (-dr, -dc):
                            continue  # don't jump back
                        sr, sc = nr + dr2, nc + dc2
                        if self._can_move(nr, nc, sr, sc) and not (sr == r and sc == c):
                            moves.append(Move(Move.PAWN, sr, sc))
            else:
                moves.append(Move(Move.PAWN, nr, nc))

        return moves

    def _walls_overlap(self, r: int, c: int, orientation: str) -> bool:
        """Check if placing a wall at (r,c) with given orientation overlaps existing walls."""
        if orientation == 'h':
            if (r, c) in self.h_walls:
                return True
            # Overlaps with adjacent horizontal walls
            if (r, c - 1) in self.h_walls or (r, c + 1) in self.h_walls:
                return True
            # Crosses a vertical wall at same position
            if (r, c) in self.v_walls:
                return True
        else:  # vertical
            if (r, c) in self.v_walls:
                return True
            if (r - 1, c) in self.v_walls or (r + 1, c) in self.v_walls:
                return True
            # Crosses a horizontal wall at same position
            if (r, c) in self.h_walls:
                return True
        return False

    def _bfs_path_exists(self, player: int) -> bool:
        """BFS to check if player can reach their goal row."""
        start = tuple(self.pawns[player])
        goal_row = self.goals[player]
        visited = set()
        visited.add(start)
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            if r == goal_row:
                return True
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited and not self._is_blocked(r, c, nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def shortest_path(self, player: int) -> int:
        """BFS shortest path length for player to reach goal row. Returns 999 if no path."""
        start = tuple(self.pawns[player])
        goal_row = self.goals[player]
        if start[0] == goal_row:
            return 0
        visited = set()
        visited.add(start)
        queue = deque([(start, 0)])

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited and not self._is_blocked(r, c, nr, nc):
                        if nr == goal_row:
                            return dist + 1
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
        return 999

    def get_wall_moves(self) -> List[Move]:
        """Get all legal wall placements for the current player."""
        p = self.current_player
        if self.walls_left[p] == 0:
            return []

        moves = []
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                for orientation in ('h', 'v'):
                    if not self._walls_overlap(r, c, orientation):
                        # Temporarily place wall and check paths
                        if orientation == 'h':
                            self.h_walls.add((r, c))
                        else:
                            self.v_walls.add((r, c))

                        if self._bfs_path_exists(0) and self._bfs_path_exists(1):
                            moves.append(Move(Move.WALL, r, c, orientation))

                        if orientation == 'h':
                            self.h_walls.remove((r, c))
                        else:
                            self.v_walls.remove((r, c))
        return moves

    def get_legal_moves(self) -> List[Move]:
        """Get all legal moves for the current player."""
        if self.winner >= 0:
            return []
        return self.get_pawn_moves() + self.get_wall_moves()

    def make_move(self, move: Move):
        """Apply a move and update state. Saves undo info in history."""
        p = self.current_player
        undo = {
            'player': p,
            'move': move,
            'old_pawn': list(self.pawns[p]),
            'old_zobrist': self.zobrist,
            'old_winner': self.winner,
        }
        self.history.append(undo)

        if move.move_type == Move.PAWN:
            # Update zobrist for pawn move
            old_idx = self.pawns[p][0] * BOARD_SIZE + self.pawns[p][1]
            new_idx = move.row * BOARD_SIZE + move.col
            self.zobrist ^= ZOBRIST_PAWN[p][old_idx]
            self.zobrist ^= ZOBRIST_PAWN[p][new_idx]

            self.pawns[p] = [move.row, move.col]

            # Check win
            if move.row == self.goals[p]:
                self.winner = p
        else:
            # Wall placement
            if move.orientation == 'h':
                self.h_walls.add((move.row, move.col))
                self.zobrist ^= ZOBRIST_HWALL[move.row * WALL_GRID + move.col]
            else:
                self.v_walls.add((move.row, move.col))
                self.zobrist ^= ZOBRIST_VWALL[move.row * WALL_GRID + move.col]

            self.zobrist ^= ZOBRIST_WALLS_LEFT[p][self.walls_left[p]]
            self.walls_left[p] -= 1
            self.zobrist ^= ZOBRIST_WALLS_LEFT[p][self.walls_left[p]]

        # Switch player
        self.zobrist ^= ZOBRIST_TURN
        self.current_player = 1 - p

    def unmake_move(self):
        """Undo the last move."""
        if not self.history:
            return
        undo = self.history.pop()
        move = undo['move']
        p = undo['player']

        self.current_player = p
        self.zobrist = undo['old_zobrist']
        self.winner = undo['old_winner']

        if move.move_type == Move.PAWN:
            self.pawns[p] = undo['old_pawn']
        else:
            if move.orientation == 'h':
                self.h_walls.discard((move.row, move.col))
            else:
                self.v_walls.discard((move.row, move.col))
            self.walls_left[p] += 1

    def is_terminal(self) -> bool:
        return self.winner >= 0

    def encode_board(self) -> list:
        """
        Encode board state as feature planes for NNUE input.
        Returns a flat list of features from the perspective of the current player.

        Features (total = 5 * 81 + 2 * 64 + 4 = 537):
        - Plane 0 (81): current player pawn position (one-hot on 9x9)
        - Plane 1 (81): opponent pawn position (one-hot on 9x9)
        - Plane 2 (81): current player goal row (binary mask)
        - Plane 3 (81): opponent goal row (binary mask)
        - Plane 4 (81): all pawn-reachable squares from current player (BFS flood)
        - Plane 5 (64): horizontal walls (on 8x8 grid)
        - Plane 6 (64): vertical walls (on 8x8 grid)
        - Scalar: current player walls remaining (normalized)
        - Scalar: opponent walls remaining (normalized)
        - Scalar: current player shortest path (normalized)
        - Scalar: opponent shortest path (normalized)
        """
        p = self.current_player
        opp = 1 - p
        features = []

        # Pawn positions as one-hot on 81
        for player in [p, opp]:
            plane = [0.0] * (BOARD_SIZE * BOARD_SIZE)
            plane[self.pawns[player][0] * BOARD_SIZE + self.pawns[player][1]] = 1.0
            features.extend(plane)

        # Goal rows as binary mask
        for player in [p, opp]:
            plane = [0.0] * (BOARD_SIZE * BOARD_SIZE)
            gr = self.goals[player]
            for c in range(BOARD_SIZE):
                plane[gr * BOARD_SIZE + c] = 1.0
            features.extend(plane)

        # Reachable squares from current player
        plane = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        visited = set()
        queue = deque([tuple(self.pawns[p])])
        visited.add(tuple(self.pawns[p]))
        while queue:
            r, c = queue.popleft()
            plane[r * BOARD_SIZE + c] = 1.0
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if (nr, nc) not in visited and not self._is_blocked(r, c, nr, nc):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        features.extend(plane)

        # Wall planes
        for wall_set in [self.h_walls, self.v_walls]:
            plane = [0.0] * (WALL_GRID * WALL_GRID)
            for (r, c) in wall_set:
                plane[r * WALL_GRID + c] = 1.0
            features.extend(plane)

        # Scalars
        features.append(self.walls_left[p] / WALLS_PER_PLAYER)
        features.append(self.walls_left[opp] / WALLS_PER_PLAYER)
        features.append(self.shortest_path(p) / 20.0)
        features.append(self.shortest_path(opp) / 20.0)

        return features

    def display(self):
        """Print the board to terminal."""
        # Build display grid
        # Each cell is 3 chars wide, walls are 1 char
        # Total width: 9*3 + 8*1 = 35
        lines = []
        col_header = "    " + "   ".join(str(c) for c in range(BOARD_SIZE))
        lines.append(col_header)

        for r in range(BOARD_SIZE):
            # Row of cells
            row_str = f" {r}  "
            for c in range(BOARD_SIZE):
                # Cell content
                if self.pawns[0] == [r, c]:
                    cell = ' 1 '
                elif self.pawns[1] == [r, c]:
                    cell = ' 2 '
                else:
                    cell = ' . '
                row_str += cell

                # Vertical wall between columns c and c+1
                if c < BOARD_SIZE - 1:
                    has_vwall = False
                    for wr in [r - 1, r]:
                        if 0 <= wr < WALL_GRID and (wr, c) in self.v_walls:
                            has_vwall = True
                            break
                    row_str += '|' if has_vwall else ' '

            lines.append(row_str)

            # Row of horizontal walls between rows r and r+1
            if r < BOARD_SIZE - 1:
                wall_str = "    "
                for c in range(BOARD_SIZE):
                    has_hwall = False
                    for wc in [c - 1, c]:
                        if 0 <= wc < WALL_GRID and (r, wc) in self.h_walls:
                            has_hwall = True
                            break
                    wall_str += '---' if has_hwall else '   '
                    if c < BOARD_SIZE - 1:
                        wall_str += '+'
                lines.append(wall_str)

        for line in lines:
            print(line)
        print(f"\nPlayer {self.current_player + 1} to move | "
              f"Walls: P1={self.walls_left[0]}, P2={self.walls_left[1]}")
        print(f"Shortest paths: P1={self.shortest_path(0)}, P2={self.shortest_path(1)}")


INPUT_SIZE = 5 * BOARD_SIZE * BOARD_SIZE + 2 * WALL_GRID * WALL_GRID + 4  # 537
