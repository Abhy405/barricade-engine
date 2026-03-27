"""
Alpha-Beta Search Engine for Quoridor.

Stockfish-inspired search with:
- Negamax framework with alpha-beta pruning
- Iterative deepening
- Transposition table (Zobrist hash based)
- Move ordering: pawn moves first, then walls near opponent
- Killer move heuristic
- History heuristic for move ordering
- Late move reductions (LMR) for wall placements
"""

import time
from typing import Optional, Tuple, Dict, List
from game import QuoridorState, Move, BOARD_SIZE
from nnue import NNUE, NNUEAccumulator

# Transposition table entry types
TT_EXACT = 0
TT_ALPHA = 1  # upper bound (failed low)
TT_BETA = 2   # lower bound (failed high)

INF = 999999.0


class TTEntry:
    __slots__ = ['zobrist', 'depth', 'score', 'flag', 'best_move']

    def __init__(self, zobrist: int, depth: int, score: float, flag: int, best_move: Optional[Move]):
        self.zobrist = zobrist
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = best_move


class SearchEngine:
    def __init__(self, model: NNUE, tt_size: int = 1 << 20):
        self.model = model
        self.model.eval()
        self.accumulator = NNUEAccumulator(model)

        # Transposition table
        self.tt_size = tt_size
        self.tt: Dict[int, TTEntry] = {}

        # Killer moves: indexed by depth, stores 2 killers per depth
        self.max_depth = 64
        self.killers: List[List[Optional[Move]]] = [[None, None] for _ in range(self.max_depth)]

        # History heuristic
        self.history: Dict[tuple, int] = {}

        # Search stats
        self.nodes = 0
        self.tt_hits = 0
        self.start_time = 0.0
        self.time_limit = 0.0

        # Use pure handcrafted until NNUE is properly trained
        self.nnue_weight = 0.0

    def _handcrafted_eval(self, state: QuoridorState) -> float:
        """
        Strong handcrafted evaluation for Quoridor.
        The key insight: shortest path difference is ~80% of what matters.
        """
        p = state.current_player
        opp = 1 - p
        my_dist = state.shortest_path(p)
        opp_dist = state.shortest_path(opp)

        # === WINNING / LOSING ===
        # If I can reach goal and it's my turn, massive bonus
        if my_dist == 0:
            return 900.0
        if opp_dist == 0:
            return -900.0

        # One move from winning with the move
        if my_dist == 1:
            return 500.0
        if opp_dist == 1 and state.walls_left[p] == 0:
            return -500.0  # can't block them

        score = 0.0

        # === PATH ADVANTAGE (dominant factor) ===
        path_diff = opp_dist - my_dist
        # Having the move matters: if paths equal, mover has advantage
        effective_diff = path_diff
        # More weight when distances are short (endgame)
        if my_dist <= 4 or opp_dist <= 4:
            score += effective_diff * 150.0
        else:
            score += effective_diff * 100.0

        # Tempo: being on the move with shorter/equal path is big
        if path_diff >= 0:
            score += 30.0
        # Even bigger when close to winning
        if my_dist <= 3 and path_diff >= 0:
            score += 50.0

        # === WALL ECONOMY ===
        my_walls = state.walls_left[p]
        opp_walls = state.walls_left[opp]

        # Walls are valuable for blocking — more valuable when opponent is close
        if opp_dist <= 4:
            score += my_walls * 20.0  # my walls can block them
        else:
            score += my_walls * 8.0

        if my_dist <= 4:
            score -= opp_walls * 20.0  # their walls can block me
        else:
            score -= opp_walls * 8.0

        # Having walls when opponent doesn't = huge advantage
        if my_walls > 0 and opp_walls == 0:
            score += 40.0
        if opp_walls > 0 and my_walls == 0:
            score -= 40.0

        # === POSITIONAL ===
        # Progress toward goal
        my_row = state.pawns[p][0]
        opp_row = state.pawns[opp][0]
        my_goal = state.goals[p]
        opp_goal = state.goals[opp]

        my_progress = abs(my_row - (8 - my_goal))  # rows advanced
        opp_progress = abs(opp_row - (8 - opp_goal))
        score += (my_progress - opp_progress) * 5.0

        # Center column control (columns 3,4,5 are better — more path options)
        my_col = state.pawns[p][1]
        opp_col = state.pawns[opp][1]
        center_bonus_me = max(0, 3 - abs(my_col - 4)) * 3.0
        center_bonus_opp = max(0, 3 - abs(opp_col - 4)) * 3.0
        score += center_bonus_me - center_bonus_opp

        # Normalize to roughly [-1, 1] range
        return score / 500.0

    def evaluate(self, state: QuoridorState) -> float:
        handcrafted = self._handcrafted_eval(state)
        if self.nnue_weight <= 0.0:
            return handcrafted

        features = state.encode_board()
        self.accumulator.full_refresh(features)
        nnue_score = self.accumulator.evaluate()
        return self.nnue_weight * nnue_score + (1.0 - self.nnue_weight) * handcrafted

    def _order_moves(self, moves: List[Move], state: QuoridorState,
                     tt_move: Optional[Move], depth: int) -> List[Move]:
        scored = []
        p = state.current_player
        goal = state.goals[p]
        pr, pc = state.pawns[p]

        for m in moves:
            score = 0

            if tt_move and m == tt_move:
                score = 10000000
            elif m.move_type == Move.PAWN:
                old_dist = abs(pr - goal)
                new_dist = abs(m.row - goal)
                score = 1000000 + (old_dist - new_dist) * 100000
                if m.row == goal:
                    score += 5000000
            else:
                if self.killers[depth][0] == m:
                    score = 500000
                elif self.killers[depth][1] == m:
                    score = 400000
                else:
                    key = (m.move_type, m.row, m.col, m.orientation)
                    score = self.history.get(key, 0)

                    opr, opc = state.pawns[1 - p]
                    dist_to_opp = abs(m.row - opr) + abs(m.col - opc)
                    score += max(0, 1000 - dist_to_opp * 100)

            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored]

    def _store_killer(self, move: Move, depth: int):
        if move.move_type == Move.WALL:
            if self.killers[depth][0] != move:
                self.killers[depth][1] = self.killers[depth][0]
                self.killers[depth][0] = move

    def _store_history(self, move: Move, depth: int):
        key = (move.move_type, move.row, move.col, move.orientation)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _time_up(self) -> bool:
        if self.time_limit <= 0:
            return False
        return time.time() - self.start_time >= self.time_limit

    def negamax(self, state: QuoridorState, depth: int, alpha: float, beta: float,
                ply: int) -> float:
        self.nodes += 1

        if self.nodes & 4095 == 0 and self._time_up():
            return 0.0

        if state.is_terminal():
            return -INF if state.winner == (1 - state.current_player) else INF

        if depth <= 0:
            return self.evaluate(state)

        # TT lookup
        tt_key = state.zobrist % self.tt_size
        tt_entry = self.tt.get(tt_key)
        tt_move = None

        if tt_entry and tt_entry.zobrist == state.zobrist:
            self.tt_hits += 1
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                if tt_entry.flag == TT_EXACT:
                    return tt_entry.score
                elif tt_entry.flag == TT_ALPHA and tt_entry.score <= alpha:
                    return alpha
                elif tt_entry.flag == TT_BETA and tt_entry.score >= beta:
                    return beta

        moves = state.get_legal_moves()
        if not moves:
            return self.evaluate(state)

        moves = self._order_moves(moves, state, tt_move, ply)

        best_score = -INF
        best_move = moves[0]
        flag = TT_ALPHA
        moves_searched = 0

        for move in moves:
            if self._time_up():
                break

            state.make_move(move)

            # LMR for late wall moves
            reduction = 0
            if (moves_searched >= 4 and depth >= 3
                    and move.move_type == Move.WALL
                    and not state.is_terminal()):
                reduction = 1

            score = -self.negamax(state, depth - 1 - reduction, -beta, -alpha, ply + 1)

            if reduction > 0 and score > alpha:
                score = -self.negamax(state, depth - 1, -beta, -alpha, ply + 1)

            state.unmake_move()
            moves_searched += 1

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                flag = TT_EXACT

                if score >= beta:
                    flag = TT_BETA
                    self._store_killer(move, ply)
                    self._store_history(move, depth)
                    break

        self.tt[tt_key] = TTEntry(state.zobrist, depth, best_score, flag, best_move)
        return best_score

    def search(self, state: QuoridorState, max_depth: int = 8,
               time_limit: float = 10.0) -> Tuple[Move, float]:
        """Iterative deepening search. Defaults to max depth 8, 10s."""
        self.nodes = 0
        self.tt_hits = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        self.history.clear()
        for i in range(self.max_depth):
            self.killers[i] = [None, None]

        best_move = None
        best_score = -INF

        for depth in range(1, max_depth + 1):
            if self._time_up():
                break

            alpha = -INF
            beta = INF

            if depth >= 4 and best_score != -INF:
                window = 0.15
                alpha = best_score - window
                beta = best_score + window

            moves = state.get_legal_moves()
            if not moves:
                break

            tt_key = state.zobrist % self.tt_size
            tt_entry = self.tt.get(tt_key)
            tt_move = tt_entry.best_move if tt_entry and tt_entry.zobrist == state.zobrist else None
            moves = self._order_moves(moves, state, tt_move, 0)

            current_best = moves[0]
            current_score = -INF

            for move in moves:
                if self._time_up():
                    break

                state.make_move(move)
                s = -self.negamax(state, depth - 1, -beta, -alpha, 1)
                state.unmake_move()

                if s > current_score:
                    current_score = s
                    current_best = move

                if s > alpha:
                    alpha = s

                if s >= beta:
                    break

            # Re-search with full window if aspiration failed
            if not self._time_up() and depth >= 4 and (current_score <= best_score - 0.15 or current_score >= best_score + 0.15):
                alpha = -INF
                beta = INF
                for move in moves:
                    if self._time_up():
                        break
                    state.make_move(move)
                    s = -self.negamax(state, depth - 1, -beta, -alpha, 1)
                    state.unmake_move()
                    if s > current_score:
                        current_score = s
                        current_best = move
                    if s > alpha:
                        alpha = s

            if not self._time_up():
                best_move = current_best
                best_score = current_score
                elapsed = time.time() - self.start_time
                nps = self.nodes / max(elapsed, 0.001)
                print(f"depth {depth:2d}  score {best_score:+.4f}  "
                      f"nodes {self.nodes:>8d}  nps {nps:>8.0f}  "
                      f"tt_hits {self.tt_hits:>6d}  "
                      f"time {elapsed:.2f}s  pv {best_move}")

        return best_move, best_score

    def get_move(self, state: QuoridorState, max_depth: int = 8,
                 time_limit: float = 10.0) -> Move:
        move, score = self.search(state, max_depth, time_limit)
        return move
