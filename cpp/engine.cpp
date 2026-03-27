/**
 * Quoridor Engine in C++17
 * Full game logic + alpha-beta search + HTTP server on port 5123
 *
 * Dependencies (single-header, downloaded by build.sh):
 *   - httplib.h  (cpp-httplib by yhirose)
 *   - json.hpp   (nlohmann/json)
 */

// Headers: prefer local copy, fall back to system install (brew: cpp-httplib, nlohmann-json)
#if __has_include("httplib.h")
#  include "httplib.h"
#else
#  include <httplib.h>
#endif
#if __has_include("json.hpp")
#  include "json.hpp"
#elif __has_include(<nlohmann/json.hpp>)
#  include <nlohmann/json.hpp>
#else
#  error "nlohmann/json not found. Run: brew install nlohmann-json"
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <climits>
#include <queue>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

// ============================================================
// Constants
// ============================================================

static constexpr int BOARD  = 9;
static constexpr int WGRID  = 8;   // 8x8 wall grid
static constexpr int WALLS_INIT = 10;
static constexpr float INF_SCORE = 999999.0f;

// ============================================================
// Zobrist keys  (seeded identically to the Python engine,
//               python random.seed(42) Mersenne Twister output)
// We pre-generate them at startup deterministically.
// ============================================================

struct Zobrist {
    uint64_t pawn[2][81];       // [player][row*9+col]
    uint64_t hwall[64];         // [row*8+col]
    uint64_t vwall[64];
    uint64_t turn;
    uint64_t walls_left[2][11]; // [player][count 0..10]
};

// Simple LCG-based PRNG that matches python's Mersenne Twister seeded with 42
// We can't exactly replicate MT in a few lines; instead we use a good 64-bit
// LCG with seed derived from the same seed value. The actual values don't need
// to match Python — they just need to be unique and consistent within one run.
// (The TT is cleared between searches anyway.)

static uint64_t lcg_state = 0;
static uint64_t lcg_next() {
    // Splitmix64
    lcg_state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = lcg_state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static Zobrist ZK;

static void init_zobrist() {
    lcg_state = 42ULL * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < 81; i++)
            ZK.pawn[p][i] = lcg_next();
    for (int i = 0; i < 64; i++) ZK.hwall[i] = lcg_next();
    for (int i = 0; i < 64; i++) ZK.vwall[i] = lcg_next();
    ZK.turn = lcg_next();
    for (int p = 0; p < 2; p++)
        for (int i = 0; i <= WALLS_INIT; i++)
            ZK.walls_left[p][i] = lcg_next();
}

// ============================================================
// Move representation
// ============================================================

enum MoveType : int8_t { PAWN = 0, WALL = 1 };
enum Orientation : int8_t { NONE = 0, H = 1, V = 2 };

struct Move {
    int8_t  row;
    int8_t  col;
    int8_t  type;        // MoveType
    int8_t  orient;      // Orientation (for walls)

    bool operator==(const Move& o) const {
        return row == o.row && col == o.col && type == o.type && orient == o.orient;
    }
    bool operator!=(const Move& o) const { return !(*this == o); }

    static Move pawn(int r, int c) {
        return {(int8_t)r, (int8_t)c, PAWN, NONE};
    }
    static Move wall(int r, int c, Orientation ori) {
        return {(int8_t)r, (int8_t)c, WALL, (int8_t)ori};
    }
    static Move null_move() {
        return {-1, -1, PAWN, NONE};
    }
    bool is_null() const { return row == -1; }
};

// ============================================================
// Game State
// ============================================================

struct UndoInfo {
    Move    move;
    int8_t  old_pawn_r, old_pawn_c;
    int8_t  player;
    int8_t  old_winner;
    uint64_t old_zobrist;
};

struct QuoridorState {
    // Pawn positions
    int8_t pawn_r[2], pawn_c[2];
    // Walls remaining
    int8_t walls_left[2];
    // Placed walls: bool arrays indexed [row][col]
    bool h_walls[WGRID][WGRID];
    bool v_walls[WGRID][WGRID];
    // Current player
    int8_t current_player;
    // Winner (-1 = none)
    int8_t winner;
    // Zobrist hash
    uint64_t zobrist;
    // Undo stack
    static constexpr int MAX_HISTORY = 256;
    UndoInfo history[MAX_HISTORY];
    int      history_size;

    void init() {
        pawn_r[0] = 0; pawn_c[0] = 4;
        pawn_r[1] = 8; pawn_c[1] = 4;
        walls_left[0] = WALLS_INIT;
        walls_left[1] = WALLS_INIT;
        memset(h_walls, 0, sizeof(h_walls));
        memset(v_walls, 0, sizeof(v_walls));
        current_player = 0;
        winner = -1;
        history_size = 0;
        zobrist = compute_zobrist();
    }

    uint64_t compute_zobrist() const {
        uint64_t h = 0;
        for (int p = 0; p < 2; p++) {
            h ^= ZK.pawn[p][pawn_r[p] * BOARD + pawn_c[p]];
            h ^= ZK.walls_left[p][walls_left[p]];
        }
        for (int r = 0; r < WGRID; r++)
            for (int c = 0; c < WGRID; c++) {
                if (h_walls[r][c]) h ^= ZK.hwall[r * WGRID + c];
                if (v_walls[r][c]) h ^= ZK.vwall[r * WGRID + c];
            }
        if (current_player == 1) h ^= ZK.turn;
        return h;
    }

    // -----------------------------------------------------------
    // Wall blocking checks — performance critical
    // -----------------------------------------------------------

    // Is move from (r1,c1) to (r2,c2) blocked by a wall?
    // dr = r2-r1, dc = c2-c1; exactly one of them is ±1, other is 0
    inline bool is_blocked(int r1, int c1, int r2, int c2) const {
        int dr = r2 - r1, dc = c2 - c1;
        if (dr == -1) {
            // moving up: blocked by h_wall at (r1-1,c1) or (r1-1,c1-1)
            int wr = r1 - 1;
            if (wr >= 0) {
                if (h_walls[wr][c1]) return true;
                if (c1 > 0 && h_walls[wr][c1-1]) return true;
            }
        } else if (dr == 1) {
            // moving down: blocked by h_wall at (r1,c1) or (r1,c1-1)
            if (h_walls[r1][c1]) return true;
            if (c1 > 0 && h_walls[r1][c1-1]) return true;
        } else if (dc == -1) {
            // moving left: blocked by v_wall at (r1,c1-1) or (r1-1,c1-1)
            int wc = c1 - 1;
            if (wc >= 0) {
                if (v_walls[r1][wc]) return true;
                if (r1 > 0 && v_walls[r1-1][wc]) return true;
            }
        } else { // dc == 1
            // moving right: blocked by v_wall at (r1,c1) or (r1-1,c1)
            if (v_walls[r1][c1]) return true;
            if (r1 > 0 && v_walls[r1-1][c1]) return true;
        }
        return false;
    }

    inline bool can_step(int r1, int c1, int r2, int c2) const {
        if ((unsigned)r2 >= (unsigned)BOARD || (unsigned)c2 >= (unsigned)BOARD)
            return false;
        return !is_blocked(r1, c1, r2, c2);
    }

    // -----------------------------------------------------------
    // Pawn moves
    // -----------------------------------------------------------

    // Returns count of legal pawn moves written into out[]
    int get_pawn_moves(Move out[], int max_out = 16) const {
        int n = 0;
        int p   = current_player;
        int opp = 1 - p;
        int r = pawn_r[p],  c  = pawn_c[p];
        int or_ = pawn_r[opp], oc = pawn_c[opp];

        static const int DR[4] = {-1, 1, 0, 0};
        static const int DC[4] = { 0, 0,-1, 1};

        for (int d = 0; d < 4 && n < max_out; d++) {
            int nr = r + DR[d], nc = c + DC[d];
            if (!can_step(r, c, nr, nc)) continue;

            if (nr == or_ && nc == oc) {
                // Adjacent to opponent — try to jump
                int jr = nr + DR[d], jc = nc + DC[d];
                if (can_step(nr, nc, jr, jc)) {
                    out[n++] = Move::pawn(jr, jc);
                } else {
                    // Side jumps
                    for (int d2 = 0; d2 < 4 && n < max_out; d2++) {
                        if (DR[d2] == -DR[d] && DC[d2] == -DC[d]) continue; // backward
                        int sr = nr + DR[d2], sc = nc + DC[d2];
                        if (can_step(nr, nc, sr, sc) && !(sr == r && sc == c))
                            out[n++] = Move::pawn(sr, sc);
                    }
                }
            } else {
                out[n++] = Move::pawn(nr, nc);
            }
        }
        return n;
    }

    // -----------------------------------------------------------
    // Wall overlap check
    // -----------------------------------------------------------

    bool walls_overlap(int r, int c, Orientation ori) const {
        if (ori == H) {
            if (h_walls[r][c]) return true;
            if (c > 0 && h_walls[r][c-1]) return true;
            if (c < WGRID-1 && h_walls[r][c+1]) return true;
            if (v_walls[r][c]) return true;   // crossing
        } else {
            if (v_walls[r][c]) return true;
            if (r > 0 && v_walls[r-1][c]) return true;
            if (r < WGRID-1 && v_walls[r+1][c]) return true;
            if (h_walls[r][c]) return true;   // crossing
        }
        return false;
    }

    // -----------------------------------------------------------
    // BFS — optimised with a flat array queue
    // -----------------------------------------------------------

    // Returns shortest path distance from player's pawn to goal row,
    // or 999 if no path. Uses flat visited bitset for speed.
    int shortest_path(int player) const {
        int sr = pawn_r[player], sc = pawn_c[player];
        int goal = (player == 0) ? 8 : 0;
        if (sr == goal) return 0;

        // visited[r*9+c]
        bool visited[81] = {};
        // BFS queue: encode as r*9+c with dist packed separately
        // Use two arrays (cell, dist) to avoid struct overhead
        int q_cell[81];
        int q_dist[81];
        int head = 0, tail = 0;

        int start = sr * BOARD + sc;
        visited[start] = true;
        q_cell[tail] = start;
        q_dist[tail] = 0;
        tail++;

        static const int DR[4] = {-1, 1, 0, 0};
        static const int DC[4] = { 0, 0,-1, 1};

        while (head < tail) {
            int cell = q_cell[head];
            int dist = q_dist[head];
            head++;
            int r = cell / BOARD, c = cell % BOARD;

            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if ((unsigned)nr >= (unsigned)BOARD || (unsigned)nc >= (unsigned)BOARD)
                    continue;
                if (is_blocked(r, c, nr, nc)) continue;
                int ncell = nr * BOARD + nc;
                if (visited[ncell]) continue;
                if (nr == goal) return dist + 1;
                visited[ncell] = true;
                q_cell[tail] = ncell;
                q_dist[tail] = dist + 1;
                tail++;
            }
        }
        return 999;
    }

    // BFS path-exists check (no distance needed — faster)
    bool bfs_path_exists(int player) const {
        int sr = pawn_r[player], sc = pawn_c[player];
        int goal = (player == 0) ? 8 : 0;
        if (sr == goal) return true;

        bool visited[81] = {};
        int queue[81];
        int head = 0, tail = 0;

        int start = sr * BOARD + sc;
        visited[start] = true;
        queue[tail++] = start;

        static const int DR[4] = {-1, 1, 0, 0};
        static const int DC[4] = { 0, 0,-1, 1};

        while (head < tail) {
            int cell = queue[head++];
            int r = cell / BOARD, c = cell % BOARD;
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if ((unsigned)nr >= (unsigned)BOARD || (unsigned)nc >= (unsigned)BOARD)
                    continue;
                if (is_blocked(r, c, nr, nc)) continue;
                int ncell = nr * BOARD + nc;
                if (visited[ncell]) continue;
                if (nr == goal) return true;
                visited[ncell] = true;
                queue[tail++] = ncell;
            }
        }
        return false;
    }

    // -----------------------------------------------------------
    // Wall moves
    // -----------------------------------------------------------

    // out[] must have room for up to 128 wall moves
    int get_wall_moves(Move out[]) const {
        int p = current_player;
        if (walls_left[p] == 0) return 0;

        int n = 0;
        for (int r = 0; r < WGRID; r++) {
            for (int c = 0; c < WGRID; c++) {
                for (int oi = 0; oi < 2; oi++) {
                    Orientation ori = (oi == 0) ? H : V;
                    if (walls_overlap(r, c, ori)) continue;

                    // Temporarily place wall
                    if (ori == H) const_cast<QuoridorState*>(this)->h_walls[r][c] = true;
                    else          const_cast<QuoridorState*>(this)->v_walls[r][c] = true;

                    bool ok = bfs_path_exists(0) && bfs_path_exists(1);

                    if (ori == H) const_cast<QuoridorState*>(this)->h_walls[r][c] = false;
                    else          const_cast<QuoridorState*>(this)->v_walls[r][c] = false;

                    if (ok) out[n++] = Move::wall(r, c, ori);
                }
            }
        }
        return n;
    }

    // All legal moves
    int get_legal_moves(Move out[]) const {
        if (winner >= 0) return 0;
        int n = get_pawn_moves(out, 16);
        n += get_wall_moves(out + n);
        return n;
    }

    // -----------------------------------------------------------
    // Make / Unmake
    // -----------------------------------------------------------

    void make_move(const Move& m) {
        int p = current_player;
        UndoInfo& u = history[history_size++];
        u.move = m;
        u.player = (int8_t)p;
        u.old_pawn_r = pawn_r[p];
        u.old_pawn_c = pawn_c[p];
        u.old_zobrist = zobrist;
        u.old_winner = winner;

        if (m.type == PAWN) {
            int old_idx = pawn_r[p] * BOARD + pawn_c[p];
            int new_idx = m.row * BOARD + m.col;
            zobrist ^= ZK.pawn[p][old_idx];
            zobrist ^= ZK.pawn[p][new_idx];
            pawn_r[p] = m.row;
            pawn_c[p] = m.col;
            if (m.row == (p == 0 ? 8 : 0)) winner = (int8_t)p;
        } else {
            if (m.orient == H) {
                h_walls[m.row][m.col] = true;
                zobrist ^= ZK.hwall[m.row * WGRID + m.col];
            } else {
                v_walls[m.row][m.col] = true;
                zobrist ^= ZK.vwall[m.row * WGRID + m.col];
            }
            zobrist ^= ZK.walls_left[p][walls_left[p]];
            walls_left[p]--;
            zobrist ^= ZK.walls_left[p][walls_left[p]];
        }

        zobrist ^= ZK.turn;
        current_player = (int8_t)(1 - p);
    }

    void unmake_move() {
        UndoInfo& u = history[--history_size];
        const Move& m = u.move;
        int p = u.player;

        current_player = (int8_t)p;
        zobrist = u.old_zobrist;
        winner  = u.old_winner;

        if (m.type == PAWN) {
            pawn_r[p] = u.old_pawn_r;
            pawn_c[p] = u.old_pawn_c;
        } else {
            if (m.orient == H) h_walls[m.row][m.col] = false;
            else                v_walls[m.row][m.col] = false;
            walls_left[p]++;
        }
    }

    bool is_terminal() const { return winner >= 0; }
};

// ============================================================
// Evaluation
// ============================================================

static float handcrafted_eval(const QuoridorState& state) {
    int p   = state.current_player;
    int opp = 1 - p;

    int my_dist  = state.shortest_path(p);
    int opp_dist = state.shortest_path(opp);

    // Winning / losing
    if (my_dist == 0)  return  900.0f;
    if (opp_dist == 0) return -900.0f;
    if (my_dist == 1)  return  500.0f;
    if (opp_dist == 1 && state.walls_left[p] == 0) return -500.0f;

    float score = 0.0f;

    // Path advantage
    int path_diff = opp_dist - my_dist;
    if (my_dist <= 4 || opp_dist <= 4)
        score += path_diff * 150.0f;
    else
        score += path_diff * 100.0f;

    // Tempo
    if (path_diff >= 0) score += 30.0f;
    if (my_dist <= 3 && path_diff >= 0) score += 50.0f;

    // Wall economy
    int my_walls  = state.walls_left[p];
    int opp_walls = state.walls_left[opp];

    if (opp_dist <= 4) score += my_walls  * 20.0f;
    else               score += my_walls  *  8.0f;

    if (my_dist <= 4)  score -= opp_walls * 20.0f;
    else               score -= opp_walls *  8.0f;

    if (my_walls > 0 && opp_walls == 0) score += 40.0f;
    if (opp_walls > 0 && my_walls == 0) score -= 40.0f;

    // Positional — progress toward goal
    // Player 0: goal is row 8; Player 1: goal is row 0
    // Python: my_progress = abs(my_row - (8 - my_goal))
    // For p=0: goal=8, 8-goal=0, progress = abs(row - 0) = row
    // For p=1: goal=0, 8-goal=8, progress = abs(row - 8) = 8-row
    int my_goal  = (p   == 0) ? 8 : 0;
    int opp_goal = (opp == 0) ? 8 : 0;
    int my_progress  = std::abs(state.pawn_r[p]   - (8 - my_goal));
    int opp_progress = std::abs(state.pawn_r[opp] - (8 - opp_goal));
    score += (my_progress - opp_progress) * 5.0f;

    // Center column control
    int my_col  = state.pawn_c[p];
    int opp_col = state.pawn_c[opp];
    float center_me  = (float)std::max(0, 3 - std::abs(my_col  - 4)) * 3.0f;
    float center_opp = (float)std::max(0, 3 - std::abs(opp_col - 4)) * 3.0f;
    score += center_me - center_opp;

    return score / 500.0f;
}

// ============================================================
// Transposition Table
// ============================================================

static constexpr int TT_EXACT = 0;
static constexpr int TT_ALPHA = 1;  // upper bound
static constexpr int TT_BETA  = 2;  // lower bound

struct TTEntry {
    uint64_t zobrist;
    float    score;
    Move     best_move;
    int16_t  depth;
    int8_t   flag;
};

static constexpr size_t TT_SIZE = 1 << 22; // ~4M entries
static TTEntry TT[TT_SIZE];

static void tt_clear() {
    memset(TT, 0, sizeof(TT));
    // Mark all entries as invalid by zeroing zobrist
}

static TTEntry* tt_probe(uint64_t zobrist) {
    size_t idx = zobrist & (TT_SIZE - 1);
    TTEntry* e = &TT[idx];
    return (e->zobrist == zobrist) ? e : nullptr;
}

static void tt_store(uint64_t zobrist, int depth, float score, int flag, const Move& best) {
    size_t idx = zobrist & (TT_SIZE - 1);
    TTEntry* e = &TT[idx];
    // Always-replace policy (simple and effective)
    e->zobrist   = zobrist;
    e->depth     = (int16_t)depth;
    e->score     = score;
    e->flag      = (int8_t)flag;
    e->best_move = best;
}

// ============================================================
// Move Ordering
// ============================================================

static constexpr int MAX_DEPTH = 64;
static constexpr int MAX_MOVES = 160; // max pawn(16) + wall(128) + safety

struct SearchContext {
    // Killer moves: [ply][0..1]
    Move killers[MAX_DEPTH][2];
    // History heuristic: indexed by (type<<17 | orient<<16 | row<<8 | col)
    // type: 0=pawn, 1=wall; orient: 0=none,1=H,2=V
    int  history[4][WGRID][WGRID]; // [type*2+orient_idx][r][c]
    long long nodes;
    long long tt_hits;
    TimePoint start_time;
    double    time_limit;

    void reset() {
        memset(killers, 0xFF, sizeof(killers)); // -1,-1 = null
        memset(history, 0, sizeof(history));
        nodes = 0;
        tt_hits = 0;
    }

    bool time_up() const {
        if (time_limit <= 0) return false;
        auto now = Clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        return elapsed >= time_limit;
    }
};

// Score a single move for ordering
static int score_move(const Move& m, const QuoridorState& state,
                      const Move& tt_move, int ply,
                      const SearchContext& ctx) {
    if (m == tt_move) return 10000000;

    int p = state.current_player;
    if (m.type == PAWN) {
        int goal = (p == 0) ? 8 : 0;
        int old_dist = std::abs(state.pawn_r[p] - goal);
        int new_dist = std::abs((int)m.row - goal);
        int s = 1000000 + (old_dist - new_dist) * 100000;
        if (m.row == goal) s += 5000000;
        return s;
    }

    // Wall
    if (!(ctx.killers[ply][0].is_null()) && m == ctx.killers[ply][0]) return 500000;
    if (!(ctx.killers[ply][1].is_null()) && m == ctx.killers[ply][1]) return 400000;

    // History
    int orient_idx = (m.orient == H) ? 1 : 2;
    int hist = ctx.history[orient_idx][m.row][m.col];

    // Proximity to opponent
    int opp = 1 - p;
    int dist_to_opp = std::abs((int)m.row - state.pawn_r[opp]) +
                      std::abs((int)m.col - state.pawn_c[opp]);
    hist += std::max(0, 1000 - dist_to_opp * 100);

    return hist;
}

// Sort moves in descending score order
static int order_moves(Move moves[], int n, const QuoridorState& state,
                       const Move& tt_move, int ply, const SearchContext& ctx) {
    int scores[MAX_MOVES];
    for (int i = 0; i < n; i++)
        scores[i] = score_move(moves[i], state, tt_move, ply, ctx);

    // Insertion sort (fast for small n, avoids alloc)
    for (int i = 1; i < n; i++) {
        Move m = moves[i];
        int  s = scores[i];
        int j = i - 1;
        while (j >= 0 && scores[j] < s) {
            moves[j+1]  = moves[j];
            scores[j+1] = scores[j];
            j--;
        }
        moves[j+1]  = m;
        scores[j+1] = s;
    }
    return n;
}

static void store_killer(SearchContext& ctx, const Move& m, int ply) {
    if (m.type != WALL) return;
    if (!(ctx.killers[ply][0] == m)) {
        ctx.killers[ply][1] = ctx.killers[ply][0];
        ctx.killers[ply][0] = m;
    }
}

static void store_history(SearchContext& ctx, const Move& m, int depth) {
    if (m.type != WALL) return;
    int orient_idx = (m.orient == H) ? 1 : 2;
    ctx.history[orient_idx][m.row][m.col] += depth * depth;
}

// ============================================================
// Negamax Alpha-Beta
// ============================================================

static float negamax(QuoridorState& state, int depth, float alpha, float beta,
                     int ply, SearchContext& ctx, bool is_root = false) {
    ctx.nodes++;

    // Time check every 4096 nodes
    if ((ctx.nodes & 4095) == 0 && ctx.time_up())
        return 0.0f;

    if (state.is_terminal()) {
        // The mover (current_player) just had the move made, winner is 1-current_player
        // Actually: terminal means the LAST player to move won
        // state.winner is set to the player who moved into the goal
        // current_player is now the OTHER player
        // So from negamax perspective: we evaluate for current_player
        // If winner == current_player → we won (shouldn't happen — they're not moving)
        // If winner == opponent → we lost
        return (state.winner == state.current_player) ? INF_SCORE : -INF_SCORE;
    }

    if (depth <= 0)
        return handcrafted_eval(state);

    // TT lookup
    TTEntry* tte = tt_probe(state.zobrist);
    Move tt_move = Move::null_move();
    if (tte) {
        ctx.tt_hits++;
        tt_move = tte->best_move;
        if (tte->depth >= depth && !is_root) {
            if (tte->flag == TT_EXACT) return tte->score;
            if (tte->flag == TT_ALPHA && tte->score <= alpha) return alpha;
            if (tte->flag == TT_BETA  && tte->score >= beta)  return beta;
        }
    }

    // Generate moves (local array — must not be static/thread_local since negamax is recursive)
    Move moves[MAX_MOVES];
    int n = state.get_legal_moves(moves);
    if (n == 0)
        return handcrafted_eval(state);

    // Order moves
    order_moves(moves, n, state, tt_move, ply, ctx);

    float best_score = -INF_SCORE;
    Move  best_move  = moves[0];
    int   flag       = TT_ALPHA;
    int   searched   = 0;

    for (int i = 0; i < n; i++) {
        if (ctx.time_up()) break;

        const Move& m = moves[i];
        state.make_move(m);

        // LMR: reduce late wall moves
        int reduction = 0;
        if (searched >= 4 && depth >= 3 && m.type == WALL && !state.is_terminal())
            reduction = 1;

        float score = -negamax(state, depth - 1 - reduction, -beta, -alpha, ply + 1, ctx);

        // Re-search if LMR caused a fail-high
        if (reduction > 0 && score > alpha)
            score = -negamax(state, depth - 1, -beta, -alpha, ply + 1, ctx);

        state.unmake_move();
        searched++;

        if (score > best_score) {
            best_score = score;
            best_move  = m;
        }

        if (score > alpha) {
            alpha = score;
            flag  = TT_EXACT;

            if (score >= beta) {
                flag = TT_BETA;
                store_killer(ctx, m, ply);
                store_history(ctx, m, depth);
                break;
            }
        }
    }

    tt_store(state.zobrist, depth, best_score, flag, best_move);
    return best_score;
}

// ============================================================
// Iterative Deepening Search with Aspiration Windows
// ============================================================

struct SearchResult {
    Move  best_move;
    float best_score;
    long long nodes;
    double elapsed;
};

static SearchResult search(QuoridorState& state, int max_depth = 15,
                            double time_limit = 30.0) {
    SearchContext ctx;
    ctx.reset();
    ctx.start_time  = Clock::now();
    ctx.time_limit  = time_limit;

    Move  best_move  = Move::null_move();
    float best_score = -INF_SCORE;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (ctx.time_up()) break;

        float alpha = -INF_SCORE, beta = INF_SCORE;

        // Aspiration windows from depth 4
        if (depth >= 4 && best_score != -INF_SCORE) {
            float window = 0.15f;
            alpha = best_score - window;
            beta  = best_score + window;
        }

        // Generate and order moves at root (must NOT be static — search() is called per-request)
        Move root_moves[MAX_MOVES];
        int n = state.get_legal_moves(root_moves);
        if (n == 0) break;

        TTEntry* tte = tt_probe(state.zobrist);
        Move tt_move = (tte && tte->zobrist == state.zobrist)
                       ? tte->best_move : Move::null_move();
        order_moves(root_moves, n, state, tt_move, 0, ctx);

        Move  cur_best  = root_moves[0];
        float cur_score = -INF_SCORE;

        for (int i = 0; i < n; i++) {
            if (ctx.time_up()) break;
            state.make_move(root_moves[i]);
            float s = -negamax(state, depth - 1, -beta, -alpha, 1, ctx, false);
            state.unmake_move();

            if (s > cur_score) {
                cur_score = s;
                cur_best  = root_moves[i];
            }
            if (s > alpha) alpha = s;
            if (s >= beta) break;
        }

        // Aspiration re-search if window failed
        if (!ctx.time_up() && depth >= 4 &&
            (cur_score <= best_score - 0.15f || cur_score >= best_score + 0.15f)) {
            alpha = -INF_SCORE; beta = INF_SCORE;
            cur_score = -INF_SCORE;
            for (int i = 0; i < n; i++) {
                if (ctx.time_up()) break;
                state.make_move(root_moves[i]);
                float s = -negamax(state, depth - 1, -beta, -alpha, 1, ctx, false);
                state.unmake_move();
                if (s > cur_score) { cur_score = s; cur_best = root_moves[i]; }
                if (s > alpha) alpha = s;
            }
        }

        if (!ctx.time_up()) {
            best_move  = cur_best;
            best_score = cur_score;

            double elapsed = std::chrono::duration<double>(
                Clock::now() - ctx.start_time).count();
            double nps = (elapsed > 0.0001) ? ctx.nodes / elapsed : 0;

            std::cerr << "depth " << depth
                      << "  score " << best_score
                      << "  nodes " << ctx.nodes
                      << "  nps "   << (long long)nps
                      << "  tt_hits " << ctx.tt_hits
                      << "  time "  << elapsed << "s"
                      << "  pv ("   << (int)best_move.row << "," << (int)best_move.col << ")"
                      << std::endl;
        }
    }

    double elapsed = std::chrono::duration<double>(
        Clock::now() - ctx.start_time).count();
    return {best_move, best_score, ctx.nodes, elapsed};
}

// ============================================================
// State from JSON (matches Python server's state_from_json)
// ============================================================

static QuoridorState state_from_json(const json& data) {
    QuoridorState s;
    s.init();

    auto pawns = data["pawns"];
    s.pawn_r[0] = (int8_t)(int)pawns[0][0];
    s.pawn_c[0] = (int8_t)(int)pawns[0][1];
    s.pawn_r[1] = (int8_t)(int)pawns[1][0];
    s.pawn_c[1] = (int8_t)(int)pawns[1][1];

    if (data.contains("wallsLeft")) {
        s.walls_left[0] = (int8_t)(int)data["wallsLeft"][0];
        s.walls_left[1] = (int8_t)(int)data["wallsLeft"][1];
    }

    s.current_player = (int8_t)(int)data.value("currentPlayer", 0);

    memset(s.h_walls, 0, sizeof(s.h_walls));
    memset(s.v_walls, 0, sizeof(s.v_walls));

    if (data.contains("walls")) {
        for (auto& w : data["walls"]) {
            int r = (int)w[0], c = (int)w[1];
            std::string ori = (std::string)w[2];
            if (ori == "h") s.h_walls[r][c] = true;
            else            s.v_walls[r][c] = true;
        }
    }

    s.winner = -1;
    s.history_size = 0;
    s.zobrist = s.compute_zobrist();
    return s;
}

// ============================================================
// Logging
// ============================================================

static std::string LOG_PATH;
static std::string BASE_DIR;
static std::mutex  log_mutex;

static std::string now_iso() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

static void log_entry(const json& entry) {
    json e = entry;
    e["timestamp"] = now_iso();
    std::lock_guard<std::mutex> lk(log_mutex);
    std::ofstream f(LOG_PATH, std::ios::app);
    f << e.dump() << "\n";
}

// ============================================================
// HTTP Server
// ============================================================

int main(int argc, char* argv[]) {
    // Determine base directory (parent of where this binary lives, or cwd)
    BASE_DIR = "/Users/abhijay/barricade-engine";
    if (argc > 1) BASE_DIR = argv[1];
    LOG_PATH = BASE_DIR + "/game_log.jsonl";

    init_zobrist();
    tt_clear();

    std::cout << "Quoridor C++ engine starting on port 5123" << std::endl;
    std::cout << "Log: " << LOG_PATH << std::endl;
    std::cout << "TT size: " << TT_SIZE << " entries ("
              << (sizeof(TT) / 1024 / 1024) << " MB)" << std::endl;

    httplib::Server svr;

    // ── GET /api/health ──────────────────────────────────────
    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"status",        "ok"},
            {"nnue_loaded",   false},
            {"eval_mode",     "handcrafted"},
            {"default_depth", 15},
            {"default_time",  30.0},
            {"engine",        "c++"}
        };
        res.set_content(body.dump(), "application/json");
    });

    // ── POST /api/best-move ──────────────────────────────────
    svr.Post("/api/best-move", [](const httplib::Request& req, httplib::Response& res) {
        json data;
        try {
            data = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"bad json\"}", "application/json");
            return;
        }

        QuoridorState state = state_from_json(data);

        int    depth      = data.value("depth",     15);
        double time_limit = data.value("timeLimit", 30.0);
        if (depth < 1)  depth = 15;
        if (depth > 64) depth = 64;

        SearchResult sr = search(state, depth, time_limit);

        if (sr.best_move.is_null()) {
            // Fallback: first legal move
            Move moves[MAX_MOVES];
            int n = state.get_legal_moves(moves);
            if (n > 0) sr.best_move = moves[0];
        }

        // Compute current-state eval (from perspective of current_player)
        float eval_score = handcrafted_eval(state);

        json move_json;
        if (sr.best_move.type == PAWN) {
            move_json = {
                {"type", "pawn"},
                {"row",  (int)sr.best_move.row},
                {"col",  (int)sr.best_move.col},
                {"orientation", nullptr}
            };
        } else {
            move_json = {
                {"type", "wall"},
                {"row",  (int)sr.best_move.row},
                {"col",  (int)sr.best_move.col},
                {"orientation", (sr.best_move.orient == H) ? "h" : "v"}
            };
        }

        json result = {
            {"move",       move_json},
            {"score",      std::round(sr.best_score * 10000.0f) / 10000.0f},
            {"nodes",      sr.nodes},
            {"evaluation", std::round(eval_score * 10000.0f) / 10000.0f},
            {"searchTime", std::round(sr.elapsed * 100.0) / 100.0}
        };

        log_entry({
            {"type",   "analysis"},
            {"input",  data},
            {"result", result},
            {"shortestPaths", {
                state.shortest_path(0),
                state.shortest_path(1)
            }}
        });

        res.set_content(result.dump(), "application/json");
    });

    // ── POST /api/log ────────────────────────────────────────
    svr.Post("/api/log", [](const httplib::Request& req, httplib::Response& res) {
        json data;
        try { data = json::parse(req.body); } catch (...) { data = {{"raw", req.body}}; }
        log_entry({{"type", "extension"}, {"data", data}});
        std::cerr << "[EXT LOG] " << req.body.substr(0, 200) << std::endl;
        res.set_content("{\"status\":\"logged\"}", "application/json");
    });

    // ── POST /api/debug-dom ──────────────────────────────────
    svr.Post("/api/debug-dom", [](const httplib::Request& req, httplib::Response& res) {
        json data;
        try { data = json::parse(req.body); } catch (...) { data = {{"raw", req.body}}; }
        std::string path = BASE_DIR + "/dom_debug.json";
        {
            std::ofstream f(path);
            f << data.dump(2) << "\n";
        }
        std::cerr << "DOM debug saved to " << path << std::endl;
        res.set_content("{\"status\":\"saved\"}", "application/json");
    });

    // CORS preflight for all routes
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });

    // Add CORS headers to every response
    svr.set_post_routing_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.listen("127.0.0.1", 5123);
    return 0;
}
