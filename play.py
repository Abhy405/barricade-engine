"""
CLI to play Quoridor against the engine.
"""

import os
import sys
from game import QuoridorState, Move, BOARD_SIZE, WALL_GRID
from nnue import NNUE, load_model
from search import SearchEngine


def parse_move(input_str: str, state: QuoridorState) -> Move:
    """
    Parse user input into a Move.

    Formats:
        Pawn move: "row col" or direction ("u", "d", "l", "r", "ul", "ur", "dl", "dr")
        Wall: "w row col h" or "w row col v"
    """
    parts = input_str.strip().lower().split()

    if not parts:
        raise ValueError("Empty input")

    # Direction shortcuts
    dir_map = {
        'u': (-1, 0), 'up': (-1, 0),
        'd': (1, 0), 'down': (1, 0),
        'l': (0, -1), 'left': (0, -1),
        'r': (0, 1), 'right': (0, 1),
    }

    if parts[0] in dir_map:
        dr, dc = dir_map[parts[0]]
        p = state.current_player
        r, c = state.pawns[p]
        return Move(Move.PAWN, r + dr, c + dc)

    if parts[0] == 'w' or parts[0] == 'wall':
        if len(parts) < 4:
            raise ValueError("Wall format: w <row> <col> <h/v>")
        r, c = int(parts[1]), int(parts[2])
        ori = parts[3]
        if ori not in ('h', 'v'):
            raise ValueError("Orientation must be 'h' or 'v'")
        if not (0 <= r < WALL_GRID and 0 <= c < WALL_GRID):
            raise ValueError(f"Wall position must be 0-{WALL_GRID-1}")
        return Move(Move.WALL, r, c, ori)

    # Direct coordinates
    if len(parts) == 2:
        r, c = int(parts[0]), int(parts[1])
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return Move(Move.PAWN, r, c)

    raise ValueError("Unrecognized move format. Use: u/d/l/r, 'row col', or 'w row col h/v'")


def main():
    # Load model if available
    model_path = os.path.join("models", "nnue_best.pt")
    model = NNUE()

    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Loaded NNUE from {model_path}")
    else:
        print("No trained model found — using handcrafted eval only")

    engine = SearchEngine(model)

    # Check for untrained model
    if not os.path.exists(model_path):
        engine.nnue_weight = 0.0
        print("(Set NNUE weight to 0, using pure handcrafted eval)\n")

    state = QuoridorState()

    # Choose side
    print("Quoridor Engine")
    print("=" * 40)
    print("Player 1 (1): starts top, goes to bottom")
    print("Player 2 (2): starts bottom, goes to top")
    side = input("\nPlay as (1/2): ").strip()
    human_player = 0 if side == '1' else 1

    # Choose engine strength
    print("\nEngine strength:")
    print("  1 = Easy   (depth 2, 1s)")
    print("  2 = Medium (depth 4, 3s)")
    print("  3 = Hard   (depth 6, 5s)")
    print("  4 = Max    (depth 8, 10s)")
    level = input("Level (1-4): ").strip()
    depths = {'1': 2, '2': 4, '3': 6, '4': 8}
    times = {'1': 1.0, '2': 3.0, '3': 5.0, '4': 10.0}
    search_depth = depths.get(level, 4)
    time_limit = times.get(level, 3.0)

    print(f"\nEngine: depth={search_depth}, time={time_limit}s")
    print("\nMove commands:")
    print("  Pawn: u/d/l/r (up/down/left/right)")
    print("  Pawn: <row> <col> (jump to specific square)")
    print("  Wall: w <row> <col> <h/v>")
    print("  Quit: q")
    print()

    move_num = 0
    while not state.is_terminal():
        state.display()
        print()

        if state.current_player == human_player:
            # Human turn
            while True:
                try:
                    inp = input(f"Your move (P{human_player + 1}): ").strip()
                    if inp.lower() in ('q', 'quit', 'exit'):
                        print("Goodbye!")
                        return

                    if inp.lower() == 'moves':
                        legal = state.get_legal_moves()
                        pawn_m = [m for m in legal if m.move_type == Move.PAWN]
                        wall_m = [m for m in legal if m.move_type == Move.WALL]
                        print(f"  Pawn moves: {pawn_m}")
                        print(f"  Wall moves: {len(wall_m)} available")
                        continue

                    if inp.lower() == 'eval':
                        score = engine.evaluate(state)
                        print(f"  Eval: {score:+.4f} (from current player's view)")
                        continue

                    move = parse_move(inp, state)

                    # Validate
                    legal = state.get_legal_moves()
                    if move not in legal:
                        print(f"  Illegal move: {move}")
                        print(f"  Legal pawn moves: {[m for m in legal if m.move_type == Move.PAWN]}")
                        continue

                    state.make_move(move)
                    move_num += 1
                    break

                except ValueError as e:
                    print(f"  Error: {e}")
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    return
        else:
            # Engine turn
            print(f"Engine thinking (depth={search_depth}, time={time_limit}s)...")
            move = engine.get_move(state, max_depth=search_depth, time_limit=time_limit)
            print(f"Engine plays: {move}")
            state.make_move(move)
            move_num += 1

        print()

    # Game over
    state.display()
    if state.winner == human_player:
        print("\n*** YOU WIN! ***")
    else:
        print("\n*** ENGINE WINS ***")
    print(f"Game lasted {move_num} moves")


if __name__ == "__main__":
    main()
