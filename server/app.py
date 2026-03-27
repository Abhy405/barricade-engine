"""
Local API server for the Quoridor engine.
Chrome extension calls this to get best moves.
Logs all game states to game_log.jsonl for debugging.
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from game import QuoridorState, Move, BOARD_SIZE, WALL_GRID
from nnue import NNUE, load_model
from search import SearchEngine

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "game_log.jsonl")

# Load engine — pure handcrafted eval until NNUE is properly trained
model = NNUE()
model_path = os.path.join(BASE_DIR, "models", "nnue_best.pt")
nnue_loaded = False
# Uncomment below once NNUE is properly trained:
# if os.path.exists(model_path):
#     model = load_model(model_path)
#     nnue_loaded = True
#     print(f"Loaded NNUE from {model_path}")

engine = SearchEngine(model)
engine.nnue_weight = 0.0  # pure handcrafted — NNUE not ready yet
print("Engine ready — using handcrafted eval (unlimited depth, 30s per move)")


def log_entry(entry: dict):
    """Append a JSON line to the game log."""
    entry['timestamp'] = datetime.now().isoformat()
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def state_from_json(data: dict) -> QuoridorState:
    state = QuoridorState()
    state.pawns[0] = list(data['pawns'][0])
    state.pawns[1] = list(data['pawns'][1])
    state.walls_left[0] = data.get('wallsLeft', [10, 10])[0]
    state.walls_left[1] = data.get('wallsLeft', [10, 10])[1]
    state.current_player = data.get('currentPlayer', 0)
    state.h_walls = set()
    state.v_walls = set()
    for w in data.get('walls', []):
        r, c, ori = w[0], w[1], w[2]
        if ori == 'h':
            state.h_walls.add((r, c))
        else:
            state.v_walls.add((r, c))
    state.zobrist = state._compute_zobrist()
    return state


@app.route('/api/best-move', methods=['POST'])
def best_move():
    data = request.json
    state = state_from_json(data)

    # Max defaults
    depth = data.get('depth', 64)
    time_limit = data.get('timeLimit', 30.0)

    t0 = time.time()
    move, score = engine.search(state, max_depth=depth, time_limit=time_limit)
    elapsed = time.time() - t0

    result = {
        'move': {
            'type': 'pawn' if move.move_type == 0 else 'wall',
            'row': int(move.row),
            'col': int(move.col),
            'orientation': move.orientation if move.move_type == 1 else None
        },
        'score': round(float(score), 4),
        'nodes': engine.nodes,
        'evaluation': round(float(engine.evaluate(state)), 4),
        'searchTime': round(elapsed, 2)
    }

    # Log for debugging
    log_entry({
        'type': 'analysis',
        'input': data,
        'result': result,
        'shortestPaths': [state.shortest_path(0), state.shortest_path(1)],
    })

    return jsonify(result)


@app.route('/api/legal-moves', methods=['POST'])
def legal_moves():
    data = request.json
    state = state_from_json(data)
    moves = state.get_legal_moves()

    result = []
    for m in moves:
        result.append({
            'type': 'pawn' if m.move_type == 0 else 'wall',
            'row': int(m.row),
            'col': int(m.col),
            'orientation': m.orientation if m.move_type == 1 else None
        })

    return jsonify({
        'moves': result,
        'count': len(result),
        'shortestPaths': [state.shortest_path(0), state.shortest_path(1)]
    })


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    state = state_from_json(data)
    score = engine.evaluate(state)
    return jsonify({
        'evaluation': round(float(score), 4),
        'shortestPaths': [state.shortest_path(0), state.shortest_path(1)],
        'wallsLeft': list(state.walls_left)
    })


@app.route('/api/debug-dom', methods=['POST'])
def debug_dom():
    data = request.json
    debug_path = os.path.join(BASE_DIR, "dom_debug.json")
    with open(debug_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"DOM debug saved to {debug_path}")
    return jsonify({'status': 'saved'})


@app.route('/api/log', methods=['POST'])
def log_from_extension():
    """Extension can send arbitrary debug logs."""
    data = request.json
    log_entry({'type': 'extension', 'data': data})
    print(f"[EXT LOG] {json.dumps(data)[:200]}")
    return jsonify({'status': 'logged'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'nnue_loaded': nnue_loaded,
        'eval_mode': 'handcrafted' if engine.nnue_weight == 0 else 'nnue_blend',
        'default_depth': 64,
        'default_time': 30.0
    })


if __name__ == '__main__':
    print(f"Game log: {LOG_PATH}")
    app.run(host='127.0.0.1', port=5123, debug=False)
