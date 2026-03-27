"""
Local API server for the Quoridor engine.
Chrome extension calls this to get best moves.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
from game import QuoridorState, Move, BOARD_SIZE, WALL_GRID
from nnue import NNUE, load_model
from search import SearchEngine

app = Flask(__name__)
CORS(app)

# Load engine
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nnue_best.pt")
model = NNUE()
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"Loaded NNUE from {model_path}")
else:
    print("No trained model found, using handcrafted eval")

engine = SearchEngine(model)
if not os.path.exists(model_path):
    engine.nnue_weight = 0.0


def state_from_json(data: dict) -> QuoridorState:
    """Build a QuoridorState from JSON board representation."""
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
    """
    Get the best move for a given board state.

    Request JSON:
    {
        "pawns": [[r1,c1], [r2,c2]],
        "walls": [[r, c, "h"|"v"], ...],
        "wallsLeft": [p1_walls, p2_walls],
        "currentPlayer": 0|1,
        "depth": 4,        // optional
        "timeLimit": 3.0   // optional, seconds
    }

    Response JSON:
    {
        "move": {"type": "pawn"|"wall", "row": r, "col": c, "orientation": "h"|"v"|null},
        "score": float,
        "depth": int,
        "nodes": int
    }
    """
    data = request.json
    state = state_from_json(data)

    depth = data.get('depth', 4)
    time_limit = data.get('timeLimit', 3.0)

    move, score = engine.search(state, max_depth=depth, time_limit=time_limit)

    return jsonify({
        'move': {
            'type': 'pawn' if move.move_type == 0 else 'wall',
            'row': int(move.row),
            'col': int(move.col),
            'orientation': move.orientation if move.move_type == 1 else None
        },
        'score': round(float(score), 4),
        'nodes': engine.nodes,
        'evaluation': round(float(engine.evaluate(state)), 4)
    })


@app.route('/api/legal-moves', methods=['POST'])
def legal_moves():
    """Get all legal moves for a position."""
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
    """Evaluate a position without searching."""
    data = request.json
    state = state_from_json(data)
    score = engine.evaluate(state)
    return jsonify({
        'evaluation': round(float(score), 4),
        'shortestPaths': [state.shortest_path(0), state.shortest_path(1)],
        'wallsLeft': list(state.walls_left)
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'nnue_loaded': os.path.exists(model_path)})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5123, debug=False)
