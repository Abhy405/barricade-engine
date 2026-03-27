"""
Self-Play Training Pipeline for the Quoridor NNUE.

Training loop (inspired by Stockfish NNUE training):
1. Self-play: engine plays against itself at a given search depth
2. Collect positions + game outcomes
3. Train NNUE to predict game outcome from position features
4. Repeat with stronger engine

The target for each position is the game outcome from that position's
current player's perspective: +1 (win), -1 (loss), 0 (draw, rare in Quoridor).
"""

import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from game import QuoridorState, Move, INPUT_SIZE
from nnue import NNUE, save_model, load_model
from search import SearchEngine


class PositionDataset(Dataset):
    """Dataset of (features, outcome) pairs from self-play games."""

    def __init__(self, positions: List[Tuple[list, float]]):
        self.features = torch.tensor([p[0] for p in positions], dtype=torch.float32)
        self.targets = torch.tensor([p[1] for p in positions], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def play_self_game(engine: SearchEngine, search_depth: int = 3,
                   time_limit: float = 2.0, noise: float = 0.1) -> List[Tuple[list, float]]:
    """
    Play one self-play game and return (features, outcome) for each position.

    Args:
        engine: The search engine
        search_depth: Search depth per move
        time_limit: Time limit per move in seconds
        noise: Probability of making a random move (for exploration)

    Returns:
        List of (encoded_features, outcome_from_current_player_perspective)
    """
    state = QuoridorState()
    positions = []  # (features, current_player)
    max_moves = 200  # safety limit

    for move_num in range(max_moves):
        if state.is_terminal():
            break

        # Record position
        features = state.encode_board()
        positions.append((features, state.current_player))

        # Choose move: random with probability `noise`, otherwise engine
        if random.random() < noise and move_num < 20:
            # Random move for exploration (mostly in opening)
            moves = state.get_legal_moves()
            # Bias toward pawn moves in random selection
            pawn_moves = [m for m in moves if m.move_type == Move.PAWN]
            if pawn_moves and random.random() < 0.7:
                move = random.choice(pawn_moves)
            else:
                move = random.choice(moves)
        else:
            move = engine.get_move(state, max_depth=search_depth, time_limit=time_limit)

        if move is None:
            break

        state.make_move(move)

    # Determine outcome
    if state.winner >= 0:
        winner = state.winner
    else:
        # Game didn't finish — evaluate final position
        winner = -1  # draw-ish

    # Label each position with outcome from that position's current player's perspective
    labeled = []
    for features, player in positions:
        if winner == -1:
            outcome = 0.0
        elif winner == player:
            outcome = 1.0
        else:
            outcome = -1.0
        labeled.append((features, outcome))

    return labeled


def generate_training_data(engine: SearchEngine, num_games: int = 100,
                           search_depth: int = 3, time_limit: float = 1.0,
                           noise: float = 0.15) -> List[Tuple[list, float]]:
    """Generate training data from self-play games."""
    all_positions = []
    wins = [0, 0]
    draws = 0

    for game_num in range(num_games):
        positions = play_self_game(engine, search_depth, time_limit, noise)
        all_positions.extend(positions)

        # Track stats
        if positions:
            outcome = positions[-1][1]
            if outcome > 0:
                wins[0] += 1
            elif outcome < 0:
                wins[1] += 1
            else:
                draws += 1

        if (game_num + 1) % 10 == 0:
            print(f"  Game {game_num + 1}/{num_games}: "
                  f"{len(all_positions)} positions, "
                  f"P1 wins: {wins[0]}, P2 wins: {wins[1]}, draws: {draws}")

    print(f"\nGenerated {len(all_positions)} positions from {num_games} games")
    return all_positions


def train_nnue(model: NNUE, positions: List[Tuple[list, float]],
               epochs: int = 20, batch_size: int = 256,
               lr: float = 0.001) -> float:
    """
    Train the NNUE on collected positions.

    Returns final loss.
    """
    dataset = PositionDataset(positions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    model.train()
    final_loss = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0

        for features, targets in loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(batches, 1)
        final_loss = avg_loss

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")

    model.eval()
    return final_loss


def training_loop(num_iterations: int = 10, games_per_iter: int = 50,
                  search_depth: int = 3, time_limit: float = 1.0,
                  epochs_per_iter: int = 20, save_dir: str = "models"):
    """
    Full training loop:
    1. Start with handcrafted eval
    2. Generate self-play data
    3. Train NNUE
    4. Increase NNUE weight in eval blend
    5. Repeat with stronger engine
    """
    os.makedirs(save_dir, exist_ok=True)

    model = NNUE()
    engine = SearchEngine(model)

    # Start with mostly handcrafted eval, gradually transition to NNUE
    all_positions = []

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")

        # Gradually increase NNUE weight
        engine.nnue_weight = min(1.0, iteration / (num_iterations * 0.7))
        print(f"NNUE weight: {engine.nnue_weight:.2f}")

        # Gradually increase search depth
        depth = min(search_depth + iteration // 4, search_depth + 2)
        noise = max(0.05, 0.2 - iteration * 0.015)
        print(f"Search depth: {depth}, Exploration noise: {noise:.2f}")

        # Generate self-play data
        print(f"\nGenerating {games_per_iter} self-play games...")
        new_positions = generate_training_data(
            engine, num_games=games_per_iter,
            search_depth=depth, time_limit=time_limit, noise=noise
        )
        all_positions.extend(new_positions)

        # Keep a sliding window of positions (prevent memory blowup)
        max_positions = 200000
        if len(all_positions) > max_positions:
            all_positions = all_positions[-max_positions:]

        # Train NNUE
        print(f"\nTraining on {len(all_positions)} positions...")
        lr = 0.001 * (0.9 ** (iteration - 1))
        loss = train_nnue(model, all_positions, epochs=epochs_per_iter,
                          batch_size=256, lr=lr)

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"nnue_iter{iteration:03d}.pt")
        save_model(model, ckpt_path)
        print(f"\nSaved checkpoint: {ckpt_path}")
        print(f"Final loss: {loss:.6f}")

        # Also save as "best" (latest is best in this scheme)
        best_path = os.path.join(save_dir, "nnue_best.pt")
        save_model(model, best_path)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best model saved to: {os.path.join(save_dir, 'nnue_best.pt')}")
    print(f"{'='*60}")

    return model


def quick_train(games: int = 20, depth: int = 2, iterations: int = 3):
    """Quick training run for testing."""
    return training_loop(
        num_iterations=iterations,
        games_per_iter=games,
        search_depth=depth,
        time_limit=0.5,
        epochs_per_iter=10,
        save_dir="models"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Quoridor NNUE")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--games", type=int, default=50, help="Games per iteration")
    parser.add_argument("--depth", type=int, default=3, help="Base search depth")
    parser.add_argument("--time", type=float, default=1.0, help="Time limit per move (seconds)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per iteration")
    parser.add_argument("--quick", action="store_true", help="Quick test run")

    args = parser.parse_args()

    if args.quick:
        quick_train()
    else:
        training_loop(
            num_iterations=args.iterations,
            games_per_iter=args.games,
            search_depth=args.depth,
            time_limit=args.time,
            epochs_per_iter=args.epochs,
        )
