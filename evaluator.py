"""
evaluate.py
============

Evaluation framework: pit the trained engine against Stockfish and
report:
  1. Win / draw / loss rate.
  2. Average centipawn loss (ACPL) of the engine's moves, as judged by
     Stockfish.
  3. Move agreement rate with Stockfish's top choice.
  4. A rough Elo estimate derived from the match score against
     Stockfish at a fixed (low) skill level / depth.

Requirements
------------
  - A Stockfish binary available on PATH (or pass `--stockfish_path`).
    Install: `apt-get install stockfish` or download from
    https://stockfishchess.org/download/.
  - `pip install chess` (python-chess, provides the UCI engine wrapper).

Usage
-----
    python evaluate.py \
        --model checkpoints/model_best.keras \
        --stockfish_path /usr/games/stockfish \
        --games 10 \
        --simulations 200 \
        --stockfish_skill 1 \
        --stockfish_depth 5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import chess.engine

from engine import ChessEngine


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    games_played: int = 0
    engine_wins: int = 0
    draws: int = 0
    engine_losses: int = 0

    total_cp_loss: float = 0.0    # sum of centipawn losses on engine moves
    num_cp_samples: int = 0

    matching_moves: int = 0       # engine move == Stockfish's top move
    total_moves: int = 0

    def win_rate(self) -> float:
        return self.engine_wins / self.games_played if self.games_played else 0.0

    def score(self) -> float:
        """Score fraction: wins + 0.5*draws, out of games played."""
        if self.games_played == 0:
            return 0.0
        return (self.engine_wins + 0.5 * self.draws) / self.games_played

    def avg_cp_loss(self) -> float:
        return self.total_cp_loss / self.num_cp_samples if self.num_cp_samples else 0.0

    def move_agreement(self) -> float:
        return self.matching_moves / self.total_moves if self.total_moves else 0.0

    def estimated_elo(self, stockfish_elo: float) -> float:
        """Estimate the engine's Elo from the match score against a
        Stockfish opponent of known (configured) Elo, using the
        standard logistic Elo formula:

            score = 1 / (1 + 10^((opponent_elo - engine_elo) / 400))
            => engine_elo = opponent_elo + 400 * log10(score / (1 - score))

        `score` is clamped away from {0, 1} to avoid +/- infinity.
        """
        s = min(max(self.score(), 1e-3), 1 - 1e-3)
        return stockfish_elo + 400.0 * math.log10(s / (1 - s))

    def summary(self, stockfish_elo: float) -> str:
        lines = [
            f"Games played:        {self.games_played}",
            f"Engine record (W/D/L): {self.engine_wins}/{self.draws}/{self.engine_losses}",
            f"Score:               {self.score():.3f}",
            f"Avg centipawn loss:  {self.avg_cp_loss():.1f}",
            f"Move agreement:      {self.move_agreement():.1%}  "
            f"({self.matching_moves}/{self.total_moves})",
            f"Estimated Elo:       {self.estimated_elo(stockfish_elo):.0f} "
            f"(assuming Stockfish @ {stockfish_elo:.0f} Elo)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stockfish helpers
# ---------------------------------------------------------------------------

def _configure_stockfish(sf: chess.engine.SimpleEngine, skill_level: int):
    """Set Stockfish's Skill Level (0-20) and disable pondering for
    deterministic, weaker play suitable for early-training evaluation.
    """
    try:
        sf.configure({"Skill Level": skill_level, "Ponder": False})
    except chess.engine.EngineError:
        # Some Stockfish builds may not expose "Skill Level"; fall back
        # to UCI_LimitStrength + UCI_Elo if available.
        try:
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320 + skill_level * 40})
        except chess.engine.EngineError:
            pass


def _stockfish_eval_cp(sf: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> Optional[int]:
    """Return Stockfish's centipawn evaluation of `board` from the
    perspective of the side to move, or None if mate is on the board
    (centipawn loss is not meaningful in forced-mate lines).
    """
    info = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].pov(board.turn)
    if score.is_mate():
        return None
    return score.score()


def _stockfish_best_move(sf: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> chess.Move:
    result = sf.play(board, chess.engine.Limit(depth=depth))
    return result.move


# ---------------------------------------------------------------------------
# Single-game evaluation
# ---------------------------------------------------------------------------

def play_evaluation_game(
    az_engine: ChessEngine,
    sf: chess.engine.SimpleEngine,
    az_plays_white: bool,
    stockfish_depth: int,
    max_moves: int,
    result: EvalResult,
):
    board = chess.Board()

    while not board.is_game_over(claim_draw=True) and board.fullmove_number <= max_moves:
        az_to_move = (board.turn == chess.WHITE) == az_plays_white

        if az_to_move:
            # --- Evaluate AZ engine's move quality before pushing ---
            cp_before = _stockfish_eval_cp(sf, board, stockfish_depth)
            sf_best = _stockfish_best_move(sf, board, stockfish_depth)

            az_move = az_engine.get_best_move(board, temperature=0.0)

            result.total_moves += 1
            if az_move == sf_best:
                result.matching_moves += 1

            board.push(az_move)

            if cp_before is not None:
                cp_after = _stockfish_eval_cp(sf, board, stockfish_depth)
                if cp_after is not None:
                    # cp_after is from the perspective of the side now to
                    # move (the opponent); negate to express it from the
                    # AZ engine's perspective, then compare to cp_before.
                    cp_after_from_az_pov = -cp_after
                    loss = max(0.0, cp_before - cp_after_from_az_pov)
                    result.total_cp_loss += loss
                    result.num_cp_samples += 1
        else:
            sf_move = _stockfish_best_move(sf, board, stockfish_depth)
            board.push(sf_move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result.draws += 1
    else:
        az_won = (outcome.winner == chess.WHITE) == az_plays_white
        if az_won:
            result.engine_wins += 1
        else:
            result.engine_losses += 1

    result.games_played += 1


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    stockfish_path: str,
    num_games: int = 10,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    stockfish_skill: int = 1,
    stockfish_depth: int = 5,
    stockfish_elo: float = 1350.0,
    max_moves: int = 200,
) -> EvalResult:
    az_engine = ChessEngine(model_path, num_simulations=num_simulations, c_puct=c_puct)

    sf = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _configure_stockfish(sf, stockfish_skill)

    result = EvalResult()

    try:
        for g in range(num_games):
            az_plays_white = (g % 2 == 0)
            play_evaluation_game(
                az_engine, sf, az_plays_white, stockfish_depth, max_moves, result
            )
            print(
                f"[eval] game {g + 1}/{num_games} done "
                f"(AZ as {'White' if az_plays_white else 'Black'}). "
                f"Running score: {result.score():.3f}"
            )
    finally:
        sf.quit()

    return result, stockfish_elo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the chess engine against Stockfish.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=200, choices=[100, 200, 400])
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--stockfish_skill", type=int, default=1,
                         help="Stockfish 'Skill Level' (0-20). Use a low value (0-3) "
                              "for early-training checkpoints; the engine will likely "
                              "lose heavily against full-strength Stockfish.")
    parser.add_argument("--stockfish_depth", type=int, default=5,
                         help="Search depth Stockfish uses both to play and to "
                              "evaluate the AZ engine's moves.")
    parser.add_argument("--stockfish_elo", type=float, default=1350.0,
                         help="Approximate Elo of the configured Stockfish skill "
                              "level, used as the anchor for Elo estimation.")
    parser.add_argument("--max_moves", type=int, default=200)
    args = parser.parse_args()

    result, sf_elo = evaluate(
        model_path=args.model,
        stockfish_path=args.stockfish_path,
        num_games=args.games,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        stockfish_skill=args.stockfish_skill,
        stockfish_depth=args.stockfish_depth,
        stockfish_elo=args.stockfish_elo,
        max_moves=args.max_moves,
    )

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(result.summary(sf_elo))


if __name__ == "__main__":
    main()
