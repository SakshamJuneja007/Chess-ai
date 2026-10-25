"""
move_encoding.py
=================

AlphaZero-style move encoding for chess: a fixed 4672-dimensional
action space, organised as 64 (from-square) x 73 (move-plane).

    action_index = from_square * 73 + plane_index   (0 <= action_index < 4672)

Move-plane layout (0-72)
-------------------------
 0 - 55  : "Queen-like" sliding moves.
           8 directions x 7 distances = 56 planes.
           Direction order (index 0-7):
               0: N   (rank+1, file 0)
               1: NE  (rank+1, file+1)
               2: E   (rank 0, file+1)
               3: SE  (rank-1, file+1)
               4: S   (rank-1, file 0)
               5: SW  (rank-1, file-1)
               6: W   (rank 0, file-1)
               7: NW  (rank+1, file-1)
           Distance is 1..7 (encoded as distance-1, i.e. 0..6).
           plane = direction * 7 + (distance - 1)

           This covers all Rook/Bishop/Queen moves, all King moves
           (distance 1), all standard (non-capture/non-promotion-
           specific) Pawn pushes/captures, and all queen-promotions
           (a queen-promotion is encoded as the underlying queen-like
           move; see `_encode_promotion` below for why distance is
           always 1 for promotions, which fits planes 0,7,14,...).

 56 - 63 : Knight moves (8 possible knight jumps), in this fixed order:
               56: (+2, +1)
               57: (+1, +2)
               58: (-1, +2)
               59: (-2, +1)
               60: (-2, -1)
               61: (-1, -2)
               62: (+1, -2)
               63: (+2, -1)

 64 - 72 : Underpromotions. A pawn reaching the last rank can promote
           to Knight, Bishop, or Rook (Queen promotion is handled via
           the queen-like planes 0-55, using the straight/diagonal
           direction with distance 1).
           3 directions (forward, capture-left/diagonal, capture-right/
           diagonal) x 3 piece types (N, B, R) = 9 planes.
               64: forward push   -> Knight
               65: forward push   -> Bishop
               66: forward push   -> Rook
               67: capture left   -> Knight   (diagonal toward queenside)
               68: capture left   -> Bishop
               69: capture left   -> Rook
               70: capture right  -> Knight   (diagonal toward kingside)
               71: capture right  -> Bishop
               72: capture right  -> Rook

Perspective / mirroring
------------------------
All moves are encoded from the perspective of the player to move, i.e.
exactly the same canonical orientation used by `encoding.encode_board`.
If it is Black's turn, both the `from_square` and `to_square` of the
move are first mirrored with `chess.square_mirror` (rank flip) before
the direction/plane is computed. `decode_action` mirrors the squares
back for Black so the returned `chess.Move` is valid on the *real*
(unmirrored) board.

Castling
--------
Castling moves are represented in python-chess as a king move of two
squares (e.g. e1g1 for White kingside). This is exactly a "queen-like"
move with distance 2 along the E direction (or W for queenside), so it
is naturally covered by planes 0-55 -- no special-casing is required
for encoding *king* moves. `decode_action` reconstructs the move as
`chess.Move(from_square, to_square)`; python-chess's `board.push()`
correctly interprets a king e1->g1 move as castling as long as the
move is legal in context. To guarantee the returned move is recognised
as the *castling* move (rather than rejected because Move() of itself
doesn't set `is_castling`), `decode_action` matches the
reconstructed (from, to[, promotion]) triple against
`board.legal_moves` and returns the exact legal `Move` object found
there. This also robustly resolves en-passant (UCI move has no special
flag besides from/to in python-chess) and any edge cases.

En passant
----------
An en-passant capture is a normal pawn diagonal move (distance 1,
diagonal direction), so it is covered by the queen-like planes with
distance 1. As with castling, `decode_action` resolves the final
`chess.Move` object via `board.legal_moves` so the en-passant flag is
set correctly by python-chess itself.

Public API
----------
    encode_move(board, move) -> int            # 0 <= result < 4672
    decode_action(board, action) -> chess.Move  # legal move on `board`
    get_legal_action_mask(board) -> np.ndarray  # shape (4672,), {0,1}
"""

from __future__ import annotations

import numpy as np
import chess


ACTION_SPACE_SIZE = 64 * 73  # 4672

# --- Queen-like directions: (delta_rank, delta_file), index 0-7 ---
QUEEN_DIRECTIONS = [
    (1, 0),   # 0 N
    (1, 1),   # 1 NE
    (0, 1),   # 2 E
    (-1, 1),  # 3 SE
    (-1, 0),  # 4 S
    (-1, -1), # 5 SW
    (0, -1),  # 6 W
    (1, -1),  # 7 NW
]
_QUEEN_DIR_LOOKUP = {d: i for i, d in enumerate(QUEEN_DIRECTIONS)}

# --- Knight moves: (delta_rank, delta_file), planes 56-63 ---
KNIGHT_MOVES = [
    (2, 1),    # 56
    (1, 2),    # 57
    (-1, 2),   # 58
    (-2, 1),   # 59
    (-2, -1),  # 60
    (-1, -2),  # 61
    (1, -2),   # 62
    (2, -1),   # 63
]
_KNIGHT_LOOKUP = {d: i for i, d in enumerate(KNIGHT_MOVES)}

# --- Underpromotion piece order ---
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_LOOKUP = {p: i for i, p in enumerate(UNDERPROMOTION_PIECES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_mirror_square(square: int, mirror: bool) -> int:
    return chess.square_mirror(square) if mirror else square


def _rc(square: int) -> tuple[int, int]:
    return chess.square_rank(square), chess.square_file(square)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a legal `chess.Move` (for `board.turn`) into an action index.

    The move must be legal in `board` (or at least pseudo-legal with a
    well-defined direction/distance; promotions must have `promotion`
    set appropriately).
    """
    mirror = (board.turn == chess.BLACK)

    from_sq = _maybe_mirror_square(move.from_square, mirror)
    to_sq = _maybe_mirror_square(move.to_square, mirror)

    fr, fc = _rc(from_sq)
    tr, tc = _rc(to_sq)

    dr = tr - fr
    dc = tc - fc

    # ----- Underpromotions (Knight / Bishop / Rook) -----
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO_LOOKUP[move.promotion]

        # Determine direction bucket: forward (dc == 0), or diagonal.
        # "Left"/"right" are defined in the *mirrored/canonical* frame,
        # i.e. relative to the player-to-move moving "up" the board
        # (increasing rank). Left = decreasing file, Right = increasing file.
        if dc == 0:
            dir_bucket = 0  # forward
        elif dc == -1:
            dir_bucket = 1  # capture toward decreasing file ("left")
        elif dc == 1:
            dir_bucket = 2  # capture toward increasing file ("right")
        else:
            raise ValueError(f"Invalid underpromotion file delta: {dc}")

        plane = 64 + dir_bucket * 3 + piece_idx
        return from_sq * 73 + plane

    # ----- Knight moves -----
    if (dr, dc) in _KNIGHT_LOOKUP:
        plane = 56 + _KNIGHT_LOOKUP[(dr, dc)]
        return from_sq * 73 + plane

    # ----- Queen-like moves (covers Queen-promotions, King incl.
    #        castling, Rook/Bishop/Queen slides, single-step Pawn
    #        pushes/captures, en passant) -----
    distance = max(abs(dr), abs(dc))
    if distance == 0:
        raise ValueError("Zero-length move cannot be encoded.")

    step_r = (dr // distance)
    step_c = (dc // distance)

    if (step_r, step_c) not in _QUEEN_DIR_LOOKUP:
        raise ValueError(
            f"Move delta ({dr},{dc}) is not a knight move and does not "
            f"correspond to a straight-line direction; cannot encode."
        )

    direction = _QUEEN_DIR_LOOKUP[(step_r, step_c)]
    plane = direction * 7 + (distance - 1)
    return from_sq * 73 + plane


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_action(board: chess.Board, action: int) -> chess.Move:
    """Decode an action index into a legal `chess.Move` for `board`.

    Raises `ValueError` if the action does not correspond to any legal
    move in the current position (this can happen if the policy head
    picks an illegal/impossible action -- callers should mask the
    policy with `get_legal_action_mask` before sampling/argmax to avoid
    this in practice).
    """
    if not (0 <= action < ACTION_SPACE_SIZE):
        raise ValueError(f"Action {action} out of range [0, {ACTION_SPACE_SIZE}).")

    from_sq_canon = action // 73
    plane = action % 73

    mirror = (board.turn == chess.BLACK)

    fr, fc = _rc(from_sq_canon)

    promotion_piece = None

    if plane < 56:
        direction = plane // 7
        distance = (plane % 7) + 1
        step_r, step_c = QUEEN_DIRECTIONS[direction]
        dr, dc = step_r * distance, step_c * distance
    elif plane < 64:
        dr, dc = KNIGHT_MOVES[plane - 56]
    else:
        underpromo_idx = plane - 64
        dir_bucket = underpromo_idx // 3
        piece_idx = underpromo_idx % 3
        promotion_piece = UNDERPROMOTION_PIECES[piece_idx]

        # Forward = +1 rank (in canonical frame), left = -1 file, right = +1 file
        dr = 1
        dc = {0: 0, 1: -1, 2: 1}[dir_bucket]

    tr, tc = fr + dr, fc + dc

    if not (0 <= tr < 8 and 0 <= tc < 8):
        raise ValueError(
            f"Decoded target square out of bounds for action {action} "
            f"(from=({fr},{fc}), delta=({dr},{dc}))."
        )

    to_sq_canon = chess.square(tc, tr)

    from_sq = _maybe_mirror_square(from_sq_canon, mirror)
    to_sq = _maybe_mirror_square(to_sq_canon, mirror)

    # ----- Determine queen-promotion automatically -----
    # If the action corresponds to a pawn moving from its 7th rank to
    # the 8th rank (in canonical frame: fr == 6 -> tr == 7) via a
    # queen-like plane, and `promotion_piece` is still None, this MUST
    # be a queen promotion (pawns cannot otherwise move to rank 8 via
    # planes 0-55 without promoting).
    moving_piece = board.piece_at(from_sq)
    if (
        promotion_piece is None
        and moving_piece is not None
        and moving_piece.piece_type == chess.PAWN
        and fr == 6 and tr == 7
    ):
        promotion_piece = chess.QUEEN

    candidate = chess.Move(from_sq, to_sq, promotion=promotion_piece)

    # ----- Resolve against legal moves to correctly set special flags
    #        (castling, en passant) and validate legality. -----
    for legal in board.legal_moves:
        if (
            legal.from_square == candidate.from_square
            and legal.to_square == candidate.to_square
            and legal.promotion == candidate.promotion
        ):
            return legal

    # Fallback: some engines encode castling as king-moves-two-squares
    # even when chess960/standard differences cause from/to mismatches
    # with `legal.promotion` defaults of None vs a piece. Try matching
    # ignoring promotion as a last resort (covers the case where
    # `candidate.promotion` is QUEEN but python-chess's legal move for
    # a forced promotion already defaults appropriately, or vice versa).
    for legal in board.legal_moves:
        if (
            legal.from_square == candidate.from_square
            and legal.to_square == candidate.to_square
        ):
            return legal

    raise ValueError(
        f"Action {action} (from={chess.square_name(from_sq)}, "
        f"to={chess.square_name(to_sq)}, promo={promotion_piece}) does not "
        f"correspond to a legal move in the current position.\nFEN: {board.fen()}"
    )


# ---------------------------------------------------------------------------
# Legal action mask
# ---------------------------------------------------------------------------

def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return a (4672,) float32 array with 1.0 at indices of legal moves."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        idx = encode_move(board, move)
        mask[idx] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Self-test (run directly: `python move_encoding.py`)
# ---------------------------------------------------------------------------

def _self_test(num_random_games: int = 200, max_plies: int = 60, seed: int = 0) -> None:
    """Round-trip test: for many random games, every legal move at every
    position must encode to a unique action index, and decoding that
    action index must reproduce a move equivalent to the original
    (same from/to/promotion on the real board).
    """
    import random

    rng = random.Random(seed)
    total_moves_checked = 0

    for game_idx in range(num_random_games):
        board = chess.Board()
        for ply in range(max_plies):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)

            # Check round-trip + uniqueness for every legal move.
            seen_actions = {}
            for mv in legal_moves:
                action = encode_move(board, mv)
                if not (0 <= action < ACTION_SPACE_SIZE):
                    raise AssertionError(f"Action out of range: {action}")

                if action in seen_actions and seen_actions[action] != mv:
                    raise AssertionError(
                        f"Action collision at ply {ply}, FEN {board.fen()}: "
                        f"{seen_actions[action]} and {mv} both map to {action}"
                    )
                seen_actions[action] = mv

                decoded = decode_action(board, action)
                # The decoded move must be legal and match from/to/promo
                if decoded not in legal_moves:
                    raise AssertionError(
                        f"Decoded move {decoded} not legal at ply {ply}, "
                        f"FEN {board.fen()} (action {action}, original {mv})"
                    )
                if (decoded.from_square, decoded.to_square, decoded.promotion) != (
                    mv.from_square, mv.to_square, mv.promotion
                ):
                    raise AssertionError(
                        f"Round-trip mismatch at ply {ply}, FEN {board.fen()}: "
                        f"original {mv} -> action {action} -> decoded {decoded}"
                    )

                total_moves_checked += 1

            # Mask sanity check
            mask = get_legal_action_mask(board)
            if int(mask.sum()) != len(legal_moves):
                raise AssertionError(
                    f"Mask has {int(mask.sum())} ones but {len(legal_moves)} "
                    f"legal moves at ply {ply}, FEN {board.fen()}"
                )

            move = rng.choice(legal_moves)
            board.push(move)

    print(f"Self-test passed. Checked {total_moves_checked} (position, move) pairs "
          f"across {num_random_games} games.")


if __name__ == "__main__":
    _self_test()
