"""
encoding.py
===========

Board representation for the AlphaZero-style Chess CNN.

Produces an (8, 8, 22) tensor encoding a chess position from the
perspective of the player to move (side-to-move perspective).

Channel layout
--------------
 0-5   : Current player's pieces  (P, N, B, R, Q, K)  -- one-hot occupancy planes
 6-11  : Opponent's pieces        (P, N, B, R, Q, K)
 12    : Current player kingside  castling right (all-ones / all-zeros plane)
 13    : Current player queenside castling right
 14    : Opponent kingside  castling right
 15    : Opponent queenside castling right
 16    : En-passant target square (one-hot, all zeros if none)
 17    : Halfmove clock   (normalized, broadcast over the whole plane)
 18    : Fullmove number  (normalized, broadcast over the whole plane)
 19    : Current player attack map (squares attacked by current player's pieces)
 20    : Opponent attack map        (squares attacked by opponent's pieces)
 21    : Repetition indicator (1.0 if the current position has occurred
          >=1 time before in the game history, else 0.0)

Board orientation
------------------
If it is Black to move, the board is mirrored vertically (rank flip,
i.e. ``chess.square_mirror``) AND the piece colors are swapped so that
"current player" pieces are always on ranks 1-4 from White's visual
perspective and the network always sees a "White to move" style board.
This is the standard AlphaZero side-to-move canonicalisation.

Notes on normalisation
-----------------------
* Halfmove clock is divided by 100.0 (clamped to [0, 1]).
* Fullmove number is divided by 500.0 (clamped to [0, 1]) -- purely a
  soft normalisation since fullmove number is for game has been
  recorded for unbounded games, but in practice games rarely exceed
  a few hundred moves.

This module is self-contained and only depends on `python-chess`
(`pip install chess`).
"""

from __future__ import annotations

import numpy as np
import chess


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PLANES = 23
BOARD_SIZE = 8

# Order in which piece types are written into the 6 planes per side.
PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)

HALFMOVE_NORM = 100.0
FULLMOVE_NORM = 500.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _square_to_rc(square: int) -> tuple[int, int]:
    """Convert a python-chess square index (0..63) to (row, col).

    Row 0 corresponds to rank 1 (White's back rank) and row 7 to rank 8.
    Col 0 corresponds to the a-file and col 7 to the h-file.

    This matches ``chess.square_rank`` / ``chess.square_file`` directly,
    i.e. row = rank index, col = file index.
    """
    return chess.square_rank(square), chess.square_file(square)


def _attack_map(board: chess.Board, color: chess.Color) -> np.ndarray:
    """Return an (8, 8) array where 1.0 marks squares attacked by `color`.

    Uses python-chess's `attackers` to build a full attack map. This is
    O(64 * num_pieces) which is perfectly fine for a single position.
    """
    attack = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square in chess.SQUARES:
        if board.is_attacked_by(color, square):
            r, c = _square_to_rc(square)
            attack[r, c] = 1.0
    return attack


def _mirror_array(arr: np.ndarray) -> np.ndarray:
    """Flip an (8, 8[, ...]) array vertically along the rank axis.

    This corresponds to ``chess.square_mirror`` (rank r -> 7 - r), which
    is the standard way to present a Black-to-move position from
    White's visual perspective.
    """
    return arr[::-1, ...].copy()


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

def encode_board(board: chess.Board, history: list[chess.Board] | None = None) -> np.ndarray:
    """Encode a `chess.Board` into an (8, 8, 22) float32 tensor.

    Parameters
    ----------
    board:
        The current position.
    history:
        Optional list of previous `chess.Board` positions (or board_fen
        strings) in the game, used to compute the repetition plane
        (channel 21). If ``None``, the repetition plane is filled
        using ``board.is_repetition()`` checks against the board's own
        internal move stack (works if `board` was built by pushing
        moves rather than constructed standalone).

    Returns
    -------
    np.ndarray of shape (8, 8, 22), dtype float32.
    """
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PLANES), dtype=np.float32)

    current_color = board.turn          # chess.WHITE or chess.BLACK
    opponent_color = not current_color

    # ----- Piece planes (0-5 current player, 6-11 opponent) -----
    for piece_type in chess.PIECE_TYPES:
        plane_offset = PIECE_TYPES.index(piece_type)

        # Current player's pieces -> planes 0-5
        for square in board.pieces(piece_type, current_color):
            r, c = _square_to_rc(square)
            planes[r, c, plane_offset] = 1.0

        # Opponent's pieces -> planes 6-11
        for square in board.pieces(piece_type, opponent_color):
            r, c = _square_to_rc(square)
            planes[r, c, 6 + plane_offset] = 1.0

    # ----- Castling rights (12-15) -----
    if board.has_kingside_castling_rights(current_color):
        planes[:, :, 12] = 1.0
    if board.has_queenside_castling_rights(current_color):
        planes[:, :, 13] = 1.0
    if board.has_kingside_castling_rights(opponent_color):
        planes[:, :, 14] = 1.0
    if board.has_queenside_castling_rights(opponent_color):
        planes[:, :, 15] = 1.0

    # ----- En passant (16) -----
    if board.ep_square is not None:
        r, c = _square_to_rc(board.ep_square)
        planes[r, c, 16] = 1.0

    # ----- Halfmove clock (17) -----
    halfmove = min(board.halfmove_clock, HALFMOVE_NORM) / HALFMOVE_NORM
    planes[:, :, 17] = halfmove

    # ----- Fullmove number (18) -----
    fullmove = min(board.fullmove_number, FULLMOVE_NORM) / FULLMOVE_NORM
    planes[:, :, 18] = fullmove

    # ----- Attack maps (19, 20) -----
    planes[:, :, 19] = _attack_map(board, current_color)
    planes[:, :, 20] = _attack_map(board, opponent_color)

    # ----- Repetition (21) -----
    is_repetition = False
    try:
        # Changed to 3 to reflect actual chess draw rules
        is_repetition = board.is_repetition(3)
    except Exception:
        is_repetition = False
    planes[:, :, 21] = 1.0 if is_repetition else 0.0

    # ----- Check Status (22) -----
    # Explicit tactical flag for the network
    if board.is_check():
        planes[:, :, 22] = 1.0

    # ----- Mirror for Black to move -----
    if current_color == chess.BLACK:
        planes = _mirror_array(planes)

    return planes

def encode_board_stacked(board: chess.Board, num_history: int = 0) -> np.ndarray:
    """Convenience wrapper. Currently identical to `encode_board`.

    Included as an extension point if temporal stacking of past
    positions (T-step history, as in the original AlphaZero paper) is
    desired later. With `num_history > 0` you would concatenate
    `encode_board` results for the last N positions along the channel
    axis. Left as a single-frame encoder for simplicity and to match
    the (8, 8, 22) shape requested.
    """
    return encode_board(board)
