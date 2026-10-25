import chess
import numpy as np

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board into an (8, 8, 23) tensor representation.
    Strictly enforces the side-to-move perspective (AlphaZero style).
    
    Planes layout:
      0-5:   Current player pieces (P, N, B, R, Q, K)
      6-11:  Opponent pieces (P, N, B, R, Q, K)
      12-15: Castling rights (WK, WQ, BK, BQ relative to side-to-move)
      16:    En Passant square
      17:    Halfmove clock (capped at 100, normalized 0.0 - 1.0)
      18:    Fullmove number (capped at 100, normalized 0.0 - 1.0)
      19:    Attack map: Squares attacked by current player
      20:    Attack map: Squares attacked by opponent
      21:    Repetition tracker (1.0 if position occurred 3 times)
      22:    Check status (1.0 if current player is in check)
    """
    # 1. Perspective Alignment: Side-to-move is ALWAYS treated as White
    working_board = board.mirror() if board.turn == chess.BLACK else board

    # Shape updated to 23 channels
    tensor = np.zeros((8, 8, 23), dtype=np.float32)

    # 2. Piece Placements (Channels 0-11)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for i, piece in enumerate(piece_types):
        for sq in working_board.pieces(piece, chess.WHITE):
            tensor[chess.square_rank(sq), chess.square_file(sq), i] = 1.0
        for sq in working_board.pieces(piece, chess.BLACK):
            tensor[chess.square_rank(sq), chess.square_file(sq), i + 6] = 1.0

    # 3. Castling Rights (Channels 12-15)
    if working_board.has_kingside_castling_rights(chess.WHITE): tensor[:, :, 12] = 1.0
    if working_board.has_queenside_castling_rights(chess.WHITE): tensor[:, :, 13] = 1.0
    if working_board.has_kingside_castling_rights(chess.BLACK): tensor[:, :, 14] = 1.0
    if working_board.has_queenside_castling_rights(chess.BLACK): tensor[:, :, 15] = 1.0

    # 4. En Passant (Channel 16)
    if working_board.has_legal_en_passant():
        ep_sq = working_board.ep_square
        tensor[chess.square_rank(ep_sq), chess.square_file(ep_sq), 16] = 1.0

    # 5. Move Counters (Channels 17-18) - FIXED: properly capped at 1.0
    tensor[:, :, 17] = min(working_board.halfmove_clock, 100) / 100.0  
    tensor[:, :, 18] = min(working_board.fullmove_number, 100) / 100.0

    # 6. Attack Maps (Channels 19-20)
    for sq in chess.SQUARES:
        r, f = chess.square_rank(sq), chess.square_file(sq)
        if working_board.is_attacked_by(chess.WHITE, sq): tensor[r, f, 19] = 1.0
        if working_board.is_attacked_by(chess.BLACK, sq): tensor[r, f, 20] = 1.0

    # 7. Repetition Tracker (Channel 21) - FIXED: tracks true 3-fold draw threshold
    if working_board.is_repetition(3):
        tensor[:, :, 21] = 1.0

    # 8. Check Status (Channel 22) - NEW: explicit tactical flag for the network
    if working_board.is_check():
        tensor[:, :, 22] = 1.0

    return tensor