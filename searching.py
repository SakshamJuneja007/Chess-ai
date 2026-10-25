import numpy as np
import chess
import time
from evaluator import evaluate_advanced
from encoding import encode_board
from move_encoding import encode_move # Note the new import name

# -----------------------------
# SETTINGS
# -----------------------------
TOP_K = 20             
USE_CNN_DEPTH = 1      # CRITICAL: Must be 1 for fast CPU Alpha-Beta search
MAX_DEPTH = 6          
transposition_table = {}

def prepare_tensor(board):
    tensor = encode_board(board)
    return np.expand_dims(tensor, axis=0)

def order_moves(board, model, ply):
    legal_moves = list(board.legal_moves)

    # Fallback: captures first if deep in tree or no model
    if model is None or ply >= USE_CNN_DEPTH:
        captures = [m for m in legal_moves if board.is_capture(m)]
        non_captures = [m for m in legal_moves if not board.is_capture(m)]
        return captures + non_captures

    # CNN-based ordering
    tensor = prepare_tensor(board)
    
    # Dual-head output: we only need the policy (index 0) for move ordering
    policy_batch, _ = model.predict(tensor, verbose=0)
    policy_probs = policy_batch[0]

    move_scores = []
    for move in legal_moves:
        try:
            # New AlphaZero encoder automatically handles perspective mirroring
            idx = encode_move(board, move)
            score = int(policy_probs[idx] * 10000)
        except ValueError:
            score = 0

        # Tactical bonuses
        if board.is_capture(move): score += 1000
        if board.gives_check(move): score += 500
        if board.is_promotion(move): score += 800

        # Penalize early queen moves
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.QUEEN and board.fullmove_number < 10:
            score -= 1000

        move_scores.append((score, move))

    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in move_scores[:TOP_K]]

# -----------------------------
# QUIESCENCE SEARCH
# -----------------------------
def quiescence(board, alpha, beta):
    stand_pat = evaluate_advanced(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move) or board.is_promotion(move):
            board.push(move)
            score = -quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
    return alpha

# -----------------------------
# NEGAMAX SEARCH
# -----------------------------
def negamax(board, depth, alpha, beta, model=None, ply=0):
    key = board.zobrist_hash() # Better hash function built into python-chess

    # Extend search in check
    if board.is_check():
        depth += 1

    # Transposition table lookup
    if key in transposition_table:
        stored_depth, stored_score = transposition_table[key]
        if stored_depth >= depth:
            return stored_score

    if depth <= 0 or board.is_game_over():
        return quiescence(board, alpha, beta)

    max_eval = -1e9
    moves = order_moves(board, model, ply)

    for move in moves:
        # Tactical extensions: extend if capture or giving check
        extend = 1 if board.is_capture(move) or board.gives_check(move) else 0

        board.push(move)
        score = -negamax(board, depth - 1 + extend, -beta, -alpha, model, ply + 1)
        board.pop()

        max_eval = max(max_eval, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff

    transposition_table[key] = (depth, max_eval)
    return max_eval

# -----------------------------
# FIND BEST MOVE
# -----------------------------
def find_best_move(board, model=None, max_depth=MAX_DEPTH, time_limit=6):
    best_move = None
    start_time = time.time()
    
    # Clear TT to avoid memory bloat between moves
    transposition_table.clear() 

    for depth in range(1, max_depth + 1):
        best_eval = -1e9
        moves = order_moves(board, model, ply=0)

        for move in moves:
            if time.time() - start_time > time_limit:
                print(f"Time limit reached. Returning best move found up to depth {depth-1}.")
                return best_move

            # Tactical extension for the first ply
            extend = 1 if board.is_capture(move) or board.gives_check(move) else 0

            board.push(move)
            score = -negamax(board, depth - 1 + extend, -1e9, 1e9, model, ply=1)
            board.pop()

            if score > best_eval:
                best_eval = score
                best_move = move

        print(f"[Depth {depth}] Best Move: {best_move}, Eval: {best_eval}")

    return best_move