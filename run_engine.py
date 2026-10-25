from tensorflow.keras.models import load_model
import chess
import time
from searching import find_best_move

# -----------------------------
# MODEL LOADING
# -----------------------------
MODEL_PATH = "D:/chess-ai/models/resnet_model.keras"
model = load_model(MODEL_PATH)
print("Model Loaded ✅")

# -----------------------------
# ENGINE MOVE FUNCTION
# -----------------------------
def get_engine_move(board, max_depth=6, time_limit=7):
    """
    Returns the best move for the current board position.
    - Uses iterative deepening up to max_depth.
    - CNN used for move ordering for first few plies.
    - Respects the time_limit strictly.
    """
    start_time = time.time()
    best_move = None

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            break  # Stop if time exceeded

        move = find_best_move(board, model=model, max_depth=depth, time_limit=time_limit - (time.time() - start_time))

        if move is not None:
            best_move = move

        print(f"[Depth {depth}] Selected Move: {best_move}")

    return best_move

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    board = chess.Board()

    while not board.is_game_over():
        move = get_engine_move(board, max_depth=6, time_limit=7)
        print(f"Engine plays: {move}")
        board.push(move)

        print(board)
        print("\n---\n")

        # Example: for testing you can push a random legal move for opponent
        if not board.is_game_over():
            opponent_move = list(board.legal_moves)[0]
            board.push(opponent_move)
            print(f"Opponent plays: {opponent_move}")
            print(board)
            print("\n---\n")