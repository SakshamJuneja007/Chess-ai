import os
import warnings
import chess
import chess.pgn
import numpy as np
import tensorflow as tf
import csv

# ----------------- Suppress TensorFlow Warnings -----------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only errors
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------- Load Models -----------------
model16 = tf.keras.models.load_model('D:/chess-ai/models/resnet_model.h5')    # shape 8,8,16
model17 = tf.keras.models.load_model('D:/chess-ai/models/resnet_model.keras') # shape 8,8,17

# ----------------- Board -> Model Input -----------------
def board_to_input(board, model_input_planes=17):
    planes = np.zeros((8, 8, model_input_planes), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        # For models with extra plane (like 17), leave last plane as zeros
        if plane < model_input_planes:
            planes[row, col, plane] = 1
    return planes[np.newaxis, ...]

# ----------------- Select Move -----------------
def select_move(model, board):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    scores = []
    model_planes = model.input_shape[-1]
    for move in legal_moves:
        board.push(move)
        input_board = board_to_input(board, model_planes)
        board.pop()
        pred = model.predict(input_board, verbose=0)
        # Adjust if dual-head: take first head if tuple
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        scores.append(pred.sum())  # simple scoring
    best_move = legal_moves[np.argmax(scores)]
    return best_move

# ----------------- Play Game -----------------
def play_game(white_model, black_model, max_moves=200):
    board = chess.Board()
    move_history = []
    print("\n=== New Game ===")
    for turn in range(max_moves):
        if board.is_game_over():
            break
        model = white_model if board.turn == chess.WHITE else black_model
        move = select_move(model, board)
        if move is None:
            break
        board.push(move)
        move_history.append(move.uci())
        # Print move
        color = "White" if board.turn == chess.BLACK else "Black"
        print(f"{color} plays: {move.uci()}")
        # Avoid simple repetition loops
        if len(move_history) > 10 and move_history[-10:] == move_history[-20:-10]:
            break
    result = board.result()
    print("Final Board:")
    print(board)
    print("Result:", result)
    return result, move_history

# ----------------- Evaluation Loop -----------------
num_games = 20  # adjust as you like
stats = {"Model16": 0, "Model17": 0, "Draw": 0}

# Open CSV to save results
with open("chess_model_battle_results.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Game", "White", "Black", "Result", "Moves"])

    for i in range(num_games):
        white_model = model17 if i % 2 == 0 else model16
        black_model = model16 if i % 2 == 0 else model17
        white_name = "Model17" if white_model == model17 else "Model16"
        black_name = "Model16" if black_model == model16 else "Model17"

        result, moves = play_game(white_model, black_model)

        if result == "1-0":
            winner = "Model17" if white_model == model17 else "Model16"
            stats[winner] += 1
        elif result == "0-1":
            winner = "Model16" if black_model == model16 else "Model17"
            stats[winner] += 1
        else:
            stats["Draw"] += 1

        # Save game to CSV
        writer.writerow([i+1, white_name, black_name, result, " ".join(moves)])

# ----------------- Final Stats -----------------
print("\n=== Final Stats ===")
print(f"Model16 Wins: {stats['Model16']}")
print(f"Model17 Wins: {stats['Model17']}")
print(f"Draws: {stats['Draw']}")