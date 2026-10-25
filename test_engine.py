import chess
import chess.pgn
import random
import time
from tensorflow.keras.models import load_policy_network
from searching import negamax, order_moves

MODEL_PATH = r"models/cnn_model.h5"

print("Loading CNN model...")
model = load_policy_network()(MODEL_PATH)
print("Model loaded.\n")

def play_full_game(max_depth=3, time_limit_per_move=5, verbose=True):
    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True):
        start_time = time.time()
        best_move = None
        best_score = -1e9

        moves = order_moves(board, model, ply=0)
        for move in moves:
            board.push(move)
            score = -negamax(board, max_depth-1, -1e9, 1e9, model, ply=1)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if (time.time() - start_time) > time_limit_per_move:
                break

        if best_move is None:
            best_move = random.choice(list(board.legal_moves))

        board.push(best_move)
        move_count += 1

        if verbose:
            print(f"Move {move_count}: {best_move} | Eval: {best_score}")
            print(board, "\n")

        if move_count > 200:
            print("Game exceeded 200 moves, stopping.")
            break

    result = board.result(claim_draw=True)
    print("Game over. Result:", result)
    print("Final FEN:", board.fen())

    # Save PGN
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Test Game"
    game.headers["White"] = "CNN Engine"
    game.headers["Black"] = "CNN Engine"
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)
    with open("engine_test_game.pgn", "w") as f:
        f.write(str(game))

    return board, result

if __name__ == "__main__":
    play_full_game(max_depth=3, time_limit_per_move=5)