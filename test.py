from tensorflow.keras.models import load_model
MODEL_PATH = "D:/chess-ai/models/resnet_model.keras"
model_keras = load_model(MODEL_PATH)

Model_path= "D:\chess-ai\models/resnet_model.h5"
model_h5 = load_model(Model_path)
import chess
import chess.pgn
from tensorflow.keras.models import load_model


# Function to select move (simplified)
def select_move(model, board):
    # Convert board to input array
    # input_board = ...
    pred = model.predict(input_board[None, :, :, :], verbose=0)[0]
    legal_moves = list(board.legal_moves)
    # Choose move with highest predicted value among legal moves
    move = legal_moves[pred[:len(legal_moves)].argmax()]
    return move

# Play single game
def play_game(white_model, black_model):
    board = chess.Board()
    moves_log = []
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = select_move(white_model, board)
        else:
            move = select_move(black_model, board)
        board.push(move)
        moves_log.append(str(move))
    
    result = board.result()
    return result, moves_log

# Play multiple games and save to file
results = {"white_h5":0, "white_keras":0, "draws":0}
with open("model_vs_model_results.txt", "w") as f:
    for i in range(50):
        white, black = (model_h5, model_keras) if i % 2 == 0 else (model_keras, model_h5)
        result, moves = play_game(white, black)
        f.write(f"Game {i+1} | White: {'h5' if i%2==0 else 'keras'} | Result: {result}\n")
        f.write("Moves: " + " ".join(moves) + "\n\n")
        # Update results
        if result == "1-0":
            results["white_h5" if i%2==0 else "white_keras"] += 1
        elif result == "0-1":
            results["white_keras" if i%2==0 else "white_h5"] += 1
        else:
            results["draws"] += 1
    
    f.write(f"\nSummary:\n{results}\n")