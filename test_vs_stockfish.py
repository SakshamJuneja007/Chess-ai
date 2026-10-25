import chess
import chess.engine
from run_engine import get_engine_move

STOCKFISH_PATH = r"D:/chess-ai/stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
depth=1
board = chess.Board()

while not board.is_game_over():
    print(board, "\n")

    if board.turn:
        move = get_engine_move(board)
        print("Your Engine:", move)
    else:
        result = engine.play(board, chess.engine.Limit(time=0.1))
        move = result.move
        print("Stockfish:", move)

    board.push(move)

print("Result:", board.result())
engine.quit()