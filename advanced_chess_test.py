import chess
import chess.engine
import pygame
import sys
import matplotlib.pyplot as plt
from run_engine import get_engine_move

# -----------------------------
# CONFIG
# -----------------------------
STOCKFISH_PATH = r"D:/chess-ai/stockfish-windows-x86-64-avx2.exe"
DEPTH = 3  # Fixed depth for Stockfish

# GUI
SQUARE_SIZE = 80
BOARD_SIZE = 8 * SQUARE_SIZE

# -----------------------------
# INIT
# -----------------------------
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
pygame.display.set_caption("Chess AI Viewer")

# -----------------------------
# DRAW BOARD
# -----------------------------
def draw_board(board):
    colors = [(240, 217, 181), (181, 136, 99)]

    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color,
                             (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)

            font = pygame.font.SysFont(None, 40)
            text = font.render(piece.symbol(), True, (0, 0, 0))
            screen.blit(text, (col*SQUARE_SIZE+25, row*SQUARE_SIZE+25))

# -----------------------------
# MOVE QUALITY
# -----------------------------
def analyze_move(board, move):
    info_before = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    eval_before = info_before["score"].white().score(mate_score=10000)

    board.push(move)

    info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
    eval_after = info_after["score"].white().score(mate_score=10000)

    board.pop()

    if eval_before is None or eval_after is None:
        return "UNKNOWN", 0

    diff = eval_after - eval_before

    if diff > 50:
        return "GOOD", diff
    elif diff > -50:
        return "OK", diff
    elif diff > -200:
        return "INACCURACY", diff
    elif diff > -500:
        return "MISTAKE", diff
    else:
        return "BLUNDER", diff

# -----------------------------
# GAME LOOP
# -----------------------------
def play_game():
    board = chess.Board()
    eval_history = []

    move_no = 1

    while not board.is_game_over():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        draw_board(board)
        pygame.display.flip()

        if board.turn:
            move = get_engine_move(board)
            player = "Your Engine"
        else:
            # Stockfish fixed depth 3
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            move = info["pv"][0]  # Best move at depth 3
            player = "Stockfish"
            print(f"Stockfish move (depth {DEPTH}): {move}")

        quality, diff = analyze_move(board, move)

        print(f"Move {move_no}: {player} -> {move} | {quality} ({diff})")

        if quality == "BLUNDER":
            print("💀 BLUNDER DETECTED!")
        elif quality == "MISTAKE":
            print("⚠️ Mistake")

        board.push(move)
        eval_history.append(diff)

        move_no += 1

        pygame.time.delay(300)

    print("\nResult:", board.result())

    # -----------------------------
    # GRAPH (LIGHTWEIGHT)
    # -----------------------------
    plt.plot(eval_history)
    plt.title("Move Quality Over Time")
    plt.xlabel("Moves")
    plt.ylabel("Eval Change")
    plt.show()

# -----------------------------
# RUN
# -----------------------------
play_game()

engine.quit()