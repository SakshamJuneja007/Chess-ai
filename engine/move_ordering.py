import chess

move_to_idx = {}
idx_to_move = {}

def initialize_move_map():

    idx = 0

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:

            move = chess.Move(from_sq, to_sq)

            move_to_idx[move] = idx
            idx_to_move[idx] = move

            idx += 1