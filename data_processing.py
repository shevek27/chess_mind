import pickle
import numpy as np
import chess
import chess.pgn
import os


def encode_board(board):

    encoded_board = np.zeros((8,8,18)) # 18 = 6x2 pieces + 4 castling rights + 1 en_passant + 1 turn

    piece_values = {chess.PAWN:0, chess.BISHOP:1, chess.KNIGHT:2, chess.ROOK:3, chess.QUEEN:4, chess.KING:5}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            layer = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                layer += 6 # different values for black pieces
            encoded_board[row][col][layer] = 1

    
    # castling 
    if board.has_kingside_castling_rights(chess.WHITE):
        encoded_board[:, :, 12] = 1  
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded_board[:, :, 13] = 1  
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded_board[:, :, 14] = 1  
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded_board[:, :, 15] = 1  

    # en passant
    if board.has_legal_en_passant():
        en_passant_square = board.ep_square
        row = en_passant_square // 8
        col = en_passant_square % 8
        encoded_board[row][col][16] = 1

    # player turn
    encoded_board[:, :, 17] = int(board.turn) # all rows, all columns, 18th channel (0 index)

    return encoded_board


def load_pgn(pgn_file):
    # returns the games one by one
    with open(pgn_file) as games_file:
        while True:
            game = chess.pgn.read_game(games_file)
            if game is None:
                break

            yield game # returns 1 at a time, incredible function


def extract_board_and_labels(game):
    x = [] # boards
    y = [] # labels
    board = game.board()
    result = game.headers["Result"]

    if result == "1-0": # white won
        label = 1
    elif result == "0-1":
        label = -1
    else:
        label = 0
    
    for move in game.mainline_moves():
        board.push(move)
        encoded_board = encode_board(board)
        x.append(encoded_board)
        y.append(label)

    return np.array(x), np.array(y)

def prepare_training_dataset(in_file, out_file):

    total_features = 0
    i = 1
    all_x = []
    all_y = []

    game_count = 0

    for game in load_pgn(in_file):
        game_count += 1
        x, y = extract_board_and_labels(game)

        # extend instead of append to add elements not lists
        all_x.extend(x)
        all_y.extend(y)
        

        if game_count % 100 == 0:
            print(f"game number: {game_count}")
            print(f"actual number of features = {len(all_x)}")
            print(f"total number of features = {total_features + len(all_x)}")



    #if all_x and all_y:
    #    save_data(all_x, all_y, out_file)
        if len(all_x) > 500000:
            filename = f"processed_dataset_{i}.npz"
            np.savez(filename, features=all_x, labels=all_y)
            i += 1
            total_features += len(all_x)
            all_x.clear()
            all_y.clear()

    all_x.clear()
    all_y.clear()



if __name__ == "__main__":
    prepare_training_dataset("lichess_2020_oct_filtered.pgn", "test_processed_lichess_2020_oct_filtered.pkl")