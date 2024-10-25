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

# due to the size of the pgn file, i have to process each game and write it
# onto the output file one at a time, otherwise it would use all my memory and crash

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


def save_data(x_data, y_data, file_path):
    # save data in batches using pickle
    # Check if the file exists
    with open(file_path, "ab") as file:
        pickle.dump(list(zip(x_data, y_data)), file)
    
    x_data.clear()
    y_data.clear()


def load_batches(filename):
    
    # loads batches of data from a pickle file
    with open(filename, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                yield batch
            except EOFError:
                break

def prepare_training_dataset(in_file, out_file):


    all_x = []
    all_y = []

    game_count = 0

    for game in load_pgn(in_file):
        game_count += 1
        x, y = extract_board_and_labels(game)

        # extend instead of append to add elements not lists
        all_x.extend(x)
        all_y.extend(y)
        print(f"game: {game_count}")
        
        if game_count % 1000 == 0:
            save_data(all_x, all_y, out_file)
            all_x.clear()
            all_y.clear()
            


    if all_x and all_y:
        save_data(all_x, all_y, out_file)


prepare_training_dataset("lichess_2020_oct_filtered.pgn", "processed_lichess_2020_oct_filtered.pkl")
