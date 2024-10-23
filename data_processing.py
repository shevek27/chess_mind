import numpy as np
import chess
import os

def encode_board(board):

    encoded_board = np.zeros((8,8,18)) # 18 = 6x2 pieces + 4 castling rights + 1 en_passant + 1 turn

    piece_values = {chess.PAWN:1, chess.BISHOP:3, chess.KNIGHT:3, chess.ROOK:5, chess.QUEEN:9}

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
        row, col = divmod(en_passant_square, 8)
        encoded_board[row][col][16] = 1

    # player turn
    encoded_board[:, :, 17] = int(board.turn) # all rows, all columns, 18th channel (0 index)

    return encoded_board


