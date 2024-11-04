import chess
import torch
from chess_model_v1 import *
from data_processing import encode_board


infinity = float('inf')


def order_moves(board):
    # get all capture moves
    captures = []
    others = []

    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            others.append(move)

    return captures + others

def evaluate_position(board, model):
    encoded_board_tensor = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0).permute(0,3,2,1)

    if board.is_checkmate():
        if board.turn: # white turn
            return -infinity
        else:
            return infinity
    with torch.inference_mode():
        score = model(encoded_board_tensor)
    return score


def minimax(board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
    

    if maximizing_player:
        max_eval = -float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth-1, False)
            board.pop()
            max_eval = max(max_eval, eval)

        return max_eval
    
    else:
        min_eval = float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth-1, True)
            board.pop()
            min_eval = min(min_eval, eval)

        return min_eval
    

def quiescence_search(board, alpha, beta, depth_limit):

    if depth_limit <= 0:
        return evaluate_position(board)

    current_eval = evaluate_position(board)

    # if the position is quiet, return current eval
    if current_eval >= beta:
        return beta # beta cutoff
    if alpha < current_eval < beta:
        alpha = current_eval  # update alpha

    # explore captures + checks
    for move in board.legal_moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            score = quiescence_search(board, alpha, beta, depth_limit-1)
            board.pop()

            # if we find a move with better evaluation, update alpha
            if score >= beta:
                return beta # beta cutoff
            if score > alpha:
                alpha = score


    return alpha


    
def alpha_beta_minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model)
        #return quiescence_search(board, alpha, beta, 3) # done at the end because its the only time we care
    
    moves = order_moves(board)

    if maximizing_player:
        max_eval = -float('inf')

        for move in moves:
            board.push(move)
            eval = alpha_beta_minimax(board, depth-1, alpha, beta, False, model=model)
            board.pop()
            max_eval = max(eval, max_eval)
            alpha = max(alpha, eval)
            
            if beta <= alpha:
                break # cut beta branch

        return max_eval
    
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta_minimax(board, depth-1, alpha, beta, True, model=model)
            board.pop()
            min_eval = min(eval, min_eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break
        return min_eval



def find_best_move(board, depth, model):
    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
    else:
        best_value = float('inf')

    for move in board.legal_moves:
        board.push(move)
        #move_value = minimax(board, depth, not board.turn)
        move_value = alpha_beta_minimax(board, depth-1, alpha=infinity, beta=-infinity,  maximizing_player=not board.turn, model=model)
        board.pop()

        if board.turn == chess.WHITE:
            if move_value > best_value:
                best_value = move_value
                best_move = move

        else:
            if move_value < best_value:
                best_value = move_value
                best_move = move


    return best_move

model = chess_model_v1()
model.load_state_dict(torch.load("chess_model_v1.pth"))



    

