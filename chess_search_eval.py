import chess

infinity = float('inf')

piece_values = {chess.PAWN: 100, chess.BISHOP: 300, chess.KNIGHT: 300, chess.ROOK: 500, chess.QUEEN: 900} 


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

def evaluate_position(board):

    if board.is_checkmate():
        if board.turn: # white turn
            return -infinity
        else:
            return infinity

    pawn_position_values = [
    [ 0, 0, 0, 0, 0, 0, 0, 0],
    [ 5, 5, 5, -10, -10, 5, 5, 5],
    [ 1, 1, 2, 0, 0, 2, 1, 1],
    [ 0.5, 0.5, 1, 1.5, 1.5, 1, 0.5, 0.5],
    [ 0, 0, 0, 2, 2, 0, 0, 0],
    [ 0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5],
    [ 0.5, 1, 1, -2, -2, 1, 1, 0.5],
    [ 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    piece_values = {chess.PAWN: 1, chess.BISHOP: 3, chess.KNIGHT: 3, chess.ROOK: 5, chess.QUEEN: 9} 
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += pawn_position_values[chess.square_rank(square)][chess.square_file(square)]
    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= pawn_position_values[7 - chess.square_rank(square)][chess.square_file(square)]

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


    
def alpha_beta_minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)
        #return quiescence_search(board, alpha, beta, 1) # done at the end because its the only time we care
    
    moves = order_moves(board)

    if maximizing_player:
        max_eval = -float('inf')

        for move in moves:
            board.push(move)
            eval = alpha_beta_minimax(board, depth-1, alpha, beta, False)
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
            eval = alpha_beta_minimax(board, depth-1, alpha, beta, True)
            board.pop()
            min_eval = min(eval, min_eval)
            beta = min(beta, eval)

            if beta <= alpha:
                break
        return min_eval



def find_best_move(board, depth):
    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
    else:
        best_value = float('inf')

    for move in board.legal_moves:
        board.push(move)
        #move_value = minimax(board, 2, not board.turn)
        move_value = alpha_beta_minimax(board, depth-1, alpha=-float('inf'), beta=float('inf'),  maximizing_player=not board.turn)
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



    

