import chess
import chess.svg
import pygame
import torch
from chess_drawing import *
from sounds import *
from chess_search_eval import *



print("imports completed")

  
# Initialize Pygame
pygame.init()

def make_move(move, board):
    if move in board.legal_moves:
        move_sound(board, move)
        board.push(move)
        global t
        t += 1
        return True
    else:
        return False

def player_move(board, move, selected_square, square):
    try:
        move = board.find_move(selected_square, square)
    except:
        pass

    make_move(move, board)
        
def get_square(event):
    x, y = event.pos
    col = x // square_size
    row = y // square_size
    square = chess.square(col, 7-row)
    return square



def main():
    board = chess.Board()
    selected_square = None
    dragging_piece = None
    drag_pos = None
    running = True
    game_on = True
    global t
    t = 0

    print("starting game")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif board.turn == False and not board.is_checkmate():
                move = find_best_move(board, 5, model=model)
                make_move(move, board)


            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board = chess.Board()

            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                print(find_best_move(board, 5))

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                try:
                    if not board.piece_at(selected_square):
                        selected_square = None
                except:
                    pass
                square = get_square(event)
                if board.piece_at(square):
                    dragging_piece = square
                    drag_pos = event.pos
                    if selected_square == None:
                        if str(board.piece_at(square)).islower() and t % 2 == 1:
                            selected_square = square
                        elif str(board.piece_at(square)).isupper() and t % 2 == 0:
                            selected_square = square
                    
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                #print(board.fen())
                if dragging_piece != None:
                    square = get_square(event)
                    if square != selected_square:
                        move = chess.Move(dragging_piece, square)
                        if player_move(board, move, selected_square, square):
                            selected_square = None

                            
                        else:
                            selected_square = None

                    dragging_piece = None
                    drag_pos = None

                elif selected_square != None:
                    square = get_square(event)
                    move = chess.Move(selected_square, square)
                    if player_move(board, move, selected_square, square):
                        selected_square = None
                    elif board.piece_at(square) == None:
                        selected_square = None
                        
            elif event.type == pygame.MOUSEMOTION:
                if dragging_piece != None:
                    drag_pos = event.pos
 
                
        if board.is_checkmate() and game_on == True:
            game_on = False
            sounds["game-end"].play()

        draw_board(selected_square)
        draw_pieces(board, dragging_piece, drag_pos)
        screen.blit(color_turn[turns[t % 2]], textRect)
        pygame.display.flip()



    pygame.quit()

if __name__ == "__main__":
    main()


