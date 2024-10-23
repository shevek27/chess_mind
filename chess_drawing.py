import pygame
import chess
import chess.svg


pygame.init()
# Set up the display
width, height = 600, 600
square_size = width // 8
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('nashe')
font = pygame.font.Font("freesansbold.ttf", 14)

# Load images
pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
piece_images = {}
for piece in pieces:
    piece_images[piece] = pygame.image.load(f"images/{piece}.png")
    piece_images[piece] = pygame.transform.scale(piece_images[piece], (square_size, square_size))

icon = pygame.image.load("images/chess_icon.png")
pygame.display.set_icon(icon)

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
highlight_piece = (186, 202, 68)
turns = ["white_turn", "black_turn"]
color_turn = {}
for turn in turns:
    color_turn[turn] = font.render(f"{turn}", True, black, None)

textRect = color_turn["white_turn"].get_rect()
textRect.center = (width // 10, height // 100)

def draw_pieces(board, dragging_piece, drag_pos):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = piece_images[piece.symbol()]
            row, col = divmod(square, 8)
            if square == dragging_piece:
                screen.blit(piece_image, piece_image.get_rect(center=drag_pos))
            else:
                screen.blit(piece_image, pygame.Rect(col*square_size, (7-row)*square_size, square_size, square_size))

def draw_board(selected_square=None):
    colors = [pygame.Color(238, 238, 210), pygame.Color(118, 150, 86)]
    for row in range(8):
        for col in range(8):
            color = colors[(row+col) % 2]
            if selected_square == chess.square(col, 7-row):
                color = highlight_piece
            pygame.draw.rect(screen, color, pygame.Rect(col*square_size, row*square_size, square_size, square_size))
