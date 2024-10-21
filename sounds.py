import pygame
import chess
import chess.svg

pygame.init()
# Load sounds
pre_sounds = ["move", "capture", "move-check", "castle", "promote", "game-end"]
sounds = {}
for sound in pre_sounds:
    sounds[sound] = pygame.mixer.Sound(f"sounds/{sound}.mp3")

def move_sound(board=None, move=None):

    if board.is_castling(move):
        sounds["castle"].play()
    elif board.gives_check(move):
        sounds["move-check"].play()
    elif board.is_capture(move):
        sounds["capture"].play()
    else:
        sounds["move"].play()