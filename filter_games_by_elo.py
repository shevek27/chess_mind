import chess.pgn

def filter_games(pgn_file, output_file, min_elo):
    with open(pgn_file) as games_file:
        with open(output_file, 'w') as out_file:
            while True:
                game = chess.pgn.read_game(games_file)
                if game is None:
                    break

                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                avg_elo = (white_elo + black_elo) / 2

                if avg_elo >= min_elo:
                    # write game to out file
                    out_file.write(str(game) + "\n\n")

filter_games("lichess_2020_oct.pgn", "lichess_2020_oct_filtered.pgn", min_elo=2000)