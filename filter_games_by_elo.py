import chess.pgn

def filter_games(pgn_file, output_file, min_elo):
    total_games = 0
    filtered_games = 0
    with open(pgn_file) as games_file:
        with open(output_file, 'w') as out_file:
            while True:
                game = chess.pgn.read_game(games_file)
                if game is None:
                    break

                white_elo = int(game.headers.get("WhiteElo", 0))
                black_elo = int(game.headers.get("BlackElo", 0))
                avg_elo = (white_elo + black_elo) / 2
                total_games += 1
                
                if avg_elo >= min_elo:
                    filtered_games += 1
                    # write game to out file
                    out_file.write(str(game) + "\n\n")
                
                if total_games % 1000 == 0:
                    print(f"total games: {total_games} \nfiltered games: {filtered_games}")
                    print(f"% of filtered games = {((filtered_games * 100) / total_games):.2f}%")

filter_games("lichess_db_standard_rated_2020-10.pgn", "lichess_2020_oct_filtered.pgn", min_elo=2000)