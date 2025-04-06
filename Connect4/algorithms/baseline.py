import random

def undo_move(game, col):
    for row in range(game.rows):
        if game.board[row][col] != ' ':
            game.board[row][col] = ' '
            game.current_winner = None
            break

def baseline_move_connect4(game, letter):
    for move in game.available_moves():
        game.make_move(move, letter)
        if game.current_winner == letter:
            undo_move(game, move)
            return move
        undo_move(game, move)

    opponent = 'O' if letter == 'X' else 'X'
    for move in game.available_moves():
        game.make_move(move, opponent)
        if game.current_winner == opponent:
            undo_move(game, move)
            return move
        undo_move(game, move)
    return random.choice(game.available_moves())