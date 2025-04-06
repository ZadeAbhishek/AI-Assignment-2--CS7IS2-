import random

def baseline_move(game, letter):
    for move in game.available_moves():
        game.make_move(move, letter)
        if game.current_winner == letter:
            game.board[move] = ' '
            game.current_winner = None
            return move
        game.board[move] = ' '
    
    opponent = 'O' if letter == 'X' else 'X'
    for move in game.available_moves():
        game.make_move(move, opponent)
        if game.current_winner == opponent:
            game.board[move] = ' '
            game.current_winner = None
            return move
        game.board[move] = ' '
    return random.choice(game.available_moves())