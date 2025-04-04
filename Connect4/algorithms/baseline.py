import random

def undo_move(game, col):
    """Undo the topmost move in the given column."""
    for row in range(game.rows):
        if game.board[row][col] != ' ':
            game.board[row][col] = ' '
            game.current_winner = None
            break

def baseline_move_connect4(game, letter):
    """
    Baseline strategy for Connect4:
      1. If a winning move exists for `letter`, return it.
      2. Else, if the opponent has a winning move, block it.
      3. Otherwise, choose a random available move.
    """
    # Check for winning move.
    for move in game.available_moves():
        game.make_move(move, letter)
        if game.current_winner == letter:
            undo_move(game, move)
            return move
        undo_move(game, move)
    
    # Block opponent's winning move.
    opponent = 'O' if letter == 'X' else 'X'
    for move in game.available_moves():
        game.make_move(move, opponent)
        if game.current_winner == opponent:
            undo_move(game, move)
            return move
        undo_move(game, move)
    
    # Otherwise, choose a random move.
    return random.choice(game.available_moves())