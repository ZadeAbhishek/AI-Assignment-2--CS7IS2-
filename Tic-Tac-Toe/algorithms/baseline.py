import random

def baseline_move(game, letter):
    """
    Baseline strategy: 
      1. If a winning move exists for `letter`, return it.
      2. Otherwise, if the opponent has a winning move, block it.
      3. Otherwise, return a random available move.
    """
    # Check for a winning move for letter.
    for move in game.available_moves():
        game.make_move(move, letter)
        if game.current_winner == letter:
            game.board[move] = ' '  # Undo move.
            game.current_winner = None
            return move
        game.board[move] = ' '  # Undo move.
    
    # Check for blocking opponent's winning move.
    opponent = 'O' if letter == 'X' else 'X'
    for move in game.available_moves():
        game.make_move(move, opponent)
        if game.current_winner == opponent:
            game.board[move] = ' '  # Undo move.
            game.current_winner = None
            return move
        game.board[move] = ' '  # Undo move.
    
    # Otherwise, choose a random available move.
    return random.choice(game.available_moves())